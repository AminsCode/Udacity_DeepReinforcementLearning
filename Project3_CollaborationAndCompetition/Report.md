## Collaboration and Competition

### DDPG and MADDPG Algorithms
In this project, we use the DDPG algorithm (Deep Deterministic Policy Gradient) and the MADDPG algorithm,
a wrapper for DDPG. MADDPG stands for Multi-Agent DDPG. DDPG is an algorithm which concurrently learns
a Q-function and a policy. It uses off-policy data and the Bellman equation to learn the Q-function, and uses
the Q-function to learn the policy. This dual mechanism is the actor-critic method. The DDPG algorithm uses
two additional mechanisms: Replay Buffer and Soft Updates.

In MADDPG, we train two separate agents, and the agents need to collaborate (like don’t let the ball hit the ground)
and compete (like gather as many points as possible). Just doing a simple extension of single agent RL
by independently training the two agents does not work very well because the agents are independently updating
their policies as learning progresses. And this causes the environment to appear non-stationary from the viewpoint
of any one agent.

In MADDPG, each agent’s critic is trained using the observations and actions from both agents , whereas
each agent’s actor is trained using just its own observations.

In the finction step() of the class madppg_agent, we collect all current info for both agents into the common variable
memory of the type ReplayBuffer. Then we get the random sample from memory into the variable experiance.
This experiance together with the current number of agent (0 or 1) go to the function learn(). We get the corresponding
agent (of type ddpg_agent):

     agent = self.agents[agent_number]

and experiance is transferred to function learn() of the class ddpg_agent. There, the actor and the critic are handled by different ways.


### Implement Learning Algorithm
To get started, there are a few high-level architecture decisions we need to make. First, we need to determine which types of algorithms are most suitable for the Tennis environment.

#### Policy-based vs Value-based Methods
There are two key differences in the Tennis environment compared to the ['Navigation'](https://github.com/tommytracey/DeepRL-P1-Navigation) environment from two projects ago:
1. **Multiple agents** &mdash; The Tennis environment has 2 different agents, whereas the Navigation project had only a single agent.
2. **Continuous action space** &mdash; The action space is now _continuous_, which allows each agent to execute more complex and precise movements. Even though each tennis agent can only move forward, backward, or jump, there's an unlimited range of possible action values that control these movements. Whereas, the agent in the Navigation project was limited to four _discrete_ actions: left, right, forward, backward.

Given the additional complexity of this environment, the **value-based method** we used for the Navigation project is not suitable &mdash; i.e., the Deep Q-Network (DQN) algorithm. Most importantly, we need an algorithm that allows the tennis agent to utilize its full range and power of movement. For this, we'll need to explore a different class of algorithms called **policy-based methods**.

Here are some advantages of policy-based methods:
- **Continuous action spaces** &mdash; Policy-based methods are well-suited for continuous action spaces.
- **Stochastic policies** &mdash; Both value-based and policy-based methods can learn deterministic policies. However, policy-based methods can also learn true stochastic policies.
- **Simplicity** &mdash; Policy-based methods directly learn the optimal policy, without having to maintain a separate value function estimate. With value-based methods, the agent uses its experience with the environment to maintain an estimate of the optimal action-value function, from which an optimal policy is derived. This intermediate step requires the storage of lots of additional data since you need to account for all possible action values. Even if you discretize the action space, the number of possible actions can get quite large. And, using DQN to determine the action that maximizes the action-value function within a continuous or high-dimensional space requires a complex optimization process at every timestep.


#### Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
The original DDPG algorithm from which I extended to create the MADDPG version, is outlined in [this paper](https://arxiv.org/pdf/1509.02971.pdf), _Continuous Control with Deep Reinforcement Learning_, by researchers at Google Deepmind. In this paper, the authors present "a model-free, off-policy actor-critic algorithm using deep function approximators that can learn policies in high-dimensional, continuous action spaces." They highlight that DDPG can be viewed as an extension of Deep Q-learning to continuous tasks.

For the DDPG foundation, I used [this vanilla, single-agent DDPG](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) as a template. Then, to make this algorithm suitable for the multiple competitive agents in the Tennis environment, I implemented components discussed in [this paper](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf), _Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments_, by Lowe and Wu, along with other researchers from OpenAI, UC Berkeley, and McGill University. Most notable, I implemented their variation of the actor-critic method (see Figure 1), which I discuss in the following section.

Lastly, I further experimented with components of the DDPG algorithm based on other concepts covered in Udacity's classroom and lessons. My implementation of this algorithm (including various customizations) are discussed below.

<img src="assets/multi-agent-actor-critic.png" width="40%" align="top-left" alt="" title="Multi-Agent Actor-Critic" />

> _Figure 1: Multi-agent decentralized actor with centralized critic ([Lowe and Wu et al](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf))._

#### Actor-Critic Method
Actor-critic methods leverage the strengths of both policy-based and value-based methods.

Using a policy-based approach, the agent (actor) learns how to act by directly estimating the optimal policy and maximizing reward through gradient ascent. Meanwhile, employing a value-based approach, the agent (critic) learns how to estimate the value (i.e., the future cumulative reward) of different state-action pairs. Actor-critic methods combine these two approaches in order to accelerate the learning process. Actor-critic agents are also more stable than value-based agents, while requiring fewer training samples than policy-based agents.

What makes this implementation unique is the **decentralized actor with centralized critic** approach from [the paper by Lowe and Wu](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf). Whereas traditional actor-critic methods have a separate critic for each agent, this approach utilizes a single critic that receives as input the actions and state observations from all agents. This extra information makes training easier and allows for centralized training with decentralized execution. Each agent still takes actions based on its own unique observations of the environment.

You can find the actor-critic logic implemented as part of the `Agent()` class [here](https://github.com/tommytracey/DeepRL-P3-Collaboration-Competition/blob/master/maddpg_agent.py#L110) in `maddpg_agent.py` of the source code. The actor-critic models can be found via their respective `Actor()` and `Critic()` classes [here](https://github.com/tommytracey/DeepRL-P3-Collaboration-Competition/blob/master/model.py#L12) in `models.py`.

Note: As we did with Double Q-Learning in the last project, we're again leveraging local and target networks to improve stability. This is where one set of parameters `w` is used to select the best action, and another set of parameters `w'` is used to evaluate that action. In this project, local and target networks are implemented separately for both the actor and the critic.

```python
# Actor Network (w/ Target Network)
self.actor_local = Actor(state_size, action_size, random_seed).to(device)
self.actor_target = Actor(state_size, action_size, random_seed).to(device)
self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

# Critic Network (w/ Target Network)
self.critic_local = Critic(state_size, action_size, random_seed).to(device)
self.critic_target = Critic(state_size, action_size, random_seed).to(device)
self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
```

#### Exploration vs Exploitation
One challenge is choosing which action to take while the agent is still learning the optimal policy. Should the agent choose an action based on the rewards observed thus far? Or, should the agent try a new action in hopes of earning a higher reward? This is known as the **exploration vs. exploitation dilemma**.

In the [Navigation project](https://github.com/tommytracey/DeepRL-P1-Navigation), I addressed this by implementing an [𝛆-greedy algorithm](https://github.com/tommytracey/DeepRL-P1-Navigation/blob/master/agent.py#L80). This algorithm allows the agent to systematically manage the exploration vs. exploitation trade-off. The agent "explores" by picking a random action with some probability epsilon `𝛜`. Meanwhile, the agent continues to "exploit" its knowledge of the environment by choosing actions based on the deterministic policy with probability (1-𝛜).

However, this approach won't work for controlling the tennis agents. The reason is that the actions are no longer a discrete set of simple directions (i.e., up, down, left, right). The actions driving the movement of the arm are forces with different magnitudes and directions. If we base our exploration mechanism on random uniform sampling, the direction actions would have a mean of zero, in turn canceling each other out. This can cause the system to oscillate without making much progress.

Instead, we'll use the **Ornstein-Uhlenbeck process**, as suggested in the previously mentioned [paper by Google DeepMind](https://arxiv.org/pdf/1509.02971.pdf) (see bottom of page 4). The Ornstein-Uhlenbeck process adds a certain amount of noise to the action values at each timestep. This noise is correlated to previous noise and therefore tends to stay in the same direction for longer durations without canceling itself out. This allows the agent to maintain velocity and explore the action space with more continuity.

You can find the Ornstein-Uhlenbeck process implemented [here](https://github.com/tommytracey/DeepRL-P3-Collaboration-Competition/blob/master/maddpg_agent.py#L167) in the `OUNoise` class in `maddpg_agent.py` of the source code.

In total, there are five hyperparameters related to this noise process.

The Ornstein-Uhlenbeck process itself has three hyperparameters that determine the noise characteristics and magnitude:
- mu: the long-running mean
- theta: the speed of mean reversion
- sigma: the volatility parameter

Of these, I only tuned sigma. After running a few experiments, I reduced sigma from 0.3 to 0.2. The reduced noise volatility seemed to help the model converge.

Notice also there's an epsilon parameter used to decay the noise level over time. This decay mechanism ensures that more noise is introduced earlier in the training process (i.e., higher exploration), and the noise decreases over time as the agent gains more experience (i.e., higher exploitation). The starting value for epsilon and its decay rate are two hyperparameters that were tuned during experimentation.

You can find the epsilon decay process implemented [here](https://github.com/tommytracey/DeepRL-P3-Collaboration-Competition/blob/master/maddpg_agent.py#L79) in the `Agent.act()` method in `maddpg_agent.py` of the source code. While the epsilon decay is performed [here](https://github.com/tommytracey/DeepRL-P3-Collaboration-Competition/blob/master/maddpg_agent.py#L150) as part of the learning step.

The final noise parameters were set as follows:

```python
OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter, volatility
OU_THETA = 0.15          # Ornstein-Uhlenbeck noise parameter, speed of mean reversion
EPS_START = 5.0         # initial value for epsilon in noise decay process in Agent.act()
EPS_EP_END = 300        # episode to end the noise decay process
EPS_FINAL = 0           # final value for epsilon after decay
```

**IMPORTANT NOTE:** Notice that the EPS_START parameter is set at 5.0. For dozens of experiments, I had this parameter set to 1.0, as I had in previous projects. But, I had a difficult time getting the model to converge, and if it did, it converged very slowly (>1500 episodes). After much trial and error, I realized that the agents had some difficulty discovering signal early in the process (i.e., most episode scores equaled zero). By boosting the noise output from the Ornstein-Uhlenbeck (OU) process, it encouraged aggressive exploration of the action space and therefore improved the chances that signal would be detected (i.e., making contact with the ball). This extra signal seemed to improve learning later in training once the noise decayed to zero.

#### Learning Interval
In the first few versions of my implementation, the agent only performed a single learning iteration per episode. Although the best model had this setting, this seemed to be a stroke of luck. In general, I found that performing multiple learning passes per episode yielded faster convergence and higher scores. This did make training slower, but it was a worthwhile trade-off. In the end, I implemented an interval in which the learning step is performed every episode. As part of each learning step, the algorithm then samples experiences from the buffer and runs the `Agent.learn()` method 10 times.

```python
LEARN_EVERY = 1         # learning interval (no. of episodes)
LEARN_NUM = 5           # number of passes per learning step
```

You can find the learning interval implemented [here](https://github.com/tommytracey/DeepRL-P3-Collaboration-Competition/blob/master/maddpg_agent.py#L67) in the `Agent.step()` method in `maddpg_agent.py` of the source code.

#### Gradient Clipping
In early versions of my implementation, I had trouble getting my agent to learn. Or, rather, it would start to learn but then become very unstable and either plateau or collapse.

I suspect that one of the causes was outsized gradients. Unfortunately, I couldn't find an easy way to investigate this, although I'm sure there's some way of doing this in PyTorch. Absent this investigation, I hypothesize that many of the weights from my critic model were becoming quite large after just 50-100 episodes of training. And since I was running the learning process multiple times per episode, it only made the problem worse.

The issue of exploding gradients is described in layman's terms in [this post](https://machinelearningmastery.com/exploding-gradients-in-neural-networks/) by Jason Brownlee. Essentially, each layer of your net amplifies the gradient it receives. This becomes a problem when the lower layers of the network accumulate huge gradients, making their respective weight updates too large to allow the model to learn anything.

To combat this, I implemented gradient clipping using the `torch.nn.utils.clip_grad_norm_` function. I set the function to "clip" the norm of the gradients at 1, therefore placing an upper limit on the size of the parameter updates, and preventing them from growing exponentially. Once this change was implemented, along with batch normalization (discussed in the next section), my model became much more stable and my agent started learning at a much faster rate.

You can find gradient clipping implemented [here](https://github.com/tommytracey/DeepRL-P3-Collaboration-Competition/blob/master/maddpg_agent.py#L128) in the "update critic" section of the `Agent.learn()` method, within `ddpg_agent.py` of the source code.

Note that this function is applied after the backward pass, but before the optimization step.

```python
# Compute critic loss
Q_expected = self.critic_local(states, actions)
critic_loss = F.mse_loss(Q_expected, Q_targets)
# Minimize the loss
self.critic_optimizer.zero_grad()
critic_loss.backward()
torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
self.critic_optimizer.step()
```

#### Experience Replay
Experience replay allows the RL agent to learn from past experience.

As with the [previous project](https://github.com/tommytracey/DeepRL-P2-Continuous-Control), the algorithm employs a replay buffer to gather experiences. Experiences are stored in a single replay buffer as each agent interacts with the environment. These experiences are then utilized by the central critic, therefore allowing the agents to learn from each others' experiences.

The replay buffer contains a collection of experience tuples with the state, action, reward, and next state `(s, a, r, s')`. The critic samples from this buffer as part of the learning step. Experiences are sampled randomly, so that the data is uncorrelated. This prevents action values from oscillating or diverging catastrophically, since a naive algorithm could otherwise become biased by correlations between sequential experience tuples.

Also, experience replay improves learning through repetition. By doing multiple passes over the data, our agents have multiple opportunities to learn from a single experience tuple. This is particularly useful for state-action pairs that occur infrequently within the environment.

The implementation of the replay buffer can be found [here](https://github.com/tommytracey/DeepRL-P3-Collaboration-Competition/blob/master/maddpg_agent.py#L196) in the `maddpg_agent.py` file of the source code.

### Hyperparameters

The used Hyperparameters in `maddpg_agent.py` are listed below:

     BUFFER_SIZE = int(1e6)  # replay buffer size
     BATCH_SIZE = 128        # minibatch size
     LR_ACTOR = 1e-3         # learning rate of the actor
     LR_CRITIC = 1e-3        # learning rate of the critic
     WEIGHT_DECAY = 0        # L2 weight decay
     LEARN_EVERY = 1         # learning timestep interval
     LEARN_NUM = 5           # number of learning passes
     GAMMA = 0.99            # discount factor
     TAU = 8e-3              # for soft update of target parameters
     OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter, volatility
     OU_THETA = 0.15          # Ornstein-Uhlenbeck noise parameter, speed of mean reversion
     EPS_START = 5.0         # initial value for epsilon in noise decay process in Agent.act()
     EPS_EP_END = 300        # episode to end the noise decay process
     EPS_FINAL = 0           # final value for epsilon after decay



## Results
Once all of the above components were in place, the agents were able to solve the Tennis environment. Again, the performance goal is an average reward of at least +0.5 over 100 episodes, taking the best score from either agent for a given episode.

The graph below shows the final training results. The agents were able to solve the environment in 1404 episodes, with a top score of 3.3. The complete set of results and steps is listed below:

     Episodes 0000-0010	Max Score: 0.100	Average Score: 0.020
     Episodes 0010-0020	Max Score: 0.000	Average Score: 0.010
     Episodes 0020-0030	Max Score: 0.000	Average Score: 0.007
     Episodes 0030-0040	Max Score: 0.100	Average Score: 0.008
     Episodes 0040-0050	Max Score: 0.000	Average Score: 0.006
     Episodes 0050-0060	Max Score: 0.100	Average Score: 0.008
     Episodes 0060-0070	Max Score: 0.100	Average Score: 0.010
     Episodes 0070-0080	Max Score: 0.000	Average Score: 0.009
     Episodes 0080-0090	Max Score: 0.100	Average Score: 0.010
     Episodes 0090-0100	Max Score: 0.100	Average Score: 0.010
     Episodes 0100-0110	Max Score: 0.100	Average Score: 0.010
     Episodes 0110-0120	Max Score: 0.300	Average Score: 0.014
     Episodes 0120-0130	Max Score: 0.100	Average Score: 0.016
     Episodes 0130-0140	Max Score: 0.200	Average Score: 0.020
     Episodes 0140-0150	Max Score: 0.100	Average Score: 0.022
     Episodes 0150-0160	Max Score: 0.200	Average Score: 0.024
     Episodes 0160-0170	Max Score: 0.100	Average Score: 0.023
     Episodes 0170-0180	Max Score: 0.100	Average Score: 0.025
     Episodes 0180-0190	Max Score: 0.100	Average Score: 0.024
     Episodes 0190-0200	Max Score: 0.100	Average Score: 0.024
     Episodes 0200-0210	Max Score: 0.100	Average Score: 0.023
     Episodes 0210-0220	Max Score: 0.100	Average Score: 0.021
     Episodes 0220-0230	Max Score: 0.500	Average Score: 0.027
     Episodes 0230-0240	Max Score: 0.300	Average Score: 0.026
     Episodes 0240-0250	Max Score: 0.100	Average Score: 0.025
     Episodes 0250-0260	Max Score: 0.200	Average Score: 0.027
     Episodes 0260-0270	Max Score: 0.200	Average Score: 0.034
     Episodes 0270-0280	Max Score: 0.200	Average Score: 0.037
     Episodes 0280-0290	Max Score: 0.200	Average Score: 0.042
     Episodes 0290-0300	Max Score: 0.200	Average Score: 0.046
     Episodes 0300-0310	Max Score: 0.100	Average Score: 0.049
     Episodes 0310-0320	Max Score: 0.000	Average Score: 0.047
     Episodes 0320-0330	Max Score: 0.300	Average Score: 0.043
     Episodes 0330-0340	Max Score: 0.100	Average Score: 0.045
     Episodes 0340-0350	Max Score: 0.100	Average Score: 0.048
     Episodes 0350-0360	Max Score: 0.200	Average Score: 0.053
     Episodes 0360-0370	Max Score: 0.200	Average Score: 0.051
     Episodes 0370-0380	Max Score: 0.100	Average Score: 0.051
     Episodes 0380-0390	Max Score: 0.200	Average Score: 0.055
     Episodes 0390-0400	Max Score: 0.200	Average Score: 0.060
     Episodes 0400-0410	Max Score: 0.100	Average Score: 0.060
     Episodes 0410-0420	Max Score: 0.200	Average Score: 0.070
     Episodes 0420-0430	Max Score: 0.100	Average Score: 0.071
     Episodes 0430-0440	Max Score: 0.100	Average Score: 0.071
     Episodes 0440-0450	Max Score: 0.200	Average Score: 0.073
     Episodes 0450-0460	Max Score: 0.100	Average Score: 0.066
     Episodes 0460-0470	Max Score: 0.200	Average Score: 0.065
     Episodes 0470-0480	Max Score: 0.300	Average Score: 0.072
     Episodes 0480-0490	Max Score: 0.200	Average Score: 0.070
     Episodes 0490-0500	Max Score: 0.100	Average Score: 0.066
     Episodes 0500-0510	Max Score: 0.200	Average Score: 0.071
     Episodes 0510-0520	Max Score: 0.200	Average Score: 0.068
     Episodes 0520-0530	Max Score: 0.200	Average Score: 0.071
     Episodes 0530-0540	Max Score: 0.200	Average Score: 0.072
     Episodes 0540-0550	Max Score: 0.100	Average Score: 0.071
     Episodes 0550-0560	Max Score: 0.200	Average Score: 0.073
     Episodes 0560-0570	Max Score: 0.300	Average Score: 0.081
     Episodes 0570-0580	Max Score: 0.200	Average Score: 0.077
     Episodes 0580-0590	Max Score: 0.300	Average Score: 0.082
     Episodes 0590-0600	Max Score: 0.200	Average Score: 0.088
     Episodes 0600-0610	Max Score: 0.200	Average Score: 0.088
     Episodes 0610-0620	Max Score: 0.600	Average Score: 0.097
     Episodes 0620-0630	Max Score: 0.100	Average Score: 0.097
     Episodes 0630-0640	Max Score: 0.100	Average Score: 0.097
     Episodes 0640-0650	Max Score: 0.200	Average Score: 0.103
     Episodes 0650-0660	Max Score: 0.300	Average Score: 0.108
     Episodes 0660-0670	Max Score: 0.200	Average Score: 0.106
     Episodes 0670-0680	Max Score: 0.100	Average Score: 0.108
     Episodes 0680-0690	Max Score: 0.200	Average Score: 0.108
     Episodes 0690-0700	Max Score: 0.200	Average Score: 0.107
     Episodes 0700-0710	Max Score: 0.100	Average Score: 0.105
     Episodes 0710-0720	Max Score: 0.300	Average Score: 0.103
     Episodes 0720-0730	Max Score: 0.200	Average Score: 0.106
     Episodes 0730-0740	Max Score: 0.300	Average Score: 0.110
     Episodes 0740-0750	Max Score: 0.100	Average Score: 0.107
     Episodes 0750-0760	Max Score: 0.200	Average Score: 0.104
     Episodes 0760-0770	Max Score: 0.200	Average Score: 0.099
     Episodes 0770-0780	Max Score: 0.200	Average Score: 0.096
     Episodes 0780-0790	Max Score: 0.100	Average Score: 0.088
     Episodes 0790-0800	Max Score: 0.300	Average Score: 0.087
     Episodes 0800-0810	Max Score: 0.400	Average Score: 0.093
     Episodes 0810-0820	Max Score: 0.400	Average Score: 0.101
     Episodes 0820-0830	Max Score: 0.100	Average Score: 0.095
     Episodes 0830-0840	Max Score: 0.500	Average Score: 0.107
     Episodes 0840-0850	Max Score: 0.400	Average Score: 0.112
     Episodes 0850-0860	Max Score: 0.300	Average Score: 0.117
     Episodes 0860-0870	Max Score: 0.400	Average Score: 0.127
     Episodes 0870-0880	Max Score: 0.900	Average Score: 0.141
     Episodes 0880-0890	Max Score: 0.600	Average Score: 0.153
     Episodes 0890-0900	Max Score: 0.400	Average Score: 0.160
     Episodes 0900-0910	Max Score: 0.300	Average Score: 0.157
     Episodes 0910-0920	Max Score: 0.900	Average Score: 0.156
     Episodes 0920-0930	Max Score: 1.300	Average Score: 0.179
     Episodes 0930-0940	Max Score: 0.600	Average Score: 0.178
     Episodes 0940-0950	Max Score: 0.700	Average Score: 0.187
     Episodes 0950-0960	Max Score: 0.400	Average Score: 0.184
     Episodes 0960-0970	Max Score: 0.200	Average Score: 0.179
     Episodes 0970-0980	Max Score: 0.700	Average Score: 0.182
     Episodes 0980-0990	Max Score: 0.200	Average Score: 0.177
     Episodes 0990-1000	Max Score: 0.500	Average Score: 0.184
     Episodes 1000-1010	Max Score: 0.700	Average Score: 0.205
     Episodes 1010-1020	Max Score: 0.500	Average Score: 0.214
     Episodes 1020-1030	Max Score: 1.500	Average Score: 0.216
     Episodes 1030-1040	Max Score: 1.500	Average Score: 0.238
     Episodes 1040-1050	Max Score: 0.800	Average Score: 0.241
     Episodes 1050-1060	Max Score: 0.900	Average Score: 0.265
     Episodes 1060-1070	Max Score: 0.700	Average Score: 0.274
     Episodes 1070-1080	Max Score: 0.700	Average Score: 0.274
     Episodes 1080-1090	Max Score: 0.500	Average Score: 0.283
     Episodes 1090-1100	Max Score: 2.100	Average Score: 0.295
     Episodes 1100-1110	Max Score: 1.200	Average Score: 0.305
     Episodes 1110-1120	Max Score: 0.500	Average Score: 0.300
     Episodes 1120-1130	Max Score: 1.300	Average Score: 0.310
     Episodes 1130-1140	Max Score: 2.300	Average Score: 0.314
     Episodes 1140-1150	Max Score: 0.200	Average Score: 0.299
     Episodes 1150-1160	Max Score: 2.000	Average Score: 0.312
     Episodes 1160-1170	Max Score: 0.800	Average Score: 0.321
     Episodes 1170-1180	Max Score: 0.600	Average Score: 0.328
     Episodes 1180-1190	Max Score: 0.500	Average Score: 0.330
     Episodes 1190-1200	Max Score: 0.500	Average Score: 0.315
     Episodes 1200-1210	Max Score: 0.700	Average Score: 0.299
     Episodes 1210-1220	Max Score: 0.800	Average Score: 0.301
     Episodes 1220-1230	Max Score: 1.500	Average Score: 0.311
     Episodes 1230-1240	Max Score: 0.700	Average Score: 0.303
     Episodes 1240-1250	Max Score: 1.200	Average Score: 0.325
     Episodes 1250-1260	Max Score: 0.700	Average Score: 0.307
     Episodes 1260-1270	Max Score: 0.800	Average Score: 0.309
     Episodes 1270-1280	Max Score: 0.500	Average Score: 0.297
     Episodes 1280-1290	Max Score: 0.900	Average Score: 0.309
     Episodes 1290-1300	Max Score: 1.000	Average Score: 0.327
     Episodes 1300-1310	Max Score: 0.900	Average Score: 0.329
     Episodes 1310-1320	Max Score: 0.900	Average Score: 0.321
     Episodes 1320-1330	Max Score: 1.800	Average Score: 0.353
     Episodes 1330-1340	Max Score: 1.300	Average Score: 0.346
     Episodes 1340-1350	Max Score: 1.500	Average Score: 0.344
     Episodes 1350-1360	Max Score: 0.900	Average Score: 0.342
     Episodes 1360-1370	Max Score: 0.900	Average Score: 0.345
     Episodes 1370-1380	Max Score: 0.900	Average Score: 0.357
     Episodes 1380-1390	Max Score: 0.500	Average Score: 0.338
     Episodes 1390-1400	Max Score: 0.200	Average Score: 0.311
     Episodes 1400-1410	Max Score: 0.500	Average Score: 0.303
     Episodes 1410-1420	Max Score: 1.100	Average Score: 0.330
     Episodes 1420-1430	Max Score: 1.300	Average Score: 0.286
     Episodes 1430-1440	Max Score: 0.500	Average Score: 0.275
     Episodes 1440-1450	Max Score: 1.700	Average Score: 0.301
     Episodes 1450-1460	Max Score: 0.900	Average Score: 0.307
     Episodes 1460-1470	Max Score: 1.100	Average Score: 0.317
     Episodes 1470-1480	Max Score: 1.300	Average Score: 0.341
     Episodes 1480-1490	Max Score: 3.300	Average Score: 0.406
     Episodes 1490-1500	Max Score: 2.500	Average Score: 0.481
      ------Environment solved in 1404 episodes!------             
      ------Average Score: 0.501 over past 100 episodes------ 

![image](https://user-images.githubusercontent.com/89017449/134804446-b715bedc-4273-4446-ba73-ca42cf781b08.png)



### Future ideas:

- Check different values for hyperparameters such as LEARNING_PERIOD, and neural network parameters fc1_units, fc2_units, etc.
- How does the addition of new nonlinear layers in the used neural networks affect the robustness of the algorithm.
- It would be interesting to train agents using MAPPO and compare them with MADDPG.
