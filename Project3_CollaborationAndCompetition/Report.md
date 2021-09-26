## Collaboration and Competition

### DDPG and MADDPG Algorithms
In this project, we use the DDPG algorithm (Deep Deterministic Policy Gradient) and the MADDPG algorithm,
a wrapper for DDPG. MADDPG stands for Multi-Agent DDPG. DDPG is an algorithm which concurrently learns
a Q-function and a policy. It uses off-policy data and the Bellman equation to learn the Q-function, and uses
the Q-function to learn the policy. This dual mechanism is the actor-critic method. The DDPG algorithm uses
two additional mechanisms: Replay Buffer and Soft Updates. The reasons of choosing this algorithm to solve the given problem is described below.

### Implement Learning Algorithm
To get started, there are a few high-level architecture decisions we need to make. First, we need to determine which types of algorithms are most suitable for the Tennis environment.

#### Policy-based vs Value-based Methods
There are two key differences in the Tennis environment compared to the 'Navigation' environment from two projects ago:
1. **Multiple agents** &mdash; The Tennis environment has 2 different agents, whereas the Navigation project had only a single agent.
2. **Continuous action space** &mdash; The action space is now _continuous_, which allows each agent to execute more complex and precise movements. Even though each tennis agent can only move forward, backward, or jump, there's an unlimited range of possible action values that control these movements. Whereas, the agent in the Navigation project was limited to four _discrete_ actions: left, right, forward, backward.

Given the additional complexity of this environment, the **value-based method** we used for the Navigation project is not suitable &mdash; i.e., the Deep Q-Network (DQN) algorithm. Most importantly, we need an algorithm that allows the tennis agent to utilize its full range and power of movement. For this, we'll need to explore a different class of algorithms called **policy-based methods**.

Here are some advantages of policy-based methods:
- **Continuous action spaces** &mdash; Policy-based methods are well-suited for continuous action spaces.
- **Stochastic policies** &mdash; Both value-based and policy-based methods can learn deterministic policies. However, policy-based methods can also learn true stochastic policies.
- **Simplicity** &mdash; Policy-based methods directly learn the optimal policy, without having to maintain a separate value function estimate. With value-based methods, the agent uses its experience with the environment to maintain an estimate of the optimal action-value function, from which an optimal policy is derived. This intermediate step requires the storage of lots of additional data since you need to account for all possible action values. Even if you discretize the action space, the number of possible actions can get quite large. And, using DQN to determine the action that maximizes the action-value function within a continuous or high-dimensional space requires a complex optimization process at every timestep.


#### Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
The original DDPG algorithm from which has been extended to create the MADDPG version, is outlined in [this paper](https://arxiv.org/pdf/1509.02971.pdf), _Continuous Control with Deep Reinforcement Learning_, by researchers at Google Deepmind. In this paper, the authors present "a model-free, off-policy actor-critic algorithm using deep function approximators that can learn policies in high-dimensional, continuous action spaces." They highlight that DDPG can be viewed as an extension of Deep Q-learning to continuous tasks.

For the DDPG foundation, [this single-agent DDPG](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) has been used as a template. Then, to make this algorithm suitable for the multiple competitive agents in the Tennis environment, I implemented components discussed in [this paper](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf), _Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments_, by Lowe and Wu, along with other researchers from OpenAI, UC Berkeley, and McGill University. Most notable, I implemented their variation of the actor-critic method (see Figure 1), which I discuss in the following section.

Lastly, I further experimented with components of the DDPG algorithm based on other concepts covered in Udacity's classroom and lessons. My implementation of this algorithm (including various customizations) are discussed below.


#### Actor-Critic Method
Actor-critic methods leverage the strengths of both policy-based and value-based methods.

Using a policy-based approach, the agent (actor) learns how to act by directly estimating the optimal policy and maximizing reward through gradient ascent. Meanwhile, employing a value-based approach, the agent (critic) learns how to estimate the value (i.e., the future cumulative reward) of different state-action pairs. Actor-critic methods combine these two approaches in order to accelerate the learning process. Actor-critic agents are also more stable than value-based agents, while requiring fewer training samples than policy-based agents.

What makes this implementation unique is the **decentralized actor with centralized critic** approach from [the paper by Lowe and Wu](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf). Whereas traditional actor-critic methods have a separate critic for each agent, this approach utilizes a single critic that receives as input the actions and state observations from all agents. This extra information makes training easier and allows for centralized training with decentralized execution. Each agent still takes actions based on its own unique observations of the environment.

You can find the actor-critic logic implemented as part of the `Agent()` class in `maddpg_agent.py` of the source code. The actor-critic models can be found via their respective `Actor()` and `Critic()` classes in `models.py`.

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

In the `Navigation` project, I addressed this by implementing an [𝛆-greedy algorithm]. This algorithm allows the agent to systematically manage the exploration vs. exploitation trade-off. The agent "explores" by picking a random action with some probability epsilon `𝛜`. Meanwhile, the agent continues to "exploit" its knowledge of the environment by choosing actions based on the deterministic policy with probability (1-𝛜).

However, this approach won't work for controlling the tennis agents. The reason is that the actions are no longer a discrete set of simple directions (i.e., up, down, left, right). The actions driving the movement of the arm are forces with different magnitudes and directions. If we base our exploration mechanism on random uniform sampling, the direction actions would have a mean of zero, in turn canceling each other out. This can cause the system to oscillate without making much progress.

Instead, we'll use the **Ornstein-Uhlenbeck process**, as suggested in the previously mentioned [paper by Google DeepMind](https://arxiv.org/pdf/1509.02971.pdf) (see bottom of page 4). The Ornstein-Uhlenbeck process adds a certain amount of noise to the action values at each timestep. This noise is correlated to previous noise and therefore tends to stay in the same direction for longer durations without canceling itself out. This allows the agent to maintain velocity and explore the action space with more continuity.

You can find the Ornstein-Uhlenbeck process implemented in the `OUNoise` class in `maddpg_agent.py` of the source code.

In total, there are five hyperparameters related to this noise process.

The Ornstein-Uhlenbeck process itself has three hyperparameters that determine the noise characteristics and magnitude:
- mu: the long-running mean
- theta: the speed of mean reversion
- sigma: the volatility parameter

Of these, I only tuned sigma. After running a few experiments, I reduced sigma from 0.3 to 0.2. The reduced noise volatility seemed to help the model converge.

Notice also there's an epsilon parameter used to decay the noise level over time. This decay mechanism ensures that more noise is introduced earlier in the training process (i.e., higher exploration), and the noise decreases over time as the agent gains more experience (i.e., higher exploitation). The starting value for epsilon and its decay rate are two hyperparameters that were tuned during experimentation.

You can find the epsilon decay process implemented in the `Agent.act()` method in `maddpg_agent.py` of the source code. While the epsilon decay is performed as part of the learning step.

The final noise parameters were set as follows:

```python
OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter, volatility
OU_THETA = 0.15          # Ornstein-Uhlenbeck noise parameter, speed of mean reversion
EPS_START = 5.0         # initial value for epsilon in noise decay process in Agent.act()
EPS_EP_END = 300        # episode to end the noise decay process
EPS_FINAL = 0           # final value for epsilon after decay
```

#### Learning Interval
In the first few versions of my implementation, the agent only performed a single learning iteration per episode. Although the best model had this setting, this seemed to be a stroke of luck. In general, I found that performing multiple learning passes per episode yielded faster convergence and higher scores. This did make training slower, but it was a worthwhile trade-off. In the end, I implemented an interval in which the learning step is performed every episode. As part of each learning step, the algorithm then samples experiences from the buffer and runs the `Agent.learn()` method 10 times.

```python
LEARN_EVERY = 1         # learning interval (no. of episodes)
LEARN_NUM = 5           # number of passes per learning step
```

You can find the learning interval implemented in the `Agent.step()` method in `maddpg_agent.py` of the source code.

#### Gradient Clipping
In early versions of my implementation, I had trouble getting my agent to learn. Or, rather, it would start to learn but then become very unstable and either plateau or collapse.

I suspect that one of the causes was outsized gradients. Unfortunately, I couldn't find an easy way to investigate this, although I'm sure there's some way of doing this in PyTorch. Absent this investigation, I hypothesize that many of the weights from my critic model were becoming quite large after just 50-100 episodes of training. And since I was running the learning process multiple times per episode, it only made the problem worse.

The issue of exploding gradients is described in layman's terms in [this post](https://machinelearningmastery.com/exploding-gradients-in-neural-networks/) by Jason Brownlee. Essentially, each layer of your net amplifies the gradient it receives. This becomes a problem when the lower layers of the network accumulate huge gradients, making their respective weight updates too large to allow the model to learn anything.

To combat this, I implemented gradient clipping using the `torch.nn.utils.clip_grad_norm_` function. I set the function to "clip" the norm of the gradients at 1, therefore placing an upper limit on the size of the parameter updates, and preventing them from growing exponentially. Once this change was implemented, along with batch normalization (discussed in the next section), my model became much more stable and my agent started learning at a much faster rate.

You can find gradient clipping implemented in the "update critic" section of the `Agent.learn()` method, within `ddpg_agent.py` of the source code.

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

As with the project `Continuous-Control`, the algorithm employs a replay buffer to gather experiences. Experiences are stored in a single replay buffer as each agent interacts with the environment. These experiences are then utilized by the central critic, therefore allowing the agents to learn from each others' experiences.

The replay buffer contains a collection of experience tuples with the state, action, reward, and next state `(s, a, r, s')`. The critic samples from this buffer as part of the learning step. Experiences are sampled randomly, so that the data is uncorrelated. This prevents action values from oscillating or diverging catastrophically, since a naive algorithm could otherwise become biased by correlations between sequential experience tuples.

Also, experience replay improves learning through repetition. By doing multiple passes over the data, our agents have multiple opportunities to learn from a single experience tuple. This is particularly useful for state-action pairs that occur infrequently within the environment.

The implementation of the replay buffer can be found in the `maddpg_agent.py` file of the source code.

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

The graph below shows the final training results. The agents were able to solve the environment in 1300 episodes, with a top score of 4.00. The complete set of results and steps is listed below:

          Episodes 0000-0010	Max Score: 0.000	Average Score: 0.000
     Episodes 0010-0020	Max Score: 0.100	Average Score: 0.005
     Episodes 0020-0030	Max Score: 0.100	Average Score: 0.007
     Episodes 0030-0040	Max Score: 0.000	Average Score: 0.005
     Episodes 0040-0050	Max Score: 0.100	Average Score: 0.006
     Episodes 0050-0060	Max Score: 0.200	Average Score: 0.008
     Episodes 0060-0070	Max Score: 0.100	Average Score: 0.010
     Episodes 0070-0080	Max Score: 0.100	Average Score: 0.010
     Episodes 0080-0090	Max Score: 0.100	Average Score: 0.010
     Episodes 0090-0100	Max Score: 0.100	Average Score: 0.011
     Episodes 0100-0110	Max Score: 0.000	Average Score: 0.011
     Episodes 0110-0120	Max Score: 0.000	Average Score: 0.010
     Episodes 0120-0130	Max Score: 0.000	Average Score: 0.009
     Episodes 0130-0140	Max Score: 0.100	Average Score: 0.010
     Episodes 0140-0150	Max Score: 0.200	Average Score: 0.014
     Episodes 0150-0160	Max Score: 0.100	Average Score: 0.016
     Episodes 0160-0170	Max Score: 0.100	Average Score: 0.016
     Episodes 0170-0180	Max Score: 0.100	Average Score: 0.016
     Episodes 0180-0190	Max Score: 0.200	Average Score: 0.023
     Episodes 0190-0200	Max Score: 0.200	Average Score: 0.024
     Episodes 0200-0210	Max Score: 0.300	Average Score: 0.031
     Episodes 0210-0220	Max Score: 0.100	Average Score: 0.035
     Episodes 0220-0230	Max Score: 0.100	Average Score: 0.039
     Episodes 0230-0240	Max Score: 0.300	Average Score: 0.048
     Episodes 0240-0250	Max Score: 0.200	Average Score: 0.050
     Episodes 0250-0260	Max Score: 0.200	Average Score: 0.054
     Episodes 0260-0270	Max Score: 0.100	Average Score: 0.054
     Episodes 0270-0280	Max Score: 0.200	Average Score: 0.062
     Episodes 0280-0290	Max Score: 0.200	Average Score: 0.059
     Episodes 0290-0300	Max Score: 0.200	Average Score: 0.065
     Episodes 0300-0310	Max Score: 0.300	Average Score: 0.067
     Episodes 0310-0320	Max Score: 0.300	Average Score: 0.073
     Episodes 0320-0330	Max Score: 0.200	Average Score: 0.075
     Episodes 0330-0340	Max Score: 0.300	Average Score: 0.075
     Episodes 0340-0350	Max Score: 0.200	Average Score: 0.075
     Episodes 0350-0360	Max Score: 0.200	Average Score: 0.078
     Episodes 0360-0370	Max Score: 0.200	Average Score: 0.084
     Episodes 0370-0380	Max Score: 0.200	Average Score: 0.082
     Episodes 0380-0390	Max Score: 0.200	Average Score: 0.084
     Episodes 0390-0400	Max Score: 0.300	Average Score: 0.090
     Episodes 0400-0410	Max Score: 0.200	Average Score: 0.089
     Episodes 0410-0420	Max Score: 0.300	Average Score: 0.092
     Episodes 0420-0430	Max Score: 0.200	Average Score: 0.097
     Episodes 0430-0440	Max Score: 0.200	Average Score: 0.097
     Episodes 0440-0450	Max Score: 0.500	Average Score: 0.105
     Episodes 0450-0460	Max Score: 0.200	Average Score: 0.106
     Episodes 0460-0470	Max Score: 0.400	Average Score: 0.110
     Episodes 0470-0480	Max Score: 0.200	Average Score: 0.110
     Episodes 0480-0490	Max Score: 0.300	Average Score: 0.117
     Episodes 0490-0500	Max Score: 0.200	Average Score: 0.109
     Episodes 0500-0510	Max Score: 0.300	Average Score: 0.114
     Episodes 0510-0520	Max Score: 0.300	Average Score: 0.115
     Episodes 0520-0530	Max Score: 0.400	Average Score: 0.122
     Episodes 0530-0540	Max Score: 0.400	Average Score: 0.124
     Episodes 0540-0550	Max Score: 0.300	Average Score: 0.125
     Episodes 0550-0560	Max Score: 0.200	Average Score: 0.124
     Episodes 0560-0570	Max Score: 0.400	Average Score: 0.128
     Episodes 0570-0580	Max Score: 0.400	Average Score: 0.136
     Episodes 0580-0590	Max Score: 0.200	Average Score: 0.132
     Episodes 0590-0600	Max Score: 0.600	Average Score: 0.145
     Episodes 0600-0610	Max Score: 0.500	Average Score: 0.147
     Episodes 0610-0620	Max Score: 0.400	Average Score: 0.149
     Episodes 0620-0630	Max Score: 0.300	Average Score: 0.144
     Episodes 0630-0640	Max Score: 0.700	Average Score: 0.148
     Episodes 0640-0650	Max Score: 0.800	Average Score: 0.162
     Episodes 0650-0660	Max Score: 0.900	Average Score: 0.177
     Episodes 0660-0670	Max Score: 0.500	Average Score: 0.183
     Episodes 0670-0680	Max Score: 0.400	Average Score: 0.176
     Episodes 0680-0690	Max Score: 0.100	Average Score: 0.169
     Episodes 0690-0700	Max Score: 0.100	Average Score: 0.153
     Episodes 0700-0710	Max Score: 0.100	Average Score: 0.139
     Episodes 0710-0720	Max Score: 0.100	Average Score: 0.126
     Episodes 0720-0730	Max Score: 0.100	Average Score: 0.116
     Episodes 0730-0740	Max Score: 0.100	Average Score: 0.105
     Episodes 0740-0750	Max Score: 0.100	Average Score: 0.077
     Episodes 0750-0760	Max Score: 0.100	Average Score: 0.055
     Episodes 0760-0770	Max Score: 0.100	Average Score: 0.037
     Episodes 0770-0780	Max Score: 0.100	Average Score: 0.034
     Episodes 0780-0790	Max Score: 0.100	Average Score: 0.036
     Episodes 0790-0800	Max Score: 0.100	Average Score: 0.035
     Episodes 0800-0810	Max Score: 0.100	Average Score: 0.038
     Episodes 0810-0820	Max Score: 0.100	Average Score: 0.036
     Episodes 0820-0830	Max Score: 0.100	Average Score: 0.038
     Episodes 0830-0840	Max Score: 0.100	Average Score: 0.035
     Episodes 0840-0850	Max Score: 0.100	Average Score: 0.035
     Episodes 0850-0860	Max Score: 0.100	Average Score: 0.037
     Episodes 0860-0870	Max Score: 0.100	Average Score: 0.039
     Episodes 0870-0880	Max Score: 0.100	Average Score: 0.042
     Episodes 0880-0890	Max Score: 0.190	Average Score: 0.044
     Episodes 0890-0900	Max Score: 0.200	Average Score: 0.048
     Episodes 0900-0910	Max Score: 0.200	Average Score: 0.053
     Episodes 0910-0920	Max Score: 0.300	Average Score: 0.059
     Episodes 0920-0930	Max Score: 0.200	Average Score: 0.063
     Episodes 0930-0940	Max Score: 0.200	Average Score: 0.065
     Episodes 0940-0950	Max Score: 0.200	Average Score: 0.071
     Episodes 0950-0960	Max Score: 0.200	Average Score: 0.068
     Episodes 0960-0970	Max Score: 0.200	Average Score: 0.068
     Episodes 0970-0980	Max Score: 0.200	Average Score: 0.067
     Episodes 0980-0990	Max Score: 0.200	Average Score: 0.066
     Episodes 0990-1000	Max Score: 0.400	Average Score: 0.068
     Episodes 1000-1010	Max Score: 0.200	Average Score: 0.062
     Episodes 1010-1020	Max Score: 0.200	Average Score: 0.062
     Episodes 1020-1030	Max Score: 0.200	Average Score: 0.057
     Episodes 1030-1040	Max Score: 0.400	Average Score: 0.070
     Episodes 1040-1050	Max Score: 0.200	Average Score: 0.073
     Episodes 1050-1060	Max Score: 0.200	Average Score: 0.078
     Episodes 1060-1070	Max Score: 0.400	Average Score: 0.085
     Episodes 1070-1080	Max Score: 0.800	Average Score: 0.089
     Episodes 1080-1090	Max Score: 0.700	Average Score: 0.100
     Episodes 1090-1100	Max Score: 0.300	Average Score: 0.108
     Episodes 1100-1110	Max Score: 0.200	Average Score: 0.115
     Episodes 1110-1120	Max Score: 0.400	Average Score: 0.125
     Episodes 1120-1130	Max Score: 0.700	Average Score: 0.136
     Episodes 1130-1140	Max Score: 0.400	Average Score: 0.139
     Episodes 1140-1150	Max Score: 0.200	Average Score: 0.136
     Episodes 1150-1160	Max Score: 0.300	Average Score: 0.143
     Episodes 1160-1170	Max Score: 0.200	Average Score: 0.141
     Episodes 1170-1180	Max Score: 0.600	Average Score: 0.143
     Episodes 1180-1190	Max Score: 0.800	Average Score: 0.155
     Episodes 1190-1200	Max Score: 0.600	Average Score: 0.159
     Episodes 1200-1210	Max Score: 0.600	Average Score: 0.170
     Episodes 1210-1220	Max Score: 0.400	Average Score: 0.172
     Episodes 1220-1230	Max Score: 0.400	Average Score: 0.180
     Episodes 1230-1240	Max Score: 0.900	Average Score: 0.188
     Episodes 1240-1250	Max Score: 0.400	Average Score: 0.201
     Episodes 1250-1260	Max Score: 0.600	Average Score: 0.204
     Episodes 1260-1270	Max Score: 0.600	Average Score: 0.211
     Episodes 1270-1280	Max Score: 0.500	Average Score: 0.221
     Episodes 1280-1290	Max Score: 0.800	Average Score: 0.215
     Episodes 1290-1300	Max Score: 0.300	Average Score: 0.212
     Episodes 1300-1310	Max Score: 0.800	Average Score: 0.215
     Episodes 1310-1320	Max Score: 0.600	Average Score: 0.221
     Episodes 1320-1330	Max Score: 1.100	Average Score: 0.227
     Episodes 1330-1340	Max Score: 0.700	Average Score: 0.231
     Episodes 1340-1350	Max Score: 0.700	Average Score: 0.238
     Episodes 1350-1360	Max Score: 1.900	Average Score: 0.272
     Episodes 1360-1370	Max Score: 0.900	Average Score: 0.280
     Episodes 1370-1380	Max Score: 3.800	Average Score: 0.347
     Episodes 1380-1390	Max Score: 2.600	Average Score: 0.428
     Episodes 1390-1400	Max Score: 4.000	Average Score: 0.521
      ------Environment solved in 1300 episodes!------             
      ------Average Score: 0.521 over past 100 episodes------ 

![image](https://user-images.githubusercontent.com/89017449/134810355-090a16cf-6334-4ed6-b3a9-d96cc3310991.png)


### Future ideas:

- Address stability issues to produce more consistent results: The results are only reproducible if you run the model numerous times. If you just run it once (or even 3-5 times) the model might not converge. More research is needed to find a more stable algorithm, or to make changes to the current DDPG algorithm.
- Add prioritized experience replay — Rather than selecting experience tuples randomly, prioritized replay selects experiences based on a priority value that is correlated with the magnitude of error. This can improve learning by increasing the probability that rare or important experience vectors are sampled.
- Check different values for hyperparameters such as LEARNING_PERIOD, and neural network parameters fc1_units, fc2_units, etc.
- How does the addition of new nonlinear layers in the used neural networks affect the robustness of the algorithm.
- It would be interesting to train agents using MAPPO and compare them with MADDPG.
