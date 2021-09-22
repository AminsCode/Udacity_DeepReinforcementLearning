## DDPG Algorithm

In this project we use Algorithm DDPG (Deep Deterministic Policy Gradient). Deep Deterministic Policy Gradient (DDPG) is an algorithm which concurrently learns a Q-function and a policy. It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.

### Quick Facts

- DDPG is an off-policy algorithm.
- DDPG can only be used for environments with continuous action spaces.
- DDPG can be thought of as being deep Q-learning for continuous action spaces.
- The Spinning Up implementation of DDPG does not support parallelization.


### Exploration vs. Exploitation

DDPG trains a deterministic policy in an off-policy way. Because the policy is deterministic, if the agent were to explore on-policy, in the beginning it would probably not try a wide enough variety of actions to find useful learning signals. To make DDPG policies explore better, we add noise to their actions at training time. The authors of the original DDPG paper recommended time-correlated OU noise, but more recent results suggest that uncorrelated, mean-zero Gaussian noise works perfectly well. Since the latter is simpler, it is preferred. To facilitate getting higher-quality training data, you may reduce the scale of the noise over the course of training. (We do not do this in our implementation, and keep noise scale fixed throughout.)

### Pseudocode: 

![image](https://user-images.githubusercontent.com/89017449/134413732-fbc1bd5d-1bf1-4a07-a124-dcc28ee474ac.png)

### Implementation of DDPG Agent

The environment for this project involves controlling a double-jointed arm, to reach target locations.
A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of
this agent is to maintain its position at the target location for as many time steps as possible.

The observation space (i.e., state space) has 33 dimensions corresponding to position, rotation, velocity,
and angular velocities of the arm. The action space has 4 dimensions corresponding to torque applicable to
two joints. Every entry in the action vector should be a number between -1 and 1.

### Target networks

The target network used for slow tracking of the learned network. We create a copy of the actor and critic networks:
actor_target (say, with the parameter vector p') and critic_target (say, with the parameter vector w'). The weights of
these target networks are updated by having them the following track:

    p'  <--  p * \tau + p' * (1 - \tau)  
    w'  <--  w * \tau + w' * (1 - \tau)

We put the very small value for \tau (= 0.001). This means that the target values are constrained to change slowly, greatly improving the stability of learning. This update is performed by function soft_update.

"This may slow learning, since the target network delays the propagation of value estimations.
However, in practice we found this was greatly outweighed by the stability of learning."
("Continuous control with deep reinforcement learning", Lillicrap et al.,2015, arXiv:1509.02971)

### DDPG Neural Networks

The DDPG algorithm uses 4 neural networks: actor_target, actor_local, critic_target and critic_local:

    actor_local = Actor(state_size, action_size, random_seed).to(device)
    actor_target = Actor(state_size, action_size, random_seed).to(device)

    critic_local = Critic(state_size, action_size, random_seed).to(device)
    critic_target = Critic(state_size, action_size, random_seed).to(device)

classes Actor and Critic are provided by model.py. The typical behavior of the actor and the critic is as follows:

    actor_target(state) -> action
    critic_target(state, action) -> Q-value

    actor_local(states) -> actions_pred
    -critic_local(states, actions_pred) -> actor_loss


### Update critic_local neural network 
 1. Get predicted next-state actions and Q-values from the actor and critic target neural networks.
    actions_next = actor_target(next_states)
    Q_targets_next = critic_target(next_states, actions_next)

 2. Compute Q-targets by for current states (by Bellman equation)
    Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

 3. Compute Q_expected and critic loss
    Q_expected = critic_local(states, actions)
    critic_loss = MSE_loss(Q_expected, Q_targets)

 4. Minimize the critic_loss. By Gradient Descent and Backward Propogation the weights of 
    the critic_local network are updated. 

### Update actor_local neural network 

1. Compute actor loss
    actions_pred = actor_local(states)
    actor_loss = -critic_local(states, actions_pred).mean()

2. Minimize the actor_loss. By Gradient Descent and Backward Propogation the weights of 
   the actor_local network are updated.

See method learn() in ddpg_agent.py

### Architecture of the actor and critic networks

Both the actor and critic classes implement the neural network
with 3 fully-connected layers and 2 rectified nonlinear layers. These networks are realized in the framework
of package PyTorch. Such a network is used in Udacity model.py code for the Pendulum model using DDPG.
The number of neurons of the fully-connected layers are as follows:

for the actor:

    Layer fc1, number of neurons: state_size x fc1_units,
    Layer fc2, number of neurons: fc1_units x fc2_units,
    Layer fc3, number of neurons: fc2_units x action_size,

for the critic:

    Layer fcs1, number of neurons: state_size x fcs1_units,
    Layer fc2, number of neurons: (fcs1_units+action_size) x fc2_units,
    Layer fc3, number of neurons: fc2_units x 1.

Here, state_size = 33, action_size = 4. The input parameters fc1_units, fc2_units, fcs1_units are all taken = 128.


For more information the following used source would help: [Deep Deterministic Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)

### Hyperparameters:


### Training the Agent:



### Ideas for improvement:

- Using Prioritized Replay (paper) has generally shown to have been quite useful. It is expected that it'll lead to an improved performance here too.

- Other algorithms like TRPO, [PPO](https://openai.com/blog/openai-baselines-ppo/), [A3C](https://openai.com/blog/openai-baselines-ppo/), A2C that have been discussed in the course could potentially lead to better results as well.

- The Q-prop algorithm, which combines both off-policy and on-policy learning, could be good one to try.

- General optimization techniques like cyclical learning rates and warm restarts could be useful as well.
