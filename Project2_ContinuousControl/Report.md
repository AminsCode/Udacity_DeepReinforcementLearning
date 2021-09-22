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

    BUFFER_SIZE = int(1e6)  # replay buffer size
    BATCH_SIZE = 256        # minibatch size
    GAMMA = 0.99            # discount factor
    TAU = 1e-3              # for soft update of target parameters
    LR_ACTOR = 1e-3         # learning rate of the actor
    LR_CRITIC = 1e-3        # learning rate of the critic
    WEIGHT_DECAY = 0        # L2 weight decay
    EPSILON = 1.0           # epsilon noise parameter
    EPSILON_DECAY = 1e-6    # decay parameter of epsilon
    LEARNING_PERIOD = 20    # learning frequency  
    UPDATE_FACTOR   = 10    # how much to learn

### Training the Agent:

        Training started!
        Episode: 1, Score: 0.31, Max: 0.82, Min: 0.00 
        Episode: 2, Score: 0.46, Max: 1.75, Min: 0.00 
        Episode: 3, Score: 0.77, Max: 3.90, Min: 0.03 
        Episode: 4, Score: 0.89, Max: 2.24, Min: 0.00 
        Episode: 5, Score: 0.77, Max: 2.02, Min: 0.00 
        Episode: 6, Score: 0.46, Max: 1.34, Min: 0.00 
        Episode: 7, Score: 0.56, Max: 1.66, Min: 0.00 
        Episode: 8, Score: 0.43, Max: 1.22, Min: 0.00 
        Episode: 9, Score: 0.88, Max: 2.12, Min: 0.16 
        Episode: 10, Score: 0.91, Max: 2.67, Min: 0.00 
        Average Score: 0.64, Time: 00:01:48
        ----------------------------
        Episode: 11, Score: 1.03, Max: 2.30, Min: 0.18 
        Episode: 12, Score: 0.77, Max: 1.77, Min: 0.12 
        Episode: 13, Score: 0.65, Max: 1.70, Min: 0.00 
        Episode: 14, Score: 1.20, Max: 2.29, Min: 0.10 
        Episode: 15, Score: 0.65, Max: 1.53, Min: 0.00 
        Episode: 16, Score: 0.96, Max: 2.09, Min: 0.13 
        Episode: 17, Score: 1.27, Max: 2.28, Min: 0.22 
        Episode: 18, Score: 1.09, Max: 2.68, Min: 0.35 
        Episode: 19, Score: 1.13, Max: 2.52, Min: 0.18 
        Episode: 20, Score: 1.52, Max: 3.19, Min: 0.47 
        Average Score: 0.84, Time: 00:03:49
        ----------------------------
        Episode: 21, Score: 1.53, Max: 3.05, Min: 0.22 
        Episode: 22, Score: 1.19, Max: 2.34, Min: 0.00 
        Episode: 23, Score: 1.61, Max: 3.59, Min: 0.35 
        Episode: 24, Score: 1.52, Max: 2.57, Min: 0.38 
        Episode: 25, Score: 1.74, Max: 3.97, Min: 0.00 
        Episode: 26, Score: 1.48, Max: 4.01, Min: 0.12 
        Episode: 27, Score: 2.18, Max: 4.90, Min: 0.31 
        Episode: 28, Score: 1.80, Max: 3.68, Min: 0.17 
        Episode: 29, Score: 1.60, Max: 2.96, Min: 0.43 
        Episode: 30, Score: 2.33, Max: 4.94, Min: 0.47 
        Average Score: 1.12, Time: 00:06:12
        ----------------------------
        Episode: 31, Score: 1.84, Max: 3.48, Min: 0.13 
        Episode: 32, Score: 2.21, Max: 4.11, Min: 0.66 
        Episode: 33, Score: 2.67, Max: 5.76, Min: 1.33 
        Episode: 34, Score: 2.44, Max: 3.75, Min: 1.21 
        Episode: 35, Score: 2.61, Max: 4.51, Min: 0.61 
        Episode: 36, Score: 2.98, Max: 4.83, Min: 0.98 
        Episode: 37, Score: 2.95, Max: 7.54, Min: 0.52 
        Episode: 38, Score: 3.42, Max: 6.37, Min: 0.22 
        Episode: 39, Score: 3.71, Max: 6.66, Min: 1.38 
        Episode: 40, Score: 3.06, Max: 4.89, Min: 1.90 
        Average Score: 1.54, Time: 00:09:02
        ----------------------------
        Episode: 41, Score: 4.09, Max: 6.47, Min: 1.52 
        Episode: 42, Score: 3.82, Max: 6.37, Min: 1.25 
        Episode: 43, Score: 4.33, Max: 6.17, Min: 2.15 
        Episode: 44, Score: 4.13, Max: 8.91, Min: 2.29 
        Episode: 45, Score: 4.67, Max: 10.16, Min: 0.60 
        Episode: 46, Score: 4.62, Max: 8.15, Min: 1.57 
        Episode: 47, Score: 4.50, Max: 7.19, Min: 1.81 
        Episode: 48, Score: 4.30, Max: 8.08, Min: 0.91 
        Episode: 49, Score: 5.19, Max: 8.64, Min: 2.03 
        Episode: 50, Score: 6.41, Max: 16.78, Min: 2.12 
        Average Score: 2.15, Time: 00:12:16
        ----------------------------
        Episode: 51, Score: 4.99, Max: 8.50, Min: 1.78 
        Episode: 52, Score: 5.20, Max: 10.32, Min: 2.89 
        Episode: 53, Score: 4.85, Max: 8.94, Min: 2.22 
        Episode: 54, Score: 5.42, Max: 7.43, Min: 3.58 
        Episode: 55, Score: 4.86, Max: 7.67, Min: 1.58 
        Episode: 56, Score: 4.44, Max: 8.08, Min: 0.92 
        Episode: 57, Score: 4.89, Max: 7.82, Min: 1.05 
        Episode: 58, Score: 5.09, Max: 9.14, Min: 1.83 
        Episode: 59, Score: 5.45, Max: 8.71, Min: 0.75 
        Episode: 60, Score: 6.33, Max: 12.91, Min: 3.99 
        Average Score: 2.65, Time: 00:15:44
        ----------------------------
        Episode: 61, Score: 8.02, Max: 30.58, Min: 3.80 
        Episode: 62, Score: 6.87, Max: 26.65, Min: 2.99 
        Episode: 63, Score: 7.41, Max: 19.09, Min: 3.70 
        Episode: 64, Score: 5.71, Max: 13.16, Min: 1.17 
        Episode: 65, Score: 10.10, Max: 29.29, Min: 4.60 
        Episode: 66, Score: 7.98, Max: 15.51, Min: 3.25 
        Episode: 67, Score: 7.94, Max: 12.32, Min: 3.26 
        Episode: 68, Score: 9.05, Max: 18.56, Min: 3.38 
        Episode: 69, Score: 9.87, Max: 16.22, Min: 3.36 
        Episode: 70, Score: 10.47, Max: 16.70, Min: 3.03 
        Average Score: 3.47, Time: 00:19:13
        ----------------------------
        Episode: 71, Score: 9.47, Max: 14.10, Min: 5.37 
        Episode: 72, Score: 9.34, Max: 15.61, Min: 4.20 
        Episode: 73, Score: 9.94, Max: 15.83, Min: 5.27 
        Episode: 74, Score: 9.51, Max: 15.84, Min: 3.23 
        Episode: 75, Score: 9.25, Max: 19.36, Min: 3.18 
        Episode: 76, Score: 9.24, Max: 12.16, Min: 5.64 
        Episode: 77, Score: 12.41, Max: 30.06, Min: 6.31 
        Episode: 78, Score: 12.03, Max: 19.00, Min: 4.25 
        Episode: 79, Score: 11.79, Max: 16.88, Min: 7.00 
        Episode: 80, Score: 13.59, Max: 22.56, Min: 7.82 
        Average Score: 4.36, Time: 00:22:41
        ----------------------------
        Episode: 81, Score: 10.58, Max: 16.80, Min: 3.17 
        Episode: 82, Score: 13.42, Max: 30.82, Min: 4.87 
        Episode: 83, Score: 13.07, Max: 17.99, Min: 8.92 
        Episode: 84, Score: 12.48, Max: 17.30, Min: 5.49 
        Episode: 85, Score: 11.83, Max: 18.89, Min: 5.16 
        Episode: 86, Score: 14.22, Max: 20.19, Min: 8.52 
        Episode: 87, Score: 12.14, Max: 18.28, Min: 2.46 
        Episode: 88, Score: 14.79, Max: 21.32, Min: 9.35 
        Episode: 89, Score: 12.86, Max: 17.61, Min: 6.12 
        Episode: 90, Score: 15.27, Max: 23.78, Min: 7.09 
        Average Score: 5.33, Time: 00:26:08
        ----------------------------
        Episode: 91, Score: 13.45, Max: 17.91, Min: 3.52 
        Episode: 92, Score: 16.17, Max: 31.16, Min: 8.55 
        Episode: 93, Score: 13.70, Max: 18.53, Min: 3.68 
        Episode: 94, Score: 14.92, Max: 21.80, Min: 5.82 
        Episode: 95, Score: 14.34, Max: 18.32, Min: 9.54 
        Episode: 96, Score: 15.27, Max: 25.33, Min: 6.36 
        Episode: 97, Score: 15.30, Max: 21.03, Min: 6.29 
        Episode: 98, Score: 16.08, Max: 22.37, Min: 8.26 
        Episode: 99, Score: 17.77, Max: 39.47, Min: 10.96 
        Episode: 100, Score: 14.18, Max: 21.90, Min: 1.59 
        Average Score: 6.31, Time: 00:29:36
        ----------------------------
        Episode: 101, Score: 15.34, Max: 20.73, Min: 9.14 
        Episode: 102, Score: 17.77, Max: 22.09, Min: 7.72 
        Episode: 103, Score: 16.44, Max: 25.77, Min: 3.52 
        Episode: 104, Score: 16.30, Max: 23.22, Min: 0.10 
        Episode: 105, Score: 17.64, Max: 21.84, Min: 7.86 
        Episode: 106, Score: 17.21, Max: 25.10, Min: 7.77 
        Episode: 107, Score: 18.85, Max: 35.11, Min: 6.72 
        Episode: 108, Score: 19.84, Max: 32.51, Min: 10.62 
        Episode: 109, Score: 17.65, Max: 29.23, Min: 6.38 
        Episode: 110, Score: 17.93, Max: 27.31, Min: 10.75 
        Average Score: 8.00, Time: 00:33:03
        ----------------------------
        Episode: 111, Score: 19.07, Max: 25.10, Min: 10.77 
        Episode: 112, Score: 21.42, Max: 26.33, Min: 9.45 
        Episode: 113, Score: 16.14, Max: 25.71, Min: 6.71 
        Episode: 114, Score: 18.61, Max: 24.55, Min: 9.74 
        Episode: 115, Score: 18.52, Max: 26.03, Min: 8.68 
        Episode: 116, Score: 20.62, Max: 27.36, Min: 9.76 
        Episode: 117, Score: 21.92, Max: 36.63, Min: 16.37 
        Episode: 118, Score: 23.66, Max: 39.44, Min: 19.50 
        Episode: 119, Score: 24.66, Max: 34.72, Min: 15.23 
        Episode: 120, Score: 23.16, Max: 29.91, Min: 9.75 
        Average Score: 9.97, Time: 00:36:31
        ----------------------------
        Episode: 121, Score: 22.25, Max: 34.46, Min: 3.16 
        Episode: 122, Score: 24.21, Max: 36.02, Min: 12.87 
        Episode: 123, Score: 22.65, Max: 32.32, Min: 15.47 
        Episode: 124, Score: 20.81, Max: 32.60, Min: 12.39 
        Episode: 125, Score: 23.90, Max: 32.22, Min: 13.00 
        Episode: 126, Score: 26.57, Max: 37.82, Min: 22.32 
        Episode: 127, Score: 24.40, Max: 29.42, Min: 11.83 
        Episode: 128, Score: 22.11, Max: 29.30, Min: 15.14 
        Episode: 129, Score: 24.99, Max: 39.07, Min: 15.69 
        Episode: 130, Score: 26.52, Max: 39.33, Min: 17.51 
        Average Score: 12.18, Time: 00:39:59
        ----------------------------
        Episode: 131, Score: 25.76, Max: 32.82, Min: 19.93 
        Episode: 132, Score: 25.39, Max: 34.31, Min: 13.73 
        Episode: 133, Score: 26.36, Max: 33.80, Min: 16.51 
        Episode: 134, Score: 27.10, Max: 37.00, Min: 16.08 
        Episode: 135, Score: 28.35, Max: 37.23, Min: 14.98 
        Episode: 136, Score: 29.23, Max: 36.15, Min: 15.05 
        Episode: 137, Score: 27.38, Max: 35.27, Min: 14.40 
        Episode: 138, Score: 27.44, Max: 38.00, Min: 15.86 
        Episode: 139, Score: 28.36, Max: 35.56, Min: 14.62 
        Episode: 140, Score: 30.91, Max: 35.01, Min: 25.33 
        Average Score: 14.67, Time: 00:43:29
        ----------------------------
        Episode: 141, Score: 31.09, Max: 37.86, Min: 20.11 
        Episode: 142, Score: 32.11, Max: 39.15, Min: 21.97 
        Episode: 143, Score: 29.93, Max: 39.35, Min: 11.90 
        Episode: 144, Score: 31.60, Max: 36.15, Min: 23.17 
        Episode: 145, Score: 32.25, Max: 36.41, Min: 25.36 
        Episode: 146, Score: 34.29, Max: 37.25, Min: 30.02 
        Episode: 147, Score: 32.80, Max: 37.67, Min: 23.58 
        Episode: 148, Score: 31.41, Max: 37.80, Min: 17.91 
        Episode: 149, Score: 32.34, Max: 36.61, Min: 19.51 
        Episode: 150, Score: 33.77, Max: 38.09, Min: 22.63 
        Average Score: 17.42, Time: 00:46:58
        ----------------------------
        Episode: 151, Score: 34.62, Max: 38.90, Min: 25.92 
        Episode: 152, Score: 33.88, Max: 39.17, Min: 23.90 
        Episode: 153, Score: 35.21, Max: 39.16, Min: 26.61 
        Episode: 154, Score: 35.27, Max: 39.52, Min: 31.15 
        Episode: 155, Score: 33.79, Max: 39.50, Min: 20.23 
        Episode: 156, Score: 34.26, Max: 39.45, Min: 26.17 
        Episode: 157, Score: 34.55, Max: 38.49, Min: 24.37 
        Episode: 158, Score: 36.35, Max: 39.19, Min: 30.45 
        Episode: 159, Score: 35.22, Max: 38.54, Min: 28.45 
        Episode: 160, Score: 33.22, Max: 38.38, Min: 27.30 
        Average Score: 20.37, Time: 00:50:27
        ----------------------------
        Episode: 161, Score: 35.21, Max: 38.26, Min: 31.03 
        Episode: 162, Score: 34.42, Max: 38.44, Min: 26.88 
        Episode: 163, Score: 35.72, Max: 39.30, Min: 30.08 
        Episode: 164, Score: 35.11, Max: 38.84, Min: 26.74 
        Episode: 165, Score: 35.95, Max: 38.97, Min: 25.23 
        Episode: 166, Score: 35.81, Max: 39.49, Min: 30.40 
        Episode: 167, Score: 33.96, Max: 38.78, Min: 20.60 
        Episode: 168, Score: 36.15, Max: 39.46, Min: 32.27 
        Episode: 169, Score: 35.30, Max: 39.47, Min: 30.27 
        Episode: 170, Score: 37.29, Max: 39.47, Min: 32.17 
        Average Score: 23.09, Time: 00:53:55
        ----------------------------
        Episode: 171, Score: 35.42, Max: 39.03, Min: 26.74 
        Episode: 172, Score: 36.07, Max: 39.24, Min: 31.35 
        Episode: 173, Score: 36.95, Max: 39.48, Min: 29.55 
        Episode: 174, Score: 35.83, Max: 39.50, Min: 24.39 
        Episode: 175, Score: 36.70, Max: 39.10, Min: 30.66 
        Episode: 176, Score: 36.73, Max: 39.40, Min: 28.42 
        Episode: 177, Score: 38.18, Max: 39.58, Min: 34.24 
        Episode: 178, Score: 35.96, Max: 39.13, Min: 30.84 
        Episode: 179, Score: 36.85, Max: 39.14, Min: 28.81 
        Episode: 180, Score: 36.28, Max: 39.46, Min: 30.43 
        Average Score: 25.67, Time: 00:57:24
        ----------------------------
        Episode: 181, Score: 36.87, Max: 39.30, Min: 30.64 
        Episode: 182, Score: 36.78, Max: 39.03, Min: 29.87 
        Episode: 183, Score: 35.78, Max: 38.46, Min: 28.22 
        Episode: 184, Score: 37.48, Max: 39.29, Min: 35.77 
        Episode: 185, Score: 36.50, Max: 39.39, Min: 32.53 
        Episode: 186, Score: 36.71, Max: 39.14, Min: 31.65 
        Episode: 187, Score: 36.81, Max: 38.97, Min: 33.91 
        Episode: 188, Score: 36.09, Max: 39.38, Min: 29.28 
        Episode: 189, Score: 36.45, Max: 39.17, Min: 30.47 
        Episode: 190, Score: 36.35, Max: 38.42, Min: 31.58 
        Average Score: 28.02, Time: 01:00:52
        ----------------------------
        Episode: 191, Score: 37.05, Max: 38.82, Min: 35.09 
        Episode: 192, Score: 35.66, Max: 39.49, Min: 27.43 
        Episode: 193, Score: 36.57, Max: 39.56, Min: 33.43 
        Episode: 194, Score: 36.77, Max: 38.62, Min: 31.92 
        Episode: 195, Score: 37.84, Max: 39.08, Min: 35.12 
        Episode: 196, Score: 37.08, Max: 39.45, Min: 33.46 
        Episode: 197, Score: 37.84, Max: 39.22, Min: 36.32 
        Episode: 198, Score: 37.65, Max: 39.57, Min: 33.53 
        Episode: 199, Score: 37.41, Max: 39.21, Min: 29.68 
        Episode: 200, Score: 36.26, Max: 39.28, Min: 33.08 
        Average Score: 30.21, Time: 01:04:20
        ----------------------------

        Environment solved in 100 episodes!	Average Score: 30.21

![image](https://user-images.githubusercontent.com/89017449/134422660-c7d609ef-9bed-451c-9d08-dc482612e792.png)



### Ideas for improvement:

- Using [Prioritized Replay](https://arxiv.org/abs/1511.05952) has generally shown to have been quite useful. It is expected that it'll lead to an improved performance here too.

- Other algorithms like TRPO, [PPO](https://openai.com/blog/openai-baselines-ppo/), [A3C](https://openai.com/blog/openai-baselines-ppo/), A2C that have been discussed in the course could potentially lead to better results as well.

- The Q-prop algorithm, which combines both off-policy and on-policy learning, could be good one to try.

- General optimization techniques like cyclical learning rates and warm restarts could be useful as well.
