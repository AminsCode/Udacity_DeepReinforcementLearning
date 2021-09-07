## Agent Implementation

**Deep Q-Networks**

This project implements a Value Based method called Deep Q-Networks.

Deep Q Learning combines 2 approaches:

- A Reinforcement Learning method called Q Learning (aka SARSA max)
- A Deep Neural Network to learn a Q-table approximation (action-values)

Especially, this implementation includes the 2 major training improvements by Deepmind and described in their Nature publication : ["Human-level control through deep reinforcement learning (2015)"](http://www.davidqiu.com:8888/research/nature14236.pdf)

- Experience Replay: The act of sampling a small batch of tuples from the replay buffer in order to learn is known as experience replay. In addition to breaking harmful correlations, experience replay allows us to learn more from individual tuples multiple times, recall rare occurrences, and in general make better use of our experience.

- Fixed Q Targets: 

  ![image](https://user-images.githubusercontent.com/89017449/132307357-6c09916f-6fa2-48b2-8adb-546e7dc7e6e0.png)

  Source: Deep Reinforcement Learning Nanodegree


## Code implementation

The code used here is derived from the Deep Reinforcement Learning Nanodegree, and has been slightly adjusted for being used with the banana environment.

The code consist of :

- model.py : In this python file, a PyTorch QNetwork class is implemented. This is a regular fully connected Deep Neural Network using the PyTorch Framework. This network will be trained to predict the action to perform depending on the environment observed states. This Neural Network is used by the DQN agent and is composed of :

  - the input layer which size depends of the state_size parameter passed in the constructor
  - 2 hidden fully connected layers of 1024 cells each
  - the output layer which size depends of the action_size parameter passed in the constructor

- dqn_agent.py : In this python file, a DQN agent and a Replay Buffer memory used by the DQN agent) are defined.

  The DQN agent class is implemented, as described in the Deep Q-Learning algorithm. It provides several methods :

  - constructor :
    Initialize the memory buffer (Replay Buffer)
    Initialize 2 instance of the Neural Network : the target network and the local network

  - step() :
    - Allows to store a step taken by the agent (state, action, reward, next_state, done) in the Replay Buffer/Memory
    - Every 4 steps (and if their are enough samples available in the Replay Buffer), update the target network weights with the current weight values from the local network (That's     part of the Fixed Q Targets technique)
   - act() which returns actions for the given state as per current policy (Note : The action selection use an Epsilon-greedy selection so that to balance between exploration and       exploitation for the Q Learning)
   - learn() which update the Neural Network value parameters using given batch of experiences from the Replay Buffer.
   - soft_update() is called by learn() to softly updates the value from the target Neural Network from the local network weights (That's part of the Fixed Q Targets technique)
   - The ReplayBuffer class implements a fixed-size buffer to store experience tuples (state, action, reward, next_state, done)
   - add() allows to add an experience step to the memory
   - sample() allows to randomly sample a batch of experience steps for the learning

- Navigation_DQN.ipynb : This Jupyter notebooks allows to train the agent. More in details it allows to :
Import the Necessary Packages
Examine the State and Action Spaces
Take Random Actions in the Environment (No display)
Train an agent using DQN
Plot the scores



## DQN parameters and results

The DQN agent uses the following parameters values (defined in dqn_agent.py)

```
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size 
GAMMA = 0.995           # discount factor 
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
```

The Neural Networks use the following architecture :

```
Input nodes (37) -> Fully Connected Layer (1024 nodes, Relu activation) -> Fully Connected Layer (1024 nodes, Relu activation) -> Ouput nodes (4)
```

The Neural Networks use the Adam optimizer with a learning rate LR=5e-4 and are trained using a BATCH_SIZE=64

Given the chosen architecture and parameters, the results are :

![Training_log](https://user-images.githubusercontent.com/89017449/132308655-84d82543-eede-4a85-9c4c-b2c16f2f8776.JPG)

![Score](https://user-images.githubusercontent.com/89017449/132308729-75a8f5ee-e0a7-45e3-94a2-434faa23a1c3.JPG)

These results meets the project's expectation as the agent is able to receive an average reward (over 100 episodes) of at least +13, and in 1023 episodes only (In comparison, according to Udacity's solution code for the project, their agent was benchmarked to be able to solve the project in fewer than 1800 episodes)

## Ideas for future work
As discussed in the Udacity Course, a further evolution to this project would be to train the agent directly from the environment's observed raw pixels instead of using the environment's internal states (37 dimensions)

To do so a Convolutional Neural Network would be added at the input of the network in order to process the raw pixels values (after some little preprocessing like rescaling the image size, converting RGB to gray scale, ...)

Other enhancements might also be implemented to increase the performance of the agent:

- Double DQN
- Dueling DQN
- Prioritized experience replay
