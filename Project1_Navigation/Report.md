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


## Instructions

Follow the instructions in Navigation_DQN.ipynb to get started with training your own agent! To watch a trained smart agent, follow the instructions below:

DQN: If you want to run the original DQN algorithm, use the checkpoint dqn.pth for loading the trained model. Also, choose the parameter qnetwork as QNetwork while defining the agent and the parameter update_type as dqn.
