### The Environment

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![image](https://user-images.githubusercontent.com/89017449/134576502-e88b76e4-b715-43e9-943b-2a7e1bc31c49.png)

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Explore the environment on your own machine

Follow the instructions below to explore the environment on your own machine! You will also learn how to use the Python API to control your agent.

### Step 1: Activate the Environment
If you haven't already, please follow the instructions in [the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment. These instructions can be found in README.md at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

(For Windows users) The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

SPECIAL NOTE TO BETA TESTERS - please also download the p3_collab-compet folder from here and place it in the DRLND GitHub repository.

### Step 2: Download the Unity Environment
For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)


Then, place the file in the p3_collab-compet/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.

(For Windows users) Check out this link if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use this link to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to enable a virtual screen, and then download the environment for the Linux operating system above.)

### Step 3: Explore the Environment
After you have followed the instructions above, open Tennis.ipynb (located in the p3_collab-compet/ folder in the DRLND GitHub repository) and follow the instructions to learn how to use the Python API to control the agent.


## Instructions

The solution contanins the following files:

- `Tennis_maddpg.ipynb`: notebook containing the solution
- `main.py`: 
- `model.py`: code containing the Q-Network used as the function approximator by the agent
- `maddpg_agent.py`: saved model weights for the 
- `checkpoint_actor_0.pth`: saved model weights for the 
- `checkpoint_actor_1.pth`: saved model weights for the 
- `checkpoint_critic_0.pth`: saved model weights for the 
- `checkpoint_critic_1.pth`: saved model weights for the 

Follow the instructions in `Tennis_maddpg.ipynb` to get started with training your own agent!

## How to start the environment

      ## Python install 
      !pip -q install ./python
      
      ## import the needed libraries
      from maddpg_agent import Agent
      from collections import deque
      import matplotlib.pyplot as plt
      import numpy as np
      import random
      import time
      import torch
      from unityagents import UnityEnvironment
      %matplotlib inline
      
      ##start the environment
      env = UnityEnvironment(file_name="/data/Tennis_Linux_NoVis/Tennis")
      
      # get the default brain
      brain_name = env.brain_names[0]
      brain = env.brains[brain_name]
      
#### Examine the State and Action Spaces

         # reset the environment
         env_info = env.reset(train_mode=True)[brain_name]

         # number of agents
         num_agents = len(env_info.agents)
         print('Number of agents:', num_agents)

         # size of each action
         action_size = brain.vector_action_space_size
         print('Size of each action:', action_size)

         # examine the state space 
         states = env_info.vector_observations
         state_size = states.shape[1]
         print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
         print('The state for the first agent looks like:', states[0])

#### Take Random Actions in the Environment

         for i in range(1, 6):                                      # play game for 5 episodes
             env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
             states = env_info.vector_observations                  # get the current state (for each agent)
             scores = np.zeros(num_agents)                          # initialize the score (for each agent)
             while True:
                 actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
                 actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
                 env_info = env.step(actions)[brain_name]           # send all actions to tne environment
                 next_states = env_info.vector_observations         # get next state (for each agent)
                 rewards = env_info.rewards                         # get reward (for each agent)
                 dones = env_info.local_done                        # see if episode finished
                 scores += env_info.rewards                         # update the score (for each agent)
                 states = next_states                               # roll over states to next time step
                 if np.any(dones):                                  # exit loop if episode finished
                     break
             print('Score (max over agents) from episode {}: {}'.format(i, np.max(scores)))


### How to import and run the model
   
           
 #### Defining the MADDPG function

      SOLVED_SCORE = 0.5
      CONSEC_EPISODES = 100
      PRINT_EVERY = 10
      ADD_NOISE = True

      # MADDPG function
      def maddpg(n_episodes=2000, max_t=1000, train_mode=True):
          """Multi-Agent Deep Deterministic Policy Gradient (MADDPG)

       Params
       ======
           n_episodes (int)      : maximum number of training episodes
           max_t (int)           : maximum number of timesteps per episode
           train_mode (bool)     : if 'True' set environment to training mode

       """
       scores_window = deque(maxlen=CONSEC_EPISODES)
       scores_all = []
       moving_average = []
       best_score = -np.inf
       best_episode = 0
       already_solved = False


       for i_episode in range(1, n_episodes+1):
           env_info = env.reset(train_mode=train_mode)[brain_name]         # reset the environment
           states = np.reshape(env_info.vector_observations, (1,48)) # get states and combine them
           agent_0.reset()
           agent_1.reset()
           scores = np.zeros(num_agents)
           while True:
               actions = get_actions(states, ADD_NOISE)           # choose agent actions and combine them
               env_info = env.step(actions)[brain_name]           # send both agents' actions together to the environment
               next_states = np.reshape(env_info.vector_observations, (1, 48)) # combine the agent next states
               rewards = env_info.rewards                         # get reward
               done = env_info.local_done                         # see if episode finished
               agent_0.step(states, actions, rewards[0], next_states, done, 0) # agent 1 learns
               agent_1.step(states, actions, rewards[1], next_states, done, 1) # agent 2 learns
               scores += np.max(rewards)                          # update the score for each agent
               states = next_states                               # roll over states to next time step
               if np.any(done):                                   # exit loop if episode finished
                   break

           ep_best_score = np.max(scores)
           scores_window.append(ep_best_score)
           scores_all.append(ep_best_score)
           moving_average.append(np.mean(scores_window))

           # save best score                        
           if ep_best_score > best_score:
               best_score = ep_best_score
               best_episode = i_episode

           # print results
           if i_episode % PRINT_EVERY == 0:
               print('Episodes {:0>4d}-{:0>4d}\tMax Score: {:.3f}\tAverage Score: {:.3f}'.format(
                   i_episode-PRINT_EVERY, i_episode, np.max(scores_all[-PRINT_EVERY:]), moving_average[-1]))

           # determine if environment is solved
           if moving_average[-1] >= SOLVED_SCORE:
               print(' ------Environment solved in {:d} episodes!------ \
               \n ------Average Score: {:.3f} over past {:d} episodes------ '.format(
                   i_episode-CONSEC_EPISODES, moving_average[-1], CONSEC_EPISODES))
               already_solved = True
               # save weights
               torch.save(agent_0.actor_local.state_dict(), 'checkpoint_actor_0.pth')
               torch.save(agent_0.critic_local.state_dict(), 'checkpoint_critic_0.pth')
               torch.save(agent_1.actor_local.state_dict(), 'checkpoint_actor_1.pth')
               torch.save(agent_1.critic_local.state_dict(), 'checkpoint_critic_1.pth')
               break
           else:
               continue

       return scores_all, moving_average

#### gets actions for each agent, combining them into one array and initializing agents

          def get_actions(states, add_noise):
          '''gets actions for each agent and then combines them into one array'''
          action_0 = agent_0.act(states, add_noise)    # agent 0 chooses an action
          action_1 = agent_1.act(states, add_noise)    # agent 1 chooses an action
          return np.concatenate((action_0, action_1), axis=0).flatten()
       
 
         agent_0 = Agent(state_size, action_size, num_agents=1, random_seed=0)
         agent_1 = Agent(state_size, action_size, num_agents=1, random_seed=0)
         
         
#### Run the training loop
         scores, avgs = maddpg()




### How to plot the scores

         # plot the scores
         fig = plt.figure()
         ax = fig.add_subplot(111)
         plt.plot(np.arange(len(scores)), scores, label='MADDPG')
         plt.plot(np.arange(len(scores)), avgs, c='r', label='moving avg')
         plt.ylabel('Score')
         plt.xlabel('Episode #')
         plt.legend(loc='upper left');
         plt.show()
        
      
        
### Weights of the Trained Agent
  
  The **weights** of the trained agent are saved into the files _checkpoint_actor.pth_  and  _checkpoint_critic.pth_.



