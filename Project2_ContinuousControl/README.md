## The Environment

For this project, we work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#reacher) environment.

![reacher](https://user-images.githubusercontent.com/89017449/134245801-9049626a-1597-47e3-9ec9-07784852dac4.gif)


In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## Distributed Training

For this project, we will provide you with two separate versions of the Unity environment:

- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.
The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.

## Solving the Environment

To solve the environment our agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
- This yields an average score for each episode (where the average is over all 20 agents).
As an example, consider the plot below, where we have plotted the average score (over all 20 agents) obtained with each episode.

As an example, consider the plot below, where the average score (over all 20 agents) is obtained with each episode.

![image](https://user-images.githubusercontent.com/89017449/134247347-9bdc0b80-ff1e-4837-8313-b3cdda84f5ad.png)

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. In the case of the plot above, the environment was solved at episode 63, since the average of the average scores from episodes 64 to 163 (inclusive) was greater than +30.

## How to explore the environment

### Step 1: Activate the Environment

If you haven't already, please follow the [instructions in the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment. These instructions can be found in README.md at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

(For Windows users) The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

### Step 2: Download the Unity Environment

For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:


#### Twenty (20) Agents
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)


Then, place the file in the p2_continuous-control/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.

(For Windows users) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use this [link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) to obtain the "headless" version of the environment. You will not be able to watch the agent without [enabling a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), but you will be able to train the agent. (To watch the agent, you should follow the instructions to enable a virtual screen, and then download the environment for the Linux operating system above.)

### Explore the environment

After you have followed the instructions above, open Continuous_Control.ipynb (located in the p2_continuous-control/ folder in the DRLND GitHub repository) and follow the instructions to learn how to use the Python API to control the agent.

Watch the (silent) video below to see what kind of output to expect from the notebook (for version 2 of the environment), if everything is working properly! Version 1 will look very similar (where you'll see a single agent, instead of 20!).

In the last code cell of the notebook, you'll learn how to design and observe an agent that always selects random actions at each timestep. Your goal in this project is to create an agent that performs much better!

## Build your Own Environment

For this project, we have built the Unity environment for you, and you must use the environment files that we have provided.

If you are interested in learning to build your own Unity environments after completing the project, you are encouraged to follow the instructions [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Getting-Started-with-Balance-Ball.md), which walk you through all of the details of building an environment from a Unity scene.

## Environment

The environment is simulated by Unity application _Reacher_ lying in the subdirectory _Reacher_Windows_x86_64_.
We start the environment as follows:

      env = UnityEnvironment(file_name='Reacher_Windows_x86_64/Reacher.exe')

We are considering the version of the environment with 20 agents. After each episode, we add up the rewards received    
by each agent, to get a score for each agent. This yields 20 (potentially different) scores. We then take the     
**average score**  over all 20 agents. The environment is considered solved, when the average (over 100 episodes)      
of those average scores  is at least +30.    

### Prepare environment on the local machine

You need at least the following three packages:

1. **deep-reinforcement-learning  (DRLND)**        
   The instructions to set up the DRLND repository can be found [here](https://github.com/udacity/deep-reinforcement-learning#dependencies). This repository contains material related to Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.

2. **ml-agents  (ML-Agents Toolkit)**
   To configure the ML-Agents Toolkit for Windows you need to complete the following steps:
    
    2.1  Creating a new Conda environment:
    
       conda create -n ml-agents python=3.6
       
    2.2 Activating ml-agents by the following command:
    
       activate ml-agents
       
    2.3 Latest versions of TensorFlow won't work, so you will need to make sure that you install version 1.7.1:
    
       pip install tensorflow==1.7.1
       
    For details on installing the ML-Agents Toolkit, see the instructions [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation-Windows.md).     
    
3. **Unity environment _Reacher_**

    For this project, we not need to install Unity because the environment already built. For 20 agents, the environment     
    can be downloaded as follows:

   Windows (64-bit), [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)    
   Windows (32-bit), [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)     

   Download this environment zip into  **p2_continuous-control/** folder, and unzip the file.

### Train the Agent

   Run the notebook _Continuous_Control_ddpg.ipynb_
   
   [1] import UnityEnvironment    
   [2] env = UnityEnvironment(file_name='/data/Reacher_Linux_NoVis/Reacher.x86_64')   # create environment      
   [3] Environments contain _brains_ which are responsible for deciding the actions of their associated agents. 
       We check for the first brain available.      
   [4] Examine the State and Action Spaces. We get the information frame as follows:   
       
      Number of agents: 20
      Size of each action: 4
      There are 20 agents. Each observes a state with length: 33
      The state for the first agent looks like: [  0.00000000e+00  -4.00000000e+00   0.00000000e+00   1.00000000e+00
        -0.00000000e+00  -0.00000000e+00  -4.37113883e-08   0.00000000e+00
         0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
         0.00000000e+00   0.00000000e+00  -1.00000000e+01   0.00000000e+00
         1.00000000e+00  -0.00000000e+00  -0.00000000e+00  -4.37113883e-08
         0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
         0.00000000e+00   0.00000000e+00   5.75471878e+00  -1.00000000e+00
         5.55726624e+00   0.00000000e+00   1.00000000e+00   0.00000000e+00
        -1.68164849e-01] 
   
   [5]  Create _env_info_ and _agent_:

     env_info = env.reset(train_mode=True)[brain_name]      
     env_info = env.reset(train_mode=True)[brain_name]
     agent = Agent(state_size=state_size, action_size=action_size, random_seed=8)    

   [6]  Define and run the main function _ddpg_ :
   
      def ddpg(n_episodes=2000, max_t = 2000, window_size=100, print_interval=10, score_threshold=30.0):
    
    scores_deque = deque(maxlen=window_size)
    scores_global = []
        
    time_start = time.time()
    print("Training started!")
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]      # reset the environment    
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        agent.reset()
        score_average = 0

        for timestep in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            agent.step(states, actions, rewards, next_states, dones, timestep)
            states = next_states                               # roll over states to next time step
            scores += rewards                                  # update the score (for each agent)    
            if np.any(dones):                                  # exit loop if episode finished
                break
        
        score = np.mean(scores)
        scores_deque.append(score)
        score_average = np.mean(scores_deque)
        scores_global.append(score)
        
        print('Episode: {}, Score: {:.2f}, Max: {:.2f}, Min: {:.2f} '\
              .format(i_episode, score, np.max(scores), np.min(scores)))
        
        if i_episode % print_interval == 0 or (len(scores_deque) == 100 and np.mean(scores_deque) >= 30) :
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            s = (int)(time.time() - time_start) 
            print('Average Score: {:.2f}, Time: {:02}:{:02}:{:02}'\
                  .format(score_average, s//3600, s%3600//60, s%60))
            print('----------------------------')
            
        if len(scores_deque) == window_size and np.mean(scores_deque) >= score_threshold:  
            print('\nEnvironment solved in {} episodes!\tAverage Score: {:.2f}'.format(i_episode-window_size, score_average))
            break
            
    return scores_global

      
   [7]  Print graph of scores over all episodes. 
        
      fig = plt.figure()
      ax = fig.add_subplot(111)
      plt.plot(np.arange(1, len(scores)+1), scores)
      plt.ylabel('Score')
      plt.xlabel('Episode #')
      plt.show()
        
### Weights of the Trained Agent
  
  The **weights** of the trained agent are saved into the files _checkpoint_actor.pth_  and  _checkpoint_critic.pth_.



Source: Udacity DRL Course

