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


### Future ideas

Check different values for hyperparameters such as LEARNING_PERIOD, and neural network parameters fc1_units, fc2_units, etc.
How does the addition of new nonlinear layers in the used neural networks affect the robustness of the algorithm.
It would be interesting to train agents using MAPPO and compare them with MADDPG.
