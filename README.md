# Banana Navigation

Banana Navigation is a Python implementation of Deep Reinforcement Learning methods to solve an environment where an agent has to collect as many yellow bananas as possible while avoiding the blue bananas in a large boxed squared environment.

## The Environnement

The **state** space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction.<br>
Given this information, the agent has to learn how to best select **actions**. Four discrete actions are available, corresponding to : <br> 
{0 - *move forward*, 1 - *move backward*, 2 - *turn left*, 3 - *turn right*} <br>
A **reward** of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.<br>
The **task** is episodic, and in order to **solve the environment**, your agent must get an average score of +13 over 100 consecutive episodes.

## Getting Started

You will need to install PyTorch, the ML-Agents toolkit, and a few more Python packages.

1. Install the dependencies using the requirements file.
    - cd to the directory where requirements.txt is located.
    - activate your virtualenv.
    - run: `pip install -r requirements.txt` in your shell.


2. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

3. Place the file in your working repository and unzip (or decompress) the file.


## Instructions

Follow the instructions in `Navigation.ipynb` to train an agent an watch it interact with the environment.  
You will need to intanciate the `agent` and make it interact with the environment throught the use of the methods `act` and `step` as in the following example.


```python
from ressource.Agent_PER import Agent

agent = Agent(state_size, action_size)

env_info = env.reset(train_mode=True)[brain_name] # reset the environment
state = env_info.vector_observations[0]           # get the current state
score = 0                                         # initialize the score

while True:
    
    action = agent.act(state)                     # select an action
    
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    
    agent.step(state, action, reward, next_state, done) 
    
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    
    if done:                                       # exit loop if episode finished
        break
```

In order to load a pre-trained agent and to watch it interact with the environnment, use the following code:

```python
from ressource.Agent_PER import Agent

def load_agent(modelname, path_model, use_dueling_net=True, use_DDQN=True):

    checkpoint = torch.load(os.path.join(path_model, fr'{modelname}.pth'))
    
    agent = Agent_PER(state_size, action_size, use_dueling_net, use_DDQN)
    agent.QNet_local.load_state_dict(checkpoint)
    agent.QNet_target.load_state_dict(checkpoint)
    
    return agent

agent = load_agent(modelname, path_model)

env_info = env.reset(train_mode=False)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
while True:
    action = agent.act(state)                      # select an action
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    if done:                                       # exit loop if episode finished
        break
    
print("Score: {}".format(score))

env.close()
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)