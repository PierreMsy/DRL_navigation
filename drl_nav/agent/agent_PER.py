import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

from drl_nav.network import VanillaQNet, Dueling_QNet
from drl_nav.component import PrioritizedReplayBuffer


EPS_INIT, EPS_MIN, EPS_DECAY = 3e-1, 1e-2, .995 # espilon greedy policy
GAMMA = .99                                     # discount rate
LR = 1e-5                                       # learning rate
TAU = 1.5e-2                                    # soft target network updates
BETA_INIT, BETA_TO_1 = .4, 800                  # PER focus on prioritized experiences
BUFFER_SIZE = 10_000                            # Replay buffer size
BATCH_SIZE = 64                                 # Learning batch size
UPDATE_EVERY = 10                               # number of actions between two learning

DEVICE = "cpu" # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent_PER:
    '''
    Agent that will interact with the environnment to maximize the expected reward
    The agent will use a buffer
    '''

    def __init__(self, state_size, action_size, use_dueling_net=True, use_DDQN=True, seed=42):
        '''
        Create the replay buffer, the local and target network and initialize learning parameters.
        Choose if a classical Q-network of a Dueling Q-network will be used and whether or not
        Double Q-Learning will be used.
        '''
        self.use_DDQN = use_DDQN
        self.action_size = action_size
        self.epsilon = EPS_INIT
        self.beta = BETA_INIT
        self.t_step = 0
        
        if use_dueling_net:
            self.QNet_local = Dueling_QNet(state_size, action_size, seed)
            self.QNet_target = Dueling_QNet(state_size, action_size, seed)
        else:
            self.QNet_local = VanillaQNet(state_size, action_size, seed)
            self.QNet_target = VanillaQNet(state_size, action_size, seed)
        
        self.replay_buffer = PrioritizedReplayBuffer(BUFFER_SIZE)
        self.optimizer = optim.Adam(self.QNet_local.parameters(), lr=LR)  

    def act(self, state):
        '''
        From a state, select an action following a epsilon greedy policy.
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)

        self.QNet_local.eval()
        with torch.no_grad():
            action_values = self.QNet_local.forward(state)
        self.QNet_local.train()

        # epsilon greedy policy
        if np.random.rand() > self.epsilon:
            action = np.argmax(action_values).item()
        else:
            action = random.choice(np.arange(self.action_size))
        self._updates_learning_parameters()

        return action
    
    def step(self, state, action, reward, next_state, done):
        '''
        save the experience and decide to learn or to continue interacting.
        '''
        td_error = self._compute_td_error(state, action, reward, next_state, done)
        self.replay_buffer.add(state, action, reward, next_state, done, td_error)

        self.t_step += 1
        if self.t_step % UPDATE_EVERY == 0:
            if len(self.replay_buffer) >= BATCH_SIZE:
                experiences = self.replay_buffer.sample(BATCH_SIZE)
                if self.use_DDQN:
                    self.learn_DDQN(experiences)
                else:
                    self.learn(experiences)
        
    def learn(self, experiences):
        '''
        Gradient descent update
            TD target = r + gamma * max_a(q(S',a,w'))
        '''
        states, actions, rewards, next_states, dones, priorities = experiences

        actual_action_values = (self.QNet_local.forward(states).gather(1, actions.unsqueeze(1)).squeeze(1)).to(DEVICE)

        next_action_values = self.QNet_target.forward(next_states)
        max_next_action_values = torch.max(next_action_values, axis=1).values
        
        td_target = (rewards + GAMMA *  max_next_action_values * (1 - dones)).to(DEVICE)

        self.optimizer.zero_grad()
        loss = ((1/len(self.replay_buffer) * 1/priorities).pow(self.beta) 
                    * (td_target.data - actual_action_values).pow(2)).mean()
        loss.backward()
        self.optimizer.step()

        # Soft updates the target network
        self.soft_update(self.QNet_local, self.QNet_target, TAU)
        
    def learn_DDQN(self, experiences):
        '''
        Gradient descent update
        double DQN :
            TD target = r + gamma * q(S',argmax_a(q(S',a,w)),w')
        '''
        states, actions, rewards, next_states, dones, priorities = experiences

        actual_action_values = (self.QNet_local.forward(states).gather(1, actions.unsqueeze(1)).squeeze(1)).to(DEVICE)

        next_Qvalues = (self.QNet_local.forward(next_states)).to(DEVICE)
        target_next_Qvalues = (self.QNet_target.forward(next_states)).to(DEVICE)
        max_next_action_values = target_next_Qvalues.gather(1, torch.argmax(next_Qvalues, axis=1).unsqueeze(1)).squeeze(1) 
        
        td_target = rewards + GAMMA *  max_next_action_values * (1 - dones)

        self.optimizer.zero_grad()
        loss = ((1/len(self.replay_buffer) * 1/priorities).pow(self.beta)
                    * (td_target.data - actual_action_values).pow(2)).mean()
        loss.backward()
        self.optimizer.step()

        # Soft updates the target network
        self.soft_update(self.QNet_local, self.QNet_target, TAU)  

    def soft_update(self, net_local, net_target, tau):
        """
        Soft update model parameters.
            θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for local_param, target_param in zip(net_local.parameters(), net_target.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def _updates_learning_parameters(self):
        '''
        Updates :
            - Decay epsilon to emphasize exploitation.
            - Increase beta to emphasize prioritized experiences.
        '''
        self.epsilon = max(EPS_MIN, self.epsilon * EPS_DECAY)
        self.beta = min(1, self.beta + (1 - BETA_INIT) / BETA_TO_1) 

    def _compute_td_error(self, state, action, reward, next_state, done):
        '''
        Compute the error for a specific experience in order to evaluate its
        priority for the future learning phases
        '''
        state  = (torch.FloatTensor(np.float32(state))).to(DEVICE)
        next_state = (torch.FloatTensor(np.float32(next_state))).to(DEVICE)

        self.QNet_local.eval()
        with torch.no_grad():
            q_value = self.QNet_local.forward(state)[action]
            if self.use_DDQN:
                max_next_action_values = self.QNet_target.forward(next_state)[torch.argmax(self.QNet_local.forward(next_state))]
                q_target = reward + GAMMA *  max_next_action_values * (1 - done)
            else:
                q_target = reward +  GAMMA * self.QNet_target.forward(next_state).max() * (1 - done)
        td_error = q_target - q_value
        self.QNet_local.train()
            
        return td_error.item()