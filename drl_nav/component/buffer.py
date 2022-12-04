from collections import namedtuple, deque
import torch
import numpy as np
import random
from typing import Tuple


DEVICE = "cpu" # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Buffer:
    
    def __init__(self, buffer_size) -> None:
        self.buffer = deque(maxlen=buffer_size)
    
    def add():
        pass

    def sample():
        pass
    
    def __len__(self):
        return len(self.buffer)

class ReplayBuffer(Buffer):
    '''
    Buffer that hold experiences tuple for future learning.
    '''
    
    def __init__(self, buffer_size):
        super().__init__(buffer_size)
        self.Experience = namedtuple("Experience",
            ['state', 'action', 'reward', 'next_state', 'done']) 

    def add(self, state, action, reward, next_state, done):
        '''
        Create an experience tuple from one interaction and add it to the buffer
        '''
        experience = self.Experience(
            state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.buffer.append(experience)
    
    def sample(self, sample_size):
        '''
        Random sample as much experiences as requested by the sample_size 
        '''
        sampled_experiences = random.sample(self.buffer, sample_size)
        # sample = (torch.from_numpy(np.array(x)).float().to(device)
        #             for x in zip(*sampled_experiences))
        states_np, actions_np, rewards_np, next_states_np, dones_np = zip(*sampled_experiences)

        states = torch.from_numpy(np.array(states_np)).float().to(DEVICE)
        actions = torch.from_numpy(np.array(actions_np)).long().to(DEVICE)
        rewards = torch.from_numpy(np.array(rewards_np)).float().to(DEVICE)
        next_states = torch.from_numpy(np.array(next_states_np)).float().to(DEVICE)
        dones = torch.from_numpy(np.array(dones_np).astype(np.uint8)).float().to(DEVICE)

        return states, actions, rewards, next_states, dones
        

E = .1   # lower bound for the priority of an experience
A = .75  # Parametrize the trade of between uniform sampling (0) and sampling based only on the priority 
        
class PrioritizedReplayBuffer(Buffer):
    '''
    Buffer that hold experiences tuple for future learning.
    The PER sample the experiences with respect to their priorites in order to improve
    the learning by emphasizing on experiences that were poorly evaluated in the past.
    '''
    def __init__(self, buffer_size):
        super().__init__(buffer_size)
        self.buffer_priority = deque(maxlen=buffer_size)
        self.Experience = namedtuple("Experience",
            ['state', 'action', 'reward', 'next_state', 'done', 'priority']) 

    def add(self, state, action, reward, next_state, done, td_error):
        '''
        Compute the priority of an experience and create an experience tuple
        from one interaction and add it to the buffer.
        '''
        priority = (abs(td_error) + E) ** A
        experience = self.Experience(
            state=state, action=action, reward=reward, next_state=next_state, done=done, priority=priority)
        
        self.buffer.append(experience)
        self.buffer_priority.append(priority)
    
    def sample(self, sample_size):
        '''
        Sample as much experiences as requested by the sample_size with respect to the
        priorities of all the experiences stored
        '''
        sampled_experiences = random.choices(self.buffer,
                                             weights=self.buffer_priority, cum_weights=None,
                                             k=sample_size)
        
        states_np, actions_np, rewards_np, next_states_np, dones_np, priority_np = zip(*sampled_experiences)

        states = torch.from_numpy(np.array(states_np)).float().to(DEVICE)
        actions = torch.from_numpy(np.array(actions_np)).long().to(DEVICE)
        rewards = torch.from_numpy(np.array(rewards_np)).float().to(DEVICE)
        next_states = torch.from_numpy(np.array(next_states_np)).float().to(DEVICE)
        dones = torch.from_numpy(np.array(dones_np).astype(np.uint8)).float().to(DEVICE)
        priorities = torch.from_numpy(np.array(priority_np)).float().to(DEVICE)

        return states, actions, rewards, next_states, dones, priorities


class ImageBuffer(Buffer):
    
    def __init__(self, buffer_size) -> None:
        super().__init__(buffer_size)
        self.labeledImg = namedtuple("labeledImg", ['state', 'labels_banana']) 

    def add(self, state, labels_banana) -> None:
        """ Add a banana labelized state to the buffer. """
        labeledImg = self.labeledImg(state=state, labels_banana=labels_banana)
        self.buffer.append(labeledImg)

    def sample(self, sample_size) -> Tuple[list, list]:
        ''' ''' 
        sampled_labeledImgs = random.sample(self.buffer, sample_size)
        
        states, labels_banana = zip(*sampled_labeledImgs)
        #torch.from_numpy(np.array(states)).float().to(DEVICE)
        states = torch.from_numpy(np.moveaxis(states, 3, 1)).float().to(DEVICE)
        labels_banana = torch.from_numpy(states).float().to(DEVICE)

        return states, labels_banana
