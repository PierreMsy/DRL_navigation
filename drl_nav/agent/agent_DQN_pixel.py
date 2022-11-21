import numpy as np
import torch

from drl_nav.utils.schedule import ExponentialSchedule
from drl_nav.network import QNet, AuxNet, LabelizerNet, ConvBody
from drl_nav.component import ReplayBuffer, ImageBuffer


class Agent_DQN_pixel:
    '''
    '''

    def __init__(self, context, config):
        '''
        '''
        self.context = context
        self.epsilon = ExponentialSchedule()

        conv_body = ConvBody()
        self.q_network_local = QNet(conv_body, context.action_size)
        self.q_network_target = QNet(ConvBody(), context.action_size)
        self.q_network_target.load_state_dict(self.q_network_local.state_dict())
        self.auxiliary_network = AuxNet(conv_body)
        self.labelizer = LabelizerNet()
        
        self.criterion_q = torch.nn.MSELoss()
        self.criterion_aux = torch.nn.CrossEntropyLoss()

        self.replay_buffer = ReplayBuffer()
        self.image_buffer = ImageBuffer()

        self.t_step = 0



    def act(self, state):
        """
        Choose the an action in a given state using a espilon-greedy policy.

        Args:
            state (_type_ ?): state where the agent is.

        Returns:
            action (int): integer corresponging to the choosen action
        """
        if np.random.rand() > self.epsilon():
            return np.random.randint(high=self.context.action_size)
        else:
            self.q_network_local.eval()
            with torch.no_grad():
                q_values = self.q_network_local.forward(state)
            self.q_network_local.train()

            return np.argmax(q_values).item()


    def step(self, state, action, reward, next_state, done):

        self.replay_buffer.add(state, action, reward, next_state, done)

        if self.t_step % self.config.add_image_every_every == 0:
            banana_labels = None
            self.image_buffer.add(state, banana_labels)
        pass

    def learn_bananas_detection():
        pass

    def learn():
        pass