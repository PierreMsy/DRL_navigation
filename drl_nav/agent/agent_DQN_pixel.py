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
        self.epsilon = ExponentialSchedule.instantiate_from_config(config)

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

        self.record_loss_image = dict()
        self.record_loss = dict()

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

        self.t_step += 1
        self.replay_buffer.add(state, action, reward, next_state, done)

        if self.t_step % self.config.add_image_every == 0:
            
            # TODO Avoid so much torch numpy conversions.
            labels_banana = self.labelizer(state)
            banana_labels = torch.from_numpy(labels_banana).float()
            self.image_buffer.add(state, banana_labels)

        if self.t_step % self.config.learn_every == 0:
            self.learn()

        if self.t_step % self.config.learn_detection_every == 0:
            self.learn_bananas_detection()

    def learn_bananas_detection(self):

        images, labels = self.image_buffer.sample()
        preds = self.auxiliary_network(images)

        self.auxiliary_network.optimizer.zero_grad()
        loss = self.criterion_aux(preds, labels)
        loss.backward()
        self.auxiliary_network.step()

        self.record_loss_images[self.t_step] = loss.item()

    def learn(self):
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        with torch.no_grad():
            q_ns_local = self.q_network_local(next_states).detach()
            q_ns_target = self.q_network_target(next_states).detach()
            max_q_ns = q_ns_target(torch.argmax(q_ns_local, axis=1))

        q_target = rewards + self.config.gamma * max_q_ns * (1 - dones)

        q = self.q_network_local(states)[actions]

        self.q_network_local.optimizer.zero_grad()
        loss = self.criterion_q(q, q_target)
        loss.backward()
        self.q_network_local.optimizer.step()

        # Soft updates the target network
        self.soft_update(self.q_network_local, self.q_network_target, self.config.tau)

        self.record_loss[self.t_step] = loss.item()

        