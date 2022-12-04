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
        self.config = config
        self.epsilon = ExponentialSchedule.instantiate_from_config(config)

        conv_body = ConvBody()
        self.q_network_local = QNet(conv_body, context.action_size)
        self.q_network_target = QNet(ConvBody(), context.action_size)
        self.q_network_target.load_state_dict(self.q_network_local.state_dict())
        self.auxiliary_network = AuxNet(conv_body)
        self.labelizer = LabelizerNet()
        
        self.criterion_q = torch.nn.MSELoss()
        self.criterion_aux = torch.nn.CrossEntropyLoss()

        self.replay_buffer = ReplayBuffer(config.buffer_size)
        self.image_buffer = ImageBuffer(config.image_buffer_size)

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
            return np.random.randint(low=0, high=self.context.action_size)
        else:
            state = torch.from_numpy(np.moveaxis(state, 3, 1)).float().to(self.config.device)

            self.q_network_local.eval()
            with torch.no_grad():
                q_values = self.q_network_local.forward(state)
            self.q_network_local.train()

            return np.argmax(q_values).item()

    def step(self, state, action, reward, next_state, done):
        """

        """
        self.t_step += 1
        self.replay_buffer.add(state, action, reward, next_state, done)

        if self.t_step % self.config.add_image_every == 0:
            labels_banana = self.labelizer(state)
            self.image_buffer.add(state, labels_banana)

        if ((self.t_step % self.config.learn_every == 0) and
                (len(self.replay_buffer) > self.config.batch_size)
            ):
            self.learn()

        if ((self.t_step % self.config.learn_detection_every == 0) and
                (len(self.image_buffer) > self.config.image_batch_size)
            ):
            self.learn_bananas_detection()

    def learn_bananas_detection(self):
        """

        """
        images, labels = self.image_buffer.sample(self.config.image_batch_size)
        preds = self.auxiliary_network(images)

        self.auxiliary_network.optimizer.zero_grad()
        loss = self.criterion_aux(preds, labels)
        loss.backward()
        self.auxiliary_network.step()

        self.record_loss_images[self.t_step] = loss.item()

    def learn(self):
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)

        with torch.no_grad():
            q_ns_local = self.q_network_local(next_states)
            q_ns_target = self.q_network_target(next_states)
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

        