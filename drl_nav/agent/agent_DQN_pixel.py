import numpy as np
import torch

from drl_nav.utils.schedule import ExponentialSchedule
from drl_nav.network import QNet, AuxNet, LabelizerNet, NetworkCreator
from drl_nav.component import ReplayBuffer2, ImageBuffer2
from drl_nav.config.constant import Input_type as INPUT


class Agent_DQN_pixel:
    '''
    Deep Q Learning agent that will interact with the environment 
    to maximize its expected reward.
    '''
    def __init__(self, context, config):
        self.context = context
        self.config = config
        self.epsilon = ExponentialSchedule.instantiate_from_config(config)

        network_body = NetworkCreator().create(config.network_body)
        self.q_network_local = QNet(network_body, config.network_head, context.action_size)
        self.q_network_target = QNet(
            NetworkCreator().create(config.network_body),
            config.network_head, context.action_size)
        self.q_network_target.load_state_dict(self.q_network_local.state_dict())
        self.auxiliary_network = AuxNet(network_body)
        self.labelizer = LabelizerNet()
        
        self.criterion_q = torch.nn.MSELoss()
        self.criterion_aux = torch.nn.BCELoss()

        self.replay_buffer = ReplayBuffer2(config.buffer_size, config.device)
        self.image_buffer = ImageBuffer2(config.image_buffer_size, config.device)

        # self.mask_learning = [10_000, 20_000]

        self.record_loss_image = dict()
        self.record_loss = dict()
        self.record_action = dict()

        self.t_step = 0

    def act(self, state):
        """
        Choose the an action in a given state using a espilon-greedy policy.
        Args:
            state (_type_ ?): state where the agent is.
        Returns:
            action (int): integer corresponging to the choosen action
        """
        if np.random.rand() < self.epsilon():
            return np.random.randint(low=0, high=self.context.action_size)
        else:
            state = self.convert_state_to_torch(state) 

            self.q_network_local.eval()
            with torch.no_grad():
                q_values = self.q_network_local.forward(state)
            self.q_network_local.train()

            action = np.argmax(q_values).item()
            self.record_action[self.t_step] = action

            return action 

    def step(self, state, action, reward, next_state, done):
        """
        Register environment infos in buffer and trigger when needed the learning
        of the task or the bananas detection.
        """
        self.t_step += 1
        state = torch.squeeze(self.convert_state_to_torch(state))
        next_state = torch.squeeze(self.convert_state_to_torch(next_state))
        self.replay_buffer.add(state, action, reward, next_state, done)

        if self._must_labelize_image():
            labels_banana = self.labelizer(state)
            self.image_buffer.add(state, labels_banana)

        if self._must_learn():
            self.learn()

        if self._must_learn_bananas_detection():
            self.learn_bananas_detection()

    def learn_bananas_detection(self):
        """
        Make one optimizer step for the supervised taks of recognizing directionaly 
        colored bananas.
        """
        images, labels = self.image_buffer.sample(self.config.image_batch_size)
        preds = self.auxiliary_network(images)

        self.auxiliary_network.optimizer.zero_grad()
        loss = self.criterion_aux(preds, labels)
        loss.backward()
        self.auxiliary_network.optimizer.step()

        self.record_loss_image[self.t_step] = loss.item()

    def learn(self):
        """
        double DQN :
            TD_target = r + gamma * q(S', argmax_a(q(S',a,w)), w')
            Δw = α(TD_target - q(S, A, w))∇w(q(S, A, w))
        """
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config.batch_size)

        with torch.no_grad():
            q_ns_local = self.q_network_local(next_states)
            q_ns_target = self.q_network_target(next_states)
            max_q_ns = q_ns_target.gather(1, torch.argmax(q_ns_local, axis=1).unsqueeze(1))
        q_target = rewards + self.config.gamma * max_q_ns * (1 - dones)
        q = self.q_network_local(states).gather(1, actions)

        self.q_network_local.optimizer.zero_grad()
        loss = self.criterion_q(q, q_target)
        loss.backward()
        self.q_network_local.optimizer.step()

        # Soft updates the target network   
        self.soft_update(self.q_network_local, self.q_network_target, self.config.tau)

        self.record_loss[self.t_step] = loss.item()

    def soft_update(self, net_local, net_target, tau):
        """
        Soft update model parameters.
            θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for local_param, target_param in zip(net_local.parameters(), net_target.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def _must_learn(self) -> bool:
        _must_learn = True
        _must_learn &= self.t_step % self.config.learn_every == 0
        _must_learn &= len(self.replay_buffer) > self.config.batch_size
        return _must_learn

    def _must_learn_bananas_detection(self) -> bool:
        _must_learn = True
        _must_learn &= self.context.input_type == INPUT.IMAGE
        _must_learn &= self.t_step % self.config.learn_detection_every == 0
        _must_learn &= len(self.image_buffer) > self.config.image_batch_size
        return _must_learn
    
    def _must_labelize_image(self) -> bool:
        _must_labelize = True
        _must_labelize &= self.context.input_type == INPUT.IMAGE
        _must_labelize &= self.t_step % self.config.add_image_every == 0
        return _must_labelize
    
    def convert_state_to_torch(self, state) -> torch.tensor:
        if self.context.input_type == INPUT.IMAGE:
            return torch.from_numpy(np.moveaxis(state, 3, 1)).float().to(self.config.device)
        elif self.context.input_type == INPUT.VECTOR:
            return torch.from_numpy(state).float().to(self.config.device)
        else:
            raise Exception(f'unknow input_type {self.config.input_type}')