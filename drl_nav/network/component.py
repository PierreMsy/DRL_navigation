from drl_nav.network.bodies import ConvBody, DummyBody
from drl_nav.network.heads import QNet, DuelingQNet


class NetworkCreator():
    '''
    Factory that build networks.
    '''
    def __init__(self):
        self.body_builders = {
            'conv': lambda kwargs: ConvBody(**kwargs),
            'dummy': lambda kwargs: DummyBody(**kwargs),
        }
        self.head_builders = {
            'qnet' : lambda network_body, action_size, config: QNet(network_body, config, action_size),
            'dueling_qnet' : lambda network_body, action_size, config: DuelingQNet(network_body, config, action_size),
        }

    def create_body(self, network_config):
        return self.body_builders[network_config.type](network_config.to_dict())
    
    def create_head(self, network_body, network_config, action_size):
        return self.head_builders[network_config.type](network_body, action_size, network_config)
    