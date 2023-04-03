from drl_nav.network.bodies import ConvBody, DummyBody


class NetworkCreator():
    '''
    Factory that build networks.
    '''
    def __init__(self):
        self.builders = {
            'conv': lambda kwargs: ConvBody(**kwargs),
            'dummy' : lambda kwargs: DummyBody(**kwargs),
        }

    def create(self, network_config):
        return self.builders[network_config.type](network_config.to_dict())