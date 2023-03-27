from drl_nav.network.bodies import ConvBody, DummyBody


class NetworkCreator():
    '''
    Factory that build networks.
    '''
    def __init__(self):
        self.builders = {
            'conv': lambda args, kwargs : ConvBody(args, **kwargs),
            'dummy' : lambda args, kwargs : DummyBody(args, **kwargs),
        }

    def create(self, network, args, kwargs):
        return self.builders[network](args, kwargs)