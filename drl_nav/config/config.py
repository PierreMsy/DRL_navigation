import os
import yaml


PATH_YML_DQN = os.path.join(
    os.path.dirname(__file__), r'./agen_dqn_config.yml')

class AgentConfiguration:

    def __init__(
        self,
        device=None,
        gamma=None,
        tau=None,
        batch_size=None,
        image_batch_size=None,
        buffer_size=None,
        image_buffer_size=None,
        learn_every=None,
        learn_detection_every=None,
        add_image_every=None,
        epsilon={},
        network_head={},
        network_body={},
        ) -> None:
        """
        """
        # Load base configuration
        with open(PATH_YML_DQN, "r") as f_yml:
            self.dict_config = yaml.safe_load(f_yml)

        # Update config dict based on the provided specific configurations    
        update_dict(self.dict_config['epsilon'], epsilon)
        update_dict(self.dict_config['network']['head'], network_head)
        update_dict(self.dict_config['network']['body'], network_body)

        # Set attribute
        self.set_attr('device', device)
        self.set_attr('gamma', gamma)
        self.set_attr('tau', tau)
        self.set_attr('batch_size', batch_size)
        self.set_attr('image_batch_size', image_batch_size)
        self.set_attr('buffer_size', buffer_size)
        self.set_attr('image_buffer_size', image_buffer_size)
        self.set_attr('learn_every', learn_every)
        self.set_attr('learn_detection_every', learn_detection_every)
        self.set_attr('add_image_every', add_image_every)
        self.epsilon = Epsilon(**self.dict_config['epsilon'])
        self.network_head = Network(**self.dict_config['network']['head'])
        self.network_body = Network(**self.dict_config['network']['body'])

    def set_attr(self, attr, value):
        '''
        If the value is different than None, update the config dictionary
        with it, otherwise, take the default dict's value.
        '''
        if value:
            setattr(self, attr, value)
            self.dict_config[attr] = value
        else:
            setattr(self, attr, self.dict_config[attr])

class Epsilon:
    """
    Esplion policy exploration configuration
    """
    def __init__(
        self,
        start_value,
        end_value,
        steps
        ) -> None:
        self.start_value = start_value
        self.end_value = end_value
        self.steps = steps

class Network:
    """
    Esplion policy exploration configuration
    """
    def __init__(
        self,
        type=None,
        hidden_layers=[],
        input_size: int=None,
        learning_rate:int=None,
        ) -> None:
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.type = type
        self.learning_rate = learning_rate
        
    def to_dict(self):
        return vars(self)

def update_dict(d_ref, d_ovr):
    """
    For every specific kv given in d_ovr, change the corresponding
    values in d_ref, the complete and udpated config dictionary.
    """

    for k_o, v_o in d_ovr.items():
        if k_o not in d_ref:
            d_ref[k_o] = v_o
        
    for k_r,v_r in d_ref.items():
        if k_r in d_ovr:
            if type(v_r) == dict:
                update_dict(d_ref[k_r], d_ovr[k_r])
            else:
                d_ref[k_r] = d_ovr[k_r]