from torch import nn, manual_seed
import torch.nn.functional as F

class QNet(nn.Module):
    '''
    Q-network. 
    Deep network that learn the relationship between the states and the action values
    '''
    
    def __init__(self, state_size, action_size, seed):
        super(QNet, self).__init__()
        '''Neural network that maps the state to action values
        
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        '''

        hidden_layers = [64,64]
        
        self.seed = manual_seed(seed) 
        self.fc1 = nn.Linear(state_size, hidden_layers[0])
        self.do1 = nn.Dropout(p=.2)

        self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.do2 = nn.Dropout(p=.2)

        self.fc3 = nn.Linear(hidden_layers[1], action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
    
class Dueling_QNet(nn.Module):
    '''
    Q-network. 
    Deep network that learn the relationship between the states and the action values
    '''
    
    def __init__(self, state_size, action_size, seed, common_layer=64, state_hidden_layers=[64,64], advantage_hidden_layers=[64,64]):
        super(Dueling_QNet, self).__init__()
        '''Neural network that maps the state to action values
        
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            common_layer (int): The dimension of the common layer that build feature for both 
                                the state and advantages computation
            state_hidden_layers (list[int]): The size of the fully connected layers used to compute
                                             the state value. For the moment the size is fixed and equal to 2.
            advantage_hidden_layers (list[int]): The size of the fully connected layers used to compute
                                                 the advantages value. For the moment the size is fixed
                                                 and equal to 2.
        '''
        self.seed = manual_seed(seed) 
        self.fc_common = nn.Linear(state_size, common_layer)
        self.do_common = nn.Dropout(p=.1)
        
        self.fc_state_1 = nn.Linear(common_layer, state_hidden_layers[0])
        self.fc_state_2 = nn.Linear(state_hidden_layers[0], state_hidden_layers[1])
        self.fc_state_value = nn.Linear(state_hidden_layers[1], 1)

        self.fc_advantage_1 = nn.Linear(common_layer, advantage_hidden_layers[0])
        self.fc_advantage_2 = nn.Linear(advantage_hidden_layers[0], advantage_hidden_layers[1])
        self.fc_advantage_value = nn.Linear(advantage_hidden_layers[1], action_size)
        
    def feature_creation(self, x):
        return F.relu(self.fc_common(x))
    
    def state_computation(self, x):
        x = F.relu(self.fc_state_1(x))
        x = F.relu(self.fc_state_2(x))
        return self.fc_state_value(x)
    
    def advantage_computation(self, x):
        x = F.relu(self.fc_advantage_1(x))
        x = F.relu(self.fc_advantage_2(x))
        return self.fc_advantage_value(x)

    def forward(self, x):
        
        features = self.feature_creation(x)
        state_value = self.state_computation(features)
        advantage_values = self.advantage_computation(features)
    
        Q_values = state_value + advantage_values - advantage_values.mean()

        return Q_values