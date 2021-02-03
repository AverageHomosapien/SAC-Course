import numpy as np
import torch as T
import torch.nn.functional as F
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(PolicyNetwork, self).__init__()

        # fully connected
        self.fc1 = nn.Linear(*input_dims, 128) # * unpacks a list
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        if T.cuda.is_available():
            print("CAN ACCESS")
        else:
            print("BOOOOOO")

        self.to(self.device)

    def forward(self, state): # output goes through forward
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
