import torch
import torch.nn as nn
from torch.nn import functional as F

# Define a custom neural network model class called Learner
class Learner(nn.Module):
    def __init__(self, input_dim=2048, drop_p=0.0):
        super(Learner, self).__init__()
        
        # Define the classifier as a sequential neural network
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),  # Fully connected layer with 2048 input features and 512 output features
            nn.ReLU(),                 # Rectified Linear Unit (ReLU) activation function
            nn.Dropout(drop_p),        # Dropout layer with a dropout probability of drop_p
            nn.Linear(512, 32),        # Fully connected layer with 512 input features and 32 output features
            nn.ReLU(),                 # ReLU activation function
            nn.Dropout(drop_p),        # Dropout layer with a dropout probability of drop_p
            nn.Linear(32, 1),          # Fully connected layer with 32 input features and 1 output feature
            nn.Sigmoid()               # Sigmoid activation function to output values in the range [0, 1]
        )
        
        # Store the dropout probability as an attribute
        self.drop_p = drop_p
        
        # Initialize the weights of the neural network layers
        self.weight_init()
        
        # Create a list of learnable parameters
        self.vars = nn.ParameterList()
        
        # Iterate through the parameters of the classifier and add them to the parameter list
        for i, param in enumerate(self.classifier.parameters()):
            self.vars.append(param)

    def weight_init(self):
        # Initialize the weights of the linear layers using Xavier initialization
        for layer in self.classifier:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x, vars=None):
        # Custom forward method for the network
        if vars is None:
            vars = self.vars
        # Apply linear transformation followed by ReLU activation and dropout for the first layer
        x = F.linear(x, vars[0], vars[1])
        x = F.relu(x)
        x = F.dropout(x, self.drop_p, training=self.training)
        
        # Apply linear transformation followed by dropout for the second layer
        x = F.linear(x, vars[2], vars[3])
        x = F.dropout(x, self.drop_p, training=self.training)
        
        # Apply linear transformation for the final layer and apply sigmoid activation
        x = F.linear(x, vars[4], vars[5])
        return torch.sigmoid(x)

    def parameters(self):
        """
        Override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars
