import torch
import torch.nn as nn

class Generator(nn.Module):
    """Implement the generator model discussed in the paper"""
    def __init__(self, noise_size=100, output_size=768, hidden_size=768, dropout_rate=0.1):
        super(Generator, self).__init__()
        layers = []
        hidden_sizes = [noise_size, hidden_size]
        for i in range(len(hidden_sizes)-1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout_rate)])
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, noise):
        output_rep = self.layers(noise)
        return output_rep

# In this code, the noise_size is set to 100, the output_size and hidden_size are both set to 768.
# The LeakyReLU activation function and Dropout are used in the design of the generator network. 
# The forward function takes a 100-dimensional noise vector 
# and outputs a 768-dimensional vector.

