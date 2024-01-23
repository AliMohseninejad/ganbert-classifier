import torch
import torch.nn as nn

"""Here The loss functions of the GAN model are defined
An example implementation is as follows:
>> class MeanSquaredError(nn.Module):
>>     def __init__(self):
>>         super(MeanSquaredError, self).__init__()
>> 
>>     def forward(self, input, target):
>>         # Calculate mean squared error
>>         loss = torch.mean((input - target)**2)
>>         return loss
"""


class GeneratorLossFunction(nn.Module):
    def __init__(
        self,
    ):
        super(GeneratorLossFunction, self).__init__()

    def forward(
        self,
    ):
        pass


class DiscriminatorLossFunction(nn.Module):
    def __init__(
        self,
    ):
        super(DiscriminatorLossFunction, self).__init__()

    def forward(
        self,
    ):
        pass
