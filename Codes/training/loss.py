import torch
import torch.nn as nn
import torch.nn.functional as F

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
        discriminator_fake_probs,
        Discriminator_real_features,
        Discriminator_fake_features,
    ):
        Loss_G_Unsupervised = -torch.mean(
            torch.log(1 - discriminator_fake_probs[:, -1] + 1e-8)
        )  # [:, -1] is used to extract the probabilities of the "extra" or "fake" class from the gen_probs tensor.
        # -------------------------------------------------------------------------------------
        # Feature Matching Loss
        Loss_FeatureMatching = torch.mean(
            torch.pow(
                torch.mean(Discriminator_real_features, dim=0)
                - torch.mean(Discriminator_fake_features, dim=0),
                2,
            )
        )  # represents the discrepancy between the real and generated feature distributions.
        # -------------------------------------------------------------------------------------
        # total loss (Generator1)
        Loss_G = Loss_FeatureMatching + Loss_G_Unsupervised
        return Loss_G


class DiscriminatorLossFunction(nn.Module):
    def __init__(
        self,
    ):
        super(DiscriminatorLossFunction, self).__init__()

    def forward(
        self,
        labels,
        supervised_indices,
        unsupervised_indices,
        real_logits,
        Discriminator_real_probability,
        Discriminator_fake_probability,
    ):
        # -------------------------------------------------------------------------------------
        Loss_Function = nn.CrossEntropyLoss(ignore_index=-1)
        # -------------------------------------------------------------------------------------
        if len(supervised_indices) == 0:
            Loss_D_supervised = torch.zeros((1,)).to(labels.device)
        else:
            Loss_D_supervised = Loss_Function(
                real_logits[supervised_indices],
                labels[supervised_indices],
            )

        # -------------------------------------------------------------------------------------
        # unsupervised loss
        Loss_Unsupervised_Real = -torch.mean(
            torch.log(1 - Discriminator_real_probability[:, -1] + 1e-8)
        )  # 1e-8 added to avoid numerical instability
        Loss_Unsupervised_Generated = -torch.mean(
            torch.log(Discriminator_fake_probability[:, -1] + 1e-8)
        )

        Loss_D_unsupervised = Loss_Unsupervised_Real + Loss_Unsupervised_Generated
        # -------------------------------------------------------------------------------------
        # total loss (Discriminator)
        Loss_D = Loss_D_supervised + Loss_D_unsupervised

        return Loss_D
