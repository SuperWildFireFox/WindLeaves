import torch.nn
from torch import nn


def cal_conv_shape(shape, n):
    shape = list(shape)
    shape[0] = shape[0] // 2 ** n
    shape[1] = shape[1] // 2 ** n
    return shape[0] * shape[1]


FC_1_LENGTH = 128
FC_2_LENGTH = 256
FC_C_LENGTH = 512


class BasePPO(nn.Module):
    def __init__(self, image_shape, feature1_length, feature2_length):
        super().__init__()
        self.image_conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.Tanh(),
        )

        self.FC_features1 = nn.Sequential(
            nn.Linear(feature1_length, FC_1_LENGTH),
            nn.Tanh(),
            nn.Linear(FC_1_LENGTH, FC_1_LENGTH),
            nn.Tanh(),
        )

        self.FC_features2 = nn.Sequential(
            nn.Linear(feature2_length, FC_2_LENGTH),
            nn.Tanh(),
            nn.Linear(FC_2_LENGTH, FC_2_LENGTH),
            nn.Tanh(),
        )

        self.FC = nn.Linear(64 * cal_conv_shape(image_shape, 2) + FC_1_LENGTH + FC_2_LENGTH, FC_C_LENGTH)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 1.0)
                nn.init.constant_(module.bias, 1e-6)

    def forward(self, image, feature1, feature2):
        image = self.image_conv(image)
        feature1 = self.FC_features1(feature1)
        feature2 = torch.max(self.FC_features2(feature2), dim=1)[0]
        x = torch.cat([image.view(image.size()[0], -1), feature1, feature2], dim=1)
        x = self.FC(x)
        return x


class Actor(BasePPO):
    def __init__(self, num_actions, image_shape, feature1_length, feature2_length):
        super(Actor, self).__init__(image_shape, feature1_length, feature2_length)
        self.actor_linear = nn.Linear(FC_C_LENGTH, num_actions)
        self._initialize_weights()

    def forward(self, image, feature1, feature2, mask):
        x = super(Actor, self).forward(image, feature1, feature2)
        logic = self.actor_linear(x)
        logic = torch.where(mask, logic, torch.tensor(-1e8).to(x.device))
        return logic


class Critic(BasePPO):
    def __init__(self, image_shape, feature1_length, feature2_length):
        super(Critic, self).__init__(image_shape, feature1_length, feature2_length)
        self.critic_linear = nn.Linear(FC_C_LENGTH, 1)
        self._initialize_weights()

    def forward(self, image, feature1, feature2):
        x = super(Critic, self).forward(image, feature1, feature2)
        value = self.critic_linear(x)
        return value
