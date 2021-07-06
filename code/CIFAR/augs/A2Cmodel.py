import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, num_ops=12):
        super(ActorCritic , self).__init__()

        self.fea_extracter = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            # nn.ReLU(),
            nn.LeakyReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            # nn.ReLU(),
            nn.LeakyReLU(),

            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32)
        )
        # N 14 2 2
        # 32*4*4
        self.value = nn.Sequential(
            nn.Dropout(),   #! to be done
            nn.Linear(512, 256),
            # nn.ReLU(),
            nn.LeakyReLU(),

            nn.Dropout(), #! to be done
            nn.Linear(256, 1)
        )
        self.policy = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Conv2d(128, num_ops, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Softmax(dim=1)
        )

    def forward(self, image):
        state = self.fea_extracter(image)

        state_value = self.value(state.view(state.size(0),-1))
        operation_dist = self.policy(state)

        return operation_dist, state_value