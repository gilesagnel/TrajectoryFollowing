import torch as T
from torch import nn
import torch.nn.functional as F
from models.network import get_base_cnn

class CNN(nn.Module):
    def __init__(self, image_dim, state_dim, out_state_dim=200, device="cuda"):
        super().__init__()  

        self.model = get_base_cnn()
        # nn.Sequential(
        #     nn.Conv2d(image_dim[0], 3, 11, stride=3, padding=1),
        #     nn.PReLU(),
        #     nn.MaxPool2d(5, stride=2),
        #     nn.Conv2d(3, 6, 3),
        #     nn.PReLU(),
        #     nn.MaxPool2d(5, stride=2),
        #     nn.Flatten()
        # )

        cnn_nc = 256

        self.fc = nn.Sequential(
            nn.Linear(cnn_nc, cnn_nc),
            nn.Tanh()
        )
        
        self.l1 = nn.Sequential(
            nn.Linear(state_dim, cnn_nc),
            nn.Tanh()
        )

        self.l2 = nn.Sequential(
            nn.Linear(cnn_nc, out_state_dim),
            nn.Tanh()
        )

        self.to(device)

    def forward(self, image, state):
          out1 = self.fc(self.model(image).flatten(start_dim=1))
          out2 = self.l1(state)
          out = out1 + out2
          return self.l2(out)


class Agent(nn.Module):
    def __init__(self, image_size, state_dim, device="cuda"):
        super().__init__()  

        self.cnn = CNN(image_size, state_dim, out_state_dim=100, device=device)
        self.model = nn.Sequential(
            nn.Linear(100, 50),
            nn.Tanh()
        )
        self.lv_model = nn.Linear(50, 5)
        self.av_model = nn.Linear(50, 5)
        self.to(device)

    def forward(self, image_state, env_state):
        features = self.cnn(image_state, env_state)
        out = self.model(features)
        lv_scores = F.softmax(self.lv_model(out), dim=-1)
        av_scores = F.softmax(self.av_model(out), dim=-1)
        return T.cat([lv_scores, av_scores], dim=1)