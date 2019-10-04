import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNIQAnet(nn.Module):
    def __init__(self, config):
        super(CNNIQAnet, self).__init__()
        if config['is_gray']:
            self.conv1  = nn.Conv2d(1, 50, 7)
        else:
            self.conv1  = nn.Conv2d(3, 50, 7)
        # self.gn1    = nn.GroupNorm(10, 50)
        self.fc1    = nn.Linear(2 * 50, 800)
        self.fc2    = nn.Linear(800, 800)
        self.fc3    = nn.Linear(800, 1)
        self.dropout = nn.Dropout(p=0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)
            # elif isinstance(m, nn.GroupNorm):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0.0)
    def forward(self, x):
        x  = x.view(-1, x.size(-3), x.size(-2), x.size(-1))  #

        h  = self.conv1(x)

        # h = self.gn1(h)
        # h1 = F.adaptive_max_pool2d(h, 1)
        # h2 = -F.adaptive_max_pool2d(-h, 1)
        h1 = F.max_pool2d(h, (h.size(-2), h.size(-1)))
        h2 = -F.max_pool2d(-h, (h.size(-2), h.size(-1)))
        h  = torch.cat((h1, h2), 1)  # max-min pooling
        h  = h.squeeze(3).squeeze(2)

        h  = F.relu(self.fc1(h))
        h  = self.dropout(h)
        h  = F.relu(self.fc2(h))

        q  = self.fc3(h)
        return q