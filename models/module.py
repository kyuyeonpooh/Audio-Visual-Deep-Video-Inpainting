import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, activation=None):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.activation = nn.LeakyReLU(0.2, inplace=True) if not activation else activation
    
    def forward(self, x):
        return self.activation(self.conv(x))


class Linear(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None):
        super(Linear, self).__init__()
        self.fc = nn.Linear(in_feats, out_feats)
        self.activation = nn.LeakyReLU(0.2, inplace=True) if not activation else activation
    
    def forward(self, x):
        return self.activation(self.fc(x))


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_num_params(self):
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of parameters: {num_params}")

    def init_weights(self, init_type="normal", gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
                if init_type == "normal":
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == "xavier":
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == "kaiming":
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif classname.find("BatchNorm") != -1 or classname.find("InstanceNorm") != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)        
        self.apply(init_func)

