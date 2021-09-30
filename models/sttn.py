import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.spectral_norm import spectral_norm


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, activation=None):
        super(Conv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.activation = nn.LeakyReLU(0.2, inplace=True) if not activation else activation
    
    def forward(self, x):
        return self.activation(self.conv(x))


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True, recompute_scale_factor=True)


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def get_num_params(self):
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return num_params

    def init_weights(self, init_type="normal", gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
                if init_type == "normal":
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == "xavier":
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == "kaiming":
                    nn.init.kaiming_normal_(m.weight.data, mode="fan_in")
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif classname.find("BatchNorm") != -1 or classname.find("InstanceNorm") != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)        
        self.apply(init_func)


class Encoder(BaseNetwork):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            Conv2D(3, 64, kernel_size=3, stride=2, padding=1),
            Conv2D(64, 64, kernel_size=3, padding=1),
            Conv2D(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2D(128, 256, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return self.encoder(x)  # (B * T, C, H, W)


class Decoder(BaseNetwork):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            Upsample(),
            Conv2D(256, 128, kernel_size=3, padding=1),
            Conv2D(128, 64, kernel_size=3, padding=1),
            Upsample(),
            Conv2D(64, 64, kernel_size=3, padding=1),
            Conv2D(64, 3, kernel_size=3, padding=1, activation=nn.Tanh())
        )
    
    def forward(self, x):
        return self.decoder(x)


class FeedForward(nn.Module):
    def __init__(self, n_channels):
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
            Conv2D(n_channels, n_channels, kernel_size=3, padding=2, dilation=2),
            Conv2D(n_channels, n_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.ff(x)


class DotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention Module
    """
    def forward(self, query, key, value, masks):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        scores.masked_fill(masks, -1e9)
        att_value = torch.matmul(F.softmax(scores, dim=-1), value)
        return att_value


class MultiheadPatchAttention(nn.Module):
    """ Multihead Patch-based Spatio-Temporal Attention Module
    """
    def __init__(self, batch_size: int):
        super(MultiheadPatchAttention, self).__init__()
        self.batch_size = batch_size
        self.patch_size = [(56, 56), (28, 28), (14, 14), (7, 7)]
        self.num_heads = len(self.patch_size)

        self.conv_query = nn.Conv2d(256, 256, kernel_size=1)
        self.conv_key = nn.Conv2d(256, 256, kernel_size=1)
        self.conv_value = nn.Conv2d(256, 256, kernel_size=1)
        self.conv_output = Conv2D(256, 256, kernel_size=3, padding=1)

        self.attention = DotProductAttention()
    
    def reshape(self, x, patch_width, patch_height, hidden_dim):
        B, T, H, W = self.B, self.T, self.H, self.W
        out_w, out_h = W // patch_width, H // patch_height
    
        x = x.view(B, T, hidden_dim, out_h, patch_height, out_w, patch_width)
        x = x.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        x = x.view(B, T * out_h * out_w, patch_height * patch_width * hidden_dim)
        if hidden_dim == 1:  # in case of mask
            x = (x.mean(-1) > 0.5).unsqueeze(1).repeat(1, T * out_h * out_w, 1)
        return x

    def forward(self, frame_feats, masks):
        BT, C, H, W = frame_feats.shape
        B = self.batch_size
        T = BT // B
        hidden_dim = C // len(self.patch_size)
        self.B, self.T, self.H, self.W = B, T, H, W
        
        Q = self.conv_query(frame_feats)
        K = self.conv_key(frame_feats)
        V = self.conv_value(frame_feats)
        outputs = []
        bundle = zip(
            self.patch_size,
            torch.chunk(Q, self.num_heads, dim=1),
            torch.chunk(K, self.num_heads, dim=1),
            torch.chunk(V, self.num_heads, dim=1)
        )        

        for (w, h), q, k, v in bundle:
            mm = self.reshape(masks, patch_width=w, patch_height=h, hidden_dim=1)
            query = self.reshape(q, patch_width=w, patch_height=h, hidden_dim=hidden_dim)
            key = self.reshape(k, patch_width=w, patch_height=h, hidden_dim=hidden_dim)
            value = self.reshape(v, patch_width=w, patch_height=h, hidden_dim=hidden_dim)
            out = self.attention(query, key, value, mm)
            out = out.view(self.B, self.T, self.H // h, self.W // w, hidden_dim, h, w)
            out = out.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
            out = out.view(BT, hidden_dim, self.H, self.W)
            outputs.append(out)

        outputs = torch.cat(outputs, dim=1)
        outputs = self.conv_output(outputs)
        return outputs


# Multi-Head Spatio-Temporal Transformer
class TransformerBlock(nn.Module):
    def __init__(self, batch_size: int):
        super(TransformerBlock, self).__init__()
        self.multihead_attention = MultiheadPatchAttention(batch_size=batch_size)
        self.feed_forward = FeedForward(n_channels=256)
    
    def forward(self, x):
        frame_feats, masks = x['frame_feats'], x['masks']
        frame_feats = frame_feats + self.multihead_attention(frame_feats, masks)
        frame_feats = frame_feats + self.feed_forward(frame_feats)        
        return {'frame_feats': frame_feats, 'masks': masks}


class STTN(BaseNetwork):
    def __init__(self, batch_size: int):
        super(STTN, self).__init__()
        self.encoder = Encoder()
        self.transformers = nn.Sequential(*[TransformerBlock(batch_size=batch_size)] * 8)
        self.decoder = Decoder()

        self.init_weights()

    def forward(self, frames, masks):
        masked_frames = frames * (1 - masks).float()
        B, T, _, H, W = masked_frames.shape
        masked_frames = masked_frames.view(B * T, -1, H, W)
        frame_feats = self.encoder(masked_frames)  # (B * T, 3, H, W)
        masks = F.interpolate(masks.view(B * T, 1, H, W), scale_factor=1/4, recompute_scale_factor=True)
        frame_feats = self.transformers({'frame_feats': frame_feats, 'masks': masks})['frame_feats']
        pred_frames = self.decoder(frame_feats)
        return pred_frames


class Discriminator(BaseNetwork):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels, 64, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(64, 128, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(128, 256, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(256, 256, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(256, 256, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 256, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2), bias=False)
        )

        self.init_weights()

    def forward(self, xs):
        xs_t = torch.transpose(xs, 0, 1)
        xs_t = xs_t.unsqueeze(0)           # (B, C, T, H, W)
        feat = self.conv(xs_t)
        out = torch.transpose(feat, 1, 2)  # (B, T, C, H, W)
        return out
