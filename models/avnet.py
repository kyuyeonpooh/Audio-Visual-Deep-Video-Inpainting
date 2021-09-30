import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class AudioVisualAttention(nn.Module):
    def __init__(self, vis_net, aud_net, num_clusters=10, classifier_enabled=False):
        super(AudioVisualAttention, self).__init__()
        self.vis_net = vis_net
        self.aud_net = aud_net
        self.num_clusters = num_clusters
        self.classifier_enabled = classifier_enabled

        self.conv1_v = nn.Conv2d(512, 128, kernel_size=1)
        self.conv2_v = nn.Conv2d(128, 128, kernel_size=1)

        self.maxpool_a = nn.AdaptiveMaxPool2d(1)
        self.fc1_a = nn.Linear(512, 128)
        self.fc2_a = nn.Linear(128, 128)

        self.conv_av = nn.Conv2d(1, 1, kernel_size=1, bias=False)  # For scaling
        self.conv_av.weight.data.fill_(5.)
        self.maxpool_av = nn.AdaptiveMaxPool2d(1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.avgpool_v = nn.AdaptiveAvgPool2d(1)
        self.classifier_v = nn.Linear(512, self.num_clusters)
        self.classifier_a = nn.Linear(512, self.num_clusters)
        
    def forward(self, image_batch, audio_batch):
        vis_feat = self.vis_net(image_batch)
        vis_att_feat = self.relu(self.conv1_v(vis_feat))
        vis_att_feat = self.conv2_v(vis_att_feat)

        aud_feat = self.aud_net(audio_batch)
        aud_feat = self.maxpool_a(aud_feat)
        aud_feat = aud_feat.flatten(start_dim=1)
        aud_att_feat = self.relu(self.fc1_a(aud_feat))
        aud_att_feat = self.fc2_a(aud_att_feat)

        vis_att_feat = F.normalize(vis_att_feat, p=2, dim=1)
        aud_att_feat = F.normalize(aud_att_feat, p=2, dim=1)
        av_attmap = torch.einsum('nchw,nc->nhw', vis_att_feat, aud_att_feat).unsqueeze(1)
        
        av_attmap = self.conv_av(av_attmap)
        av_attmap = self.sigmoid(av_attmap)
        av_score = self.maxpool_av(av_attmap)
        av_score = av_score.squeeze()

        vis_logits = None
        aud_logits = None
        if self.classifier_enabled:
            vis_class_feat = self.avgpool_v(vis_feat)
            vis_class_feat = vis_class_feat.flatten(start_dim=1)
            vis_logits = self.classifier_v(vis_class_feat)
            aud_logits = self.classifier_a(aud_feat)

        return {
            'av_score': av_score,
            'av_attmap': av_attmap,
            'vis_feat': vis_feat,
            'aud_feat': aud_feat,
            'vis_logits': vis_logits,
            'aud_logits': aud_logits
        }

    def init_classifier(self, device):
        self.classifier_v = nn.Linear(512, self.num_clusters)
        self.classifier_v.weight.data.normal_(0, 0.01)
        self.classifier_v.bias.data.zero_()
        self.classifier_v.to(device)

        self.classifier_a = nn.Linear(512, self.num_clusters)
        self.classifier_a.weight.data.normal_(0, 0.01)
        self.classifier_a.bias.data.zero_()
        self.classifier_a.to(device)

        params = list(self.classifier_v.parameters()) + list(self.classifier_a.parameters())
        return params
    
    def enable_classifier(self):
        self.classifier_enabled = True

    def disable_classifier(self):
        self.classifier_enabled = False
