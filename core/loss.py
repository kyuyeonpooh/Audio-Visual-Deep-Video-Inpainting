import torch
import torch.nn as nn
import torch.nn.functional as F


class AdversarialLoss(nn.Module):
    def __init__(self, loss_type='hinge', target_real_label=1., target_fake_label=0.):
        super(AdversarialLoss, self).__init__()
        self.loss_type = loss_type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if loss_type == 'nsgan':
            self.criterion = nn.BCELoss()
        elif loss_type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif loss_type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real: bool, is_discriminator: bool):
        if self.loss_type == 'hinge':
            if is_discriminator:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()
        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss


class AudioVisualAttentionLoss(nn.Module):
    def __init__(self, avnet):
        super(AudioVisualAttentionLoss, self).__init__()
        for param in avnet.parameters():
            param.requires_grad = False
        self.avnet = avnet
        self.mse_loss = nn.MSELoss()
    
    def forward(self, image_pred, image_gt, audio):
        av_attmap_pred = self.avnet(image_pred, audio)['av_attmap']
        av_attmap_gt = self.avnet(image_gt, audio)['av_attmap']
        return self.mse_loss(av_attmap_pred, av_attmap_gt)


class L2ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(L2ContrastiveLoss, self).__init__()
        self.margin = margin

    def l2_similarity(self, feature1, feature2):
        feature = feature1.expand(feature1.size(0), feature1.size(0), feature1.size(1)).transpose(0, 1)
        sim_scores = torch.norm(feature - feature2, p=2, dim=2)
        return sim_scores

    def forward(self, feature1, feature2):
        sim_scores = self.l2_similarity(feature1, feature2)
        diagonal_dist = sim_scores.diag()
        cost_s = (self.margin - sim_scores).clamp(min=0)

        # clear diagonals
        mask = torch.eye(sim_scores.size(0)) > .5
        I = mask
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)

        loss = (torch.sum(cost_s ** 2) + torch.sum(diagonal_dist ** 2)) / (2 * feature1.size(0))
        return loss