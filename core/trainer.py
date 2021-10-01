import os

import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from core.dataset import AVEInpainting, MUSICSoloInpainting
from core.loss import *
from models.sttn import *
from models.avnet import AudioVisualAttention
from models.resnet import resnet18


class Trainer:
    def __init__(self, args: dict):
        self.args = args
        self.name_of_run = args['name']
        self.path_args = args['path']
        self.dataset_args = args['dataset']
        self.loss_args = args['loss']
        self.train_args = args['train']
        self.device = torch.device('cuda')

        # Set log directories and writers
        self.log_dir = os.path.join(self.path_args['log_dir'], self.name_of_run)
        self.checkpoint_dir = os.path.join(self.path_args['checkpoint_dir'], self.name_of_run)
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.summary = {}
        self.scalar_writer = SummaryWriter(os.path.join(self.log_dir, 'scalar', self.name_of_run))
        self.image_writer = SummaryWriter(os.path.join(self.log_dir, 'image', self.name_of_run))
        self.iteration = 0

        # Set dataset and data loader
        self.audio_flag = self.loss_args['lambda_av_att'] > 0 or self.loss_args['lambda_av_cls'] > 0
        dataset_map = {
            'AVE': AVEInpainting,
            'MUSIC-Solo': MUSICSoloInpainting
        }
        dataset_class = dataset_map[self.dataset_args['name']]
        self.train_dataset = dataset_class(
            split='train',
            dataset_args=self.dataset_args,
            mask_type=self.dataset_args['mask_type'],
            get_audio=self.audio_flag
        )
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_args['batch_size'],
            shuffle=True,
            drop_last=True,
            num_workers=self.train_args['num_workers'],
            pin_memory=True
        )

        # Set basical loss functions
        self.adv_loss = AdversarialLoss(loss_type=self.loss_args['adv_loss_type'])
        self.adv_loss = self.adv_loss.to(self.device)
        self.l1_loss = nn.L1Loss()

        # Our proposed loss functions
        if self.audio_flag:
            vis_net = resnet18(modal='V')
            aud_net = resnet18(modal='A')
            self.av_net = AudioVisualAttention(vis_net, aud_net, self.loss_args['num_clusters'])
            self.av_net.load_state_dict(torch.load(self.path_args['avnet_ckpt_file'], map_location=self.device))
            self.av_net.eval()
            self.av_net.to(self.device)
            for param in self.av_net.parameters():
                param.requires_grad = False
            if self.loss_args['lambda_av_att'] > 0:
                self.av_net.disable_classifier()
                self.mse_loss = nn.MSELoss()
            if self.loss_args['lambda_av_cls'] > 0:
                self.av_net.enable_classifier()
                self.ce_loss = nn.CrossEntropyLoss()
            self.imagenet_mean = torch.cuda.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            self.imagenet_std = torch.cuda.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        # Set models and optimizers
        self.netG = STTN(batch_size=self.train_args['batch_size'] // torch.cuda.device_count())
        self.netG = self.netG.to(self.device)
        self.netD = Discriminator()
        self.netD = self.netD.to(self.device)
        self.optimG = optim.Adam(
            self.netG.parameters(),
            lr=self.train_args['lr'],
        )
        self.optimD = optim.Adam(
            self.netD.parameters(),
            lr=self.train_args['lr'],
        )
        self.load()
        if torch.cuda.device_count() > 1:
            self.netG = DataParallel(self.netG)
            self.netD = DataParallel(self.netD)

    def load(self):
        if os.path.isfile(os.path.join(self.checkpoint_dir, 'latest_ckpt.txt')):
            with open(os.path.join(self.checkpoint_dir, 'latest_ckpt.txt')) as txt:
                latest_iter = txt.read().splitlines()[-1]
        else:
            ckpts = [os.path.basename(i).split('.pth')[0] for i in glob(os.path.join(self.checkpoint_dir, '*.pth'))]
            ckpts = sorted(ckpts)
            latest_iter = ckpts[-1] if len(ckpts) > 0 else None
        if latest_iter:
            latest_iter = int(latest_iter)
            ckpt_path = os.path.join(self.checkpoint_dir, f'iter_{str(latest_iter).zfill(6)}.pth')
            print(f'Loading model from {ckpt_path} ...')
            ckpt_bundle = torch.load(ckpt_path, map_location=self.device)
            self.netG.load_state_dict(ckpt_bundle['netG'])
            self.netD.load_state_dict(ckpt_bundle['netD'])
            self.optimG.load_state_dict(ckpt_bundle['optimG'])
            self.optimD.load_state_dict(ckpt_bundle['optimD'])
            self.iteration = ckpt_bundle['iteration']
        else:
            print('No trained model found. An initialized model will be used.')
            if self.dataset_args['name'] == 'MUSIC-Solo':
                ave_ckpt_bundle = torch.load(self.path_args['ave_ckpt_path'], map_location=self.device)
                self.netG.load_state_dict(ave_ckpt_bundle['netG'])
                self.netD.load_state_dict(ave_ckpt_bundle['netD'])
                print('Loaded pretrained weights from', self.path_args['ave_ckpt_path'])

    def save(self, iteration: int):
        ckpt_path = os.path.join(self.checkpoint_dir, f'iter_{str(iteration).zfill(6)}.pth')
        print(f'\nSaving model to {ckpt_path} ...')
        ckpt_bundle = {
            'netG': self.netG.module.state_dict(),
            'netD': self.netD.module.state_dict(),
            'optimG': self.optimG.state_dict(),
            'optimD': self.optimD.state_dict(),
            'iteration': self.iteration
        }
        torch.save(ckpt_bundle, ckpt_path)
        with open(os.path.join(self.checkpoint_dir, 'latest_ckpt.txt'), 'w') as txt:
            txt.write(str(self.iteration).zfill(5))
    
    def update_lr(self):
        decay_rate = 0.1 ** (
            min(self.iteration, self.train_args['iter_lr_update_2']) // self.train_args['iter_lr_update_1']
        )
        new_lr = self.train_args['lr'] * decay_rate
        if new_lr != self.optimG.param_groups[0]['lr']:
            for param_group in self.optimG.param_groups:
                param_group['lr'] = new_lr
            for param_group in self.optimD.param_groups:
                param_group['lr'] = new_lr

    def add_scalar_summary(self, tag, value):
        if tag not in self.summary:
            self.summary[tag] = 0
        self.summary[tag] += value
        if self.iteration % self.train_args['iter_log'] == 0:
            self.scalar_writer.add_scalar(
                tag, self.summary[tag] / self.train_args['iter_log'], self.iteration
            )
            self.summary[tag] = 0.
    
    def add_image_summary(self, tag, image):
        if self.iteration % self.train_args['iter_image_log'] == 0:
            t = image.shape[0] // self.train_args['batch_size']
            grid = make_grid((image * 0.5 + 0.5), nrow=t)
            grid = F.interpolate(grid.unsqueeze(0), scale_factor=1/2, recompute_scale_factor=True)
            self.image_writer.add_image(tag, grid.squeeze(), self.iteration)
            self.image_writer.flush()
    
    def add_att_summary(self, tag, attmap, image):
        if self.iteration % self.train_args['iter_image_log'] == 0:
            res = list()
            attmap = attmap.numpy()
            image = (image * self.imagenet_std.cpu() + self.imagenet_mean.cpu())
            image = image.permute(0, 2, 3, 1).contiguous().numpy()
            t = image.shape[0] // self.train_args['batch_size']
            for i in range(len(attmap)):
                am = attmap[i, 0]
                am = cv2.resize(am, (224, 224), interpolation=cv2.INTER_NEAREST) * 255
                am = cv2.applyColorMap(am.astype(np.uint8), cv2.COLORMAP_HOT)
                am = 0.6 * image[i] + 0.4 * (am[:, :, ::-1] / 255)
                res.append(am)
            res = torch.from_numpy(np.array(res)).permute(0, 3, 1, 2)
            grid = make_grid(res, nrow=t)
            grid = F.interpolate(grid.unsqueeze(0), scale_factor=1/2, recompute_scale_factor=True)
            self.image_writer.add_image(tag, grid.squeeze(), self.iteration)

    def train(self):
        pbar = range(int(self.train_args['iter_total']))
        pbar = tqdm(pbar, initial=self.iteration, dynamic_ncols=True, smoothing=0.01)
        while self.iteration < self.train_args['iter_total']:
            self.train_iter(pbar)
        print(f'\n\nEnd training {self.name_of_run}!')
    
    def train_iter(self, pbar):
        for data in self.train_loader:
            self.update_lr()
            self.iteration += 1

            # frames, masks = data['frames'], data['masks']
            if self.audio_flag:
                frames, masks, specgrams = data
                frames, masks, specgrams = frames.to(self.device), masks.to(self.device), specgrams.to(self.device)
            else:
                frames, masks = data
                frames, masks = frames.to(self.device), masks.to(self.device)
            B, T, _, H, W = frames.shape
            
            pred_frames = self.netG(frames, masks)

            frames = frames.view(B * T, 3, H, W)
            masks = masks.view(B * T, 1, H, W)
            comp_frames = frames * (1 - masks).float() + pred_frames * masks

            self.add_image_summary('image/1.input', (frames * (1 - masks)).cpu().detach())
            self.add_image_summary('image/2.output', comp_frames.cpu().detach())
            self.add_image_summary('image/3.GT', frames.cpu().detach())

            lossD, lossG = 0., 0.

            # Discriminator GAN loss
            real_feat = self.netD(frames)
            fake_feat = self.netD(comp_frames.detach())
            real_lossD = self.adv_loss(real_feat, is_real=True, is_discriminator=True)
            fake_lossD = self.adv_loss(fake_feat, is_real=False, is_discriminator=True)
            lossD += (real_lossD + fake_lossD) / 2
            self.optimD.zero_grad()
            lossD.backward()
            self.optimD.step()
            self.add_scalar_summary('lossD/1.lossD', lossD.item())
            self.add_scalar_summary('lossD/2.real_lossD', real_lossD.item())
            self.add_scalar_summary('lossD/3.fake_lossD', fake_lossD.item())

            # Generator adversarial loss
            gen_feat = self.netD(comp_frames)
            adv_loss = self.adv_loss(gen_feat, is_real=True, is_discriminator=False)
            adv_loss *= self.loss_args['lambda_adv']
            lossG += adv_loss
            self.add_scalar_summary('lossG/4.adv_loss', adv_loss.item())

            # Generator L1 loss
            hole_loss = self.l1_loss(pred_frames * masks, frames * masks) / torch.mean(masks)
            hole_loss *= self.loss_args['lambda_l1']
            lossG += hole_loss
            self.add_scalar_summary('lossG/2.hole_loss', hole_loss.item())

            valid_loss = self.l1_loss(pred_frames * (1 - masks), frames * (1 - masks)) / torch.mean(1 - masks)
            valid_loss *= self.loss_args['lambda_l1']
            lossG += valid_loss
            self.add_scalar_summary('lossG/3.valid_loss', valid_loss.item())

            if self.audio_flag:
                B, T, _, Hs, Ws = specgrams.shape
                specgrams = specgrams.view(B * T, -1, Hs, Ws)

            # Generator AV-att loss
            if self.loss_args['lambda_av_att'] > 0:
                comp_frames = ((comp_frames * 0.5 + 0.5) - self.imagenet_mean) / self.imagenet_std
                gt_frames = ((frames * 0.5 + 0.5) - self.imagenet_mean) / self.imagenet_std
                av_att_pred = self.av_net(comp_frames, specgrams)['av_attmap']
                av_att_gt = self.av_net(gt_frames, specgrams)['av_attmap']
                av_att_loss = self.mse_loss(av_att_pred, av_att_gt)
                av_att_loss *= self.loss_args['lambda_av_att']
                lossG += av_att_loss
                self.add_scalar_summary('lossG/3.av_att_loss', av_att_loss.item())
                self.add_att_summary('att/1.output_att', av_att_pred.cpu().detach(), comp_frames.cpu().detach())
                self.add_att_summary('att/2.GT_att', av_att_gt.cpu().detach(), gt_frames.cpu().detach())

            # Generator AV-cls loss
            if self.loss_args['lambda_av_cls'] > 0:
                return

            self.optimG.zero_grad()
            lossG.backward()
            self.optimG.step()
            self.add_scalar_summary('lossG/1.lossG', lossG.item())
            
            # Console logs
            pbar.update(1)
            desc = (
                f'lossD: {lossD.item():.3f}; lossG: {lossG.item():.3f}; '
                f'hole: {hole_loss.item():.3f}; valid: {valid_loss.item():.3f}; '
                f'adv: {adv_loss.item():.3f}; '
            )
            if self.loss_args['lambda_av_att'] > 0:
                desc += f'att: {av_att_loss.item():.3f}'
            if self.loss_args['lambda_av_cls'] > 0:
                return
            pbar.set_description(desc)

            # Save checkpoint
            if self.iteration % self.train_args['iter_checkpoint'] == 0:
                self.save(self.iteration)
            if self.iteration >= self.train_args['iter_total']:
                return
