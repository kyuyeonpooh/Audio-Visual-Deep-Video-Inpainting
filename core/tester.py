import os
import csv

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models import sttn
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.dataset import *
from core.utils import *


class Tester:
    def __init__(self, config: dict, ckpt_path: str, output_dir: str, name: str, device='0'):
        self.config = config
        self.dataset_args = config["dataset"]

        self.name = name
        self.output_dir = os.path.join(output_dir, name)
        print('Output dir', self.output_dir)
        os.makedirs(os.path.join(self.output_dir, 'GT'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'Corrupted'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'Inpainted'), exist_ok=True)
        self.csvfile = open(os.path.join(self.output_dir, 'result.csv'), 'w')
        self.writer = csv.writer(self.csvfile)

        self.test_dataset = AVEInpaintingTest(self.dataset_args)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=4,
            shuffle=False
        )
        self.ref_frames = 4
        self.ref_stride = 4
        assert self.ref_frames % 2 == 0

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        self.device = torch.device("cuda")
        self.model = sttn.STTN(batch_size=1)
        self.model.to(self.device)
        self.load_model(ckpt_path)
        self.model.eval()
        
        # get pretrained I3D network
        self.i3d_model = get_pretrained_i3d()
        self.i3d_model.to(self.device)
        self.i3d_model.eval()

    def load_model(self, ckpt_path: str):
        print(f"Loading {ckpt_path} ...")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Model checkpoint {ckpt_path} does not exist.")
        load_data = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(load_data["netG"])

    def psnr(self, gt_frame, comp_frame):
        assert gt_frame.dtype == np.uint8 and comp_frame.dtype == np.uint8
        mse = np.mean((gt_frame.astype(np.float32) - comp_frame.astype(np.float32)) ** 2)
        if mse == 0:
            return np.inf
        return 10 * np.log10((255 ** 2) / mse)

    def compute_video_psnr(self, gt_frames: np.ndarray, comp_frames: np.ndarray):  # (N, H, W, C)
        psnr_images = list(map(lambda gt, comp: self.psnr(gt, comp), gt_frames, comp_frames))
        psnr_images = np.array(psnr_images)
        psnr_images = psnr_images[psnr_images != np.inf]  # exclude inf values (extremely easy cases)
        return psnr_images.mean()

    def ssim(self, gt_frame, comp_frame):
        to_grayscale = lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        return structural_similarity(to_grayscale(gt_frame), to_grayscale(comp_frame))

    def compute_video_ssim(self, gt_frames: np.ndarray, comp_frames: np.ndarray):  # (N, H, W, C)
        ssim_images = list(map(lambda gt, comp: self.ssim(gt, comp), gt_frames, comp_frames))
        ssim_images = np.array(ssim_images)
        return ssim_images.mean()

    def get_i3d_activation(self, frames: np.ndarray):  # (T, H, W, C)
        frames = Normalize()(ToTorchTensor()(frames))  # (T, C, H, W)
        frames = frames.to(self.device)
        frames = frames.unsqueeze(0)  # (1, T, C, H, W)
        frames = frames.transpose(1, 2)  # (1, C, T, H, W)
        with torch.no_grad():
            i3d_feat = self.i3d_model.extract_features(frames, target_endpoint="Logits")
            i3d_feat = i3d_feat.flatten()
        return i3d_feat.detach().cpu().numpy()

    def compute_video_fid_score(self, gt_i3d_acts: np.ndarray, comp_i3d_acts: np.ndarray):
        """ Given two distribution of features, compute the FID score between them
        """
        m1 = np.mean(gt_i3d_acts, axis=0)
        m2 = np.mean(comp_i3d_acts, axis=0)
        s1 = np.cov(gt_i3d_acts, rowvar=False)
        s2 = np.cov(comp_i3d_acts, rowvar=False)
        return calculate_frechet_distance(m1, s1, m2, s2)

    def save_frames(self, vid_id, gt_frames, masked_frames, comp_frames):
        for i, (x, y, z) in enumerate(zip(gt_frames, masked_frames, comp_frames)):
            im1 = Image.fromarray(x)
            im2 = Image.fromarray(y)
            im3 = Image.fromarray(z)
            gt_dir = os.path.join(self.output_dir, "GT", vid_id)
            corr_dir = os.path.join(self.output_dir, 'Corrupted', vid_id)
            fake_dir = os.path.join(self.output_dir, "Inpainted", vid_id)
            os.makedirs(gt_dir, exist_ok=True)
            os.makedirs(corr_dir, exist_ok=True)
            os.makedirs(fake_dir, exist_ok=True)
            im1.save(os.path.join(self.output_dir, 'GT', vid_id, f"{str(i+1).zfill(5)}.jpg"))
            im2.save(os.path.join(self.output_dir, 'Corrupted', vid_id, f"{str(i+1).zfill(5)}.jpg"))
            im3.save(os.path.join(self.output_dir, 'Inpainted', vid_id, f"{str(i+1).zfill(5)}.jpg"))

    def test_checkpoint(self):
        vid_psnr_list = []  # PSNR
        vid_ssim_list = []  # SSIM
        gt_i3d_acts = []  # I3D features of ground truth videos
        comp_i3d_acts = []  # I3D features of predicted (completed) videos

        with torch.no_grad():
            for _, data in enumerate(tqdm(self.test_loader, ncols=80)):  # for each video
                frames, masks, video_id = data
                frames, masks = frames.to(self.device), masks.to(self.device)
                gt_frames = frames[0].cpu().clone().detach()
                gt_frames = ToImage()(gt_frames)  # ground truth frames
                gt_masks = masks[0].cpu().clone().detach()
                gt_masks = gt_masks.permute(0, 2, 3, 1).numpy().astype(np.uint8)
                vid_length = len(gt_frames)
                comp_frames = [None] * vid_length  # completed frames (pixels in hole region are predicted)
                masked_frames = frames * (1 - masks).float()
                feats = self.model.encoder(masked_frames.squeeze())
                feats = feats.unsqueeze(0)
                neighbor_stride = self.ref_frames // 2
                for f in range(0, vid_length, neighbor_stride):  # for each frame
                    neighbors = list(range(max(0, f - neighbor_stride), min(vid_length, f + neighbor_stride)))
                    refs = list(range(0, vid_length, self.ref_stride))
                    feats_index = neighbors + refs
                    pred_feats = self.model.transformers({
                        'frame_feats': feats[0, feats_index, :, :, :],
                        'masks': F.interpolate(masks[0, feats_index, :, :, :], scale_factor=1/4, recompute_scale_factor=True)
                    })['frame_feats']
                    pred_frames = self.model.decoder(pred_feats[:len(neighbors), :, :, :])
                    pred_frames = pred_frames.detach().cpu()
                    pred_frames = ToImage()(pred_frames)
                    for i in range(len(neighbors)):  # add each completed frame to the list
                        index = neighbors[i]
                        comp_frame = pred_frames[i] * gt_masks[index] + gt_frames[index] * (1 - gt_masks[index])
                        if comp_frames[index] is None:
                            comp_frames[index] = comp_frame.astype(np.uint8)
                        else:
                            comp_frames[index] = (comp_frames[index].astype(np.float32) + comp_frame.astype(np.float32)) / 2.
                            comp_frames[index] = comp_frames[index].astype(np.uint8)
                comp_frames = np.array(comp_frames).astype(np.uint8)
            
                # evaluation
                assert gt_frames.shape == comp_frames.shape  # (T, H, W, C)
                vid_psnr_list.append(self.compute_video_psnr(gt_frames, comp_frames))
                vid_ssim_list.append(self.compute_video_ssim(gt_frames, comp_frames))
                self.writer.writerow([video_id[0], '{:.2f}'.format(vid_psnr_list[-1]), '{:.4f}'.format(vid_ssim_list[-1])])
                self.csvfile.flush()
                masked_frames = ToImage()(masked_frames[0])
                self.save_frames(video_id[0], gt_frames, masked_frames, comp_frames)
                gt_i3d_acts.append(self.get_i3d_activation(gt_frames))
                comp_i3d_acts.append(self.get_i3d_activation(comp_frames))
                # print(vid_psnr_list,vid_ssim_list)
                
        perfs = {}
        perfs["psnr"] = np.array(vid_psnr_list).mean()
        perfs["ssim"] = np.array(vid_ssim_list).mean()
        perfs["vfid"] = self.compute_video_fid_score(np.array(gt_i3d_acts), np.array(comp_i3d_acts))
        print(f'{self.name} - PSNR: {perfs["psnr"]:.4f}, SSIM: {perfs["ssim"]:.4f}, VFID: {perfs["vfid"]:.4f}')

