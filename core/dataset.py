import os
import random

import numpy as np
import librosa
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

from core.utils import *
from core.transforms import *


class MUSICSoloInpainting(Dataset):
    def __init__(self, dataset_args: dict, split='train'):
        self.dataset_args = dataset_args
        self.root_dir = dataset_args['video_dir']
        self.ref_frames = dataset_args['ref_frames']
        self.image_width, self.image_height = dataset_args['image_width'], dataset_args['image_height']
        self.image_shape = (self.image_width, self.image_height)
        self.split = split

        self.video_dict = dict()
        for video_id in os.listdir(f'{self.root_dir}/png'):
            self.video_dict[video_id] = len(os.listdir(f'{self.root_dir}/png/{video_id}'))
        self.video_ids = sorted(list(self.video_dict.keys()))

        self.hflipper = transforms.RandomHorizontalFlip(1.)
        self.image_transforms = transforms.Compose([
            Stack(),
            ToTorchTensor(),
            Normalize()
        ])
        self.mask_transforms = transforms.Compose([
            Stack(),
            ToTorchTensor()
        ])

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):  # (B, T, C, H, W)
        video_id = self.video_ids[index]
        all_frames = [f'{str(i).zfill(5)}.png' for i in range(1, min(240, self.video_dict[video_id] + 1))]
        sampled_idxs = self.get_frame_index(len(all_frames), self.ref_frames)
        frames = list()
        
        if self.split == 'train':
            for i in sampled_idxs:
                image_path = f'{self.root_dir}/png/{video_id}/{all_frames[i]}'
                image = Image.open(image_path)
                assert image.size == self.image_shape
                frames.append(image)
            if random.uniform(0, 1) > 0.5:
                frames = [self.hflipper(frame) for frame in frames]
            masks = create_fixed_rectangular_mask(5, 256, 256, 42)
        else:
            for i in all_frames:
                image_path = f'{self.root_dir}/png/{video_id}/{i}'
                image = Image.open(image_path)
                if image.size != self.image_shape:
                    image.resize(self.image_shape)
                frames.append(image)
            masks = create_fixed_rectangular_mask(len(all_frames), 256, 256, 42)

        frame_tensors = self.image_transforms(frames)  # (T, C, H, W)
        mask_tensors = self.mask_transforms(masks)
        data_dict = {'frames': frame_tensors, 'masks': mask_tensors, 'video_id': video_id}
        return data_dict

    # Index sampling function
    def get_frame_index(self, length, n_refs):
        if random.uniform(0, 1) > 0.5:
            ref_index = random.sample(range(length), n_refs)
            ref_index.sort()
        else:
            pivot = random.randint(0, length-n_refs)
            ref_index = [pivot+i for i in range(n_refs)]
        return ref_index


class AVEInpainting(Dataset):
    def __init__(self, split: str, dataset_args: dict, get_audio=False):
        self.split = split
        assert self.split in ['train', 'val', 'test']
        self.video_dir = dataset_args['video_dir']
        self.image_dir = os.path.join(self.video_dir, split, 'png')
        self.mask_dir = dataset_args['mask_dir']
        self.audio_dir = os.path.join(self.video_dir, split, 'wav')
        self.n_refs = dataset_args['n_refs']
        self.fps = dataset_args['fps']
        self.sr = dataset_args['audio_sr']
        self.get_audio = get_audio

        self.video_num_frames = dict()
        for video_id in os.listdir(self.image_dir):
            self.video_num_frames[video_id] = len(os.listdir(os.path.join(self.image_dir, video_id)))
        self.video_id_list = sorted(list(self.video_num_frames.keys()))

        self.hflipper = transforms.RandomHorizontalFlip(1.)
        self.image_transforms = transforms.Compose([
            GroupRandomCrop((224, 224)),
            GroupRandomHorizontalFlip(),
            Stack(),
            ToTorchTensor(),
            Normalize()
        ])
        self.specgram_transforms = transforms.Compose([
            Normalize(mean=-40., std=40.)
        ])
        self.mask_transforms = transforms.Compose([
            Stack(),
            ToTorchTensor()
        ])

    def __len__(self):
        return len(self.video_id_list)

    def __getitem__(self, index):  # (B, T, C, H, W)
        video_id = self.video_id_list[index]
        all_frames = [f'{str(i).zfill(3)}.png' for i in range(1, self.video_num_frames[video_id] + 1)]

        # Pick random mask
        if self.split == 'train':
            mask_path = os.path.join(self.mask_dir, f'{str(random.randrange(0, 6000)).zfill(5)}.png')
            mask = Image.open(mask_path).resize((224, 224)).convert('L')
            if random.uniform(0, 1) > 0.5:
                mask = self.hflipper(mask)
            all_masks = [mask] * len(all_frames)
        else:
            return
            # mask_path = f'{self.mask_dir}/{str(4007).zfill(5)}.png'
            # mask = Image.open(mask_path).resize((self.image_height, self.image_width)).convert('L')
            # all_masks = [mask] * len(all_frames)

        frames = list()
        masks = list()  # hole: 1 (white), visible: 0 (black)
        specgrams = list()
        frame_index = self.get_frame_index(len(all_frames), self.n_refs)  # sample temporal index
        
        # Frame and masks
        if self.split == 'train':
            for i in frame_index:
                image_path = os.path.join(self.image_dir, video_id, all_frames[i])
                image = Image.open(image_path).resize((224, 224))
                frames.append(image)
                masks.append(all_masks[i])
            if random.uniform(0, 1) > 0.5:
                frames = [self.hflipper(frame) for frame in frames]
        else:
            for i in all_frames:
                image_path = f'{self.video_dir}/{self.split}/image/{video_id}/{i}'
                image = Image.open(image_path)
                if image.size != self.image_shape:
                    image.resize(self.image_shape)
                frames.append(image)
            masks = all_masks

        # Mel-spectrograms
        if self.get_audio:
            audio_path = os.path.join(self.audio_dir, f'{video_id}.wav')
            audio_full, sr = librosa.load(audio_path, sr=self.sr)
            half_sec = self.sr // 2
            win_length, hop_length = int(self.sr * 0.01), int(self.sr * 0.005)
            # max_length = int(len(all_frames) / self.fps) * self.sr
            # if len(audio_full) < max_length:  # audio padding
            #     audio_full = np.concatenate((audio_full, audio_full[len(audio_full) - max_length :]))
            if self.split == 'train':                
                for i in frame_index:
                    center = int((i + 0.5) / self.fps * self.sr)
                    center = np.clip(center, half_sec, int(len(all_frames) / self.fps) * self.sr - half_sec)
                    audio = audio_full[center - half_sec : center + half_sec]
                    specgram = librosa.feature.melspectrogram(
                        audio, sr=self.sr, n_fft=256, win_length=win_length, hop_length=hop_length, n_mels=80
                    )
                    specgram = librosa.power_to_db(specgram, ref=np.max)
                    specgrams.append(specgram)
                specgrams = torch.from_numpy(np.array(specgrams)).unsqueeze(1)  # (T, 1, H, W)
            else:
                return
                # for i in range(len(all_frames)):
                #     center = int((i + 0.5) / self.fps * self.sr)
                #     half_interval = self.sr // 2
                #     center = np.clip(center, half_interval, max_length - half_interval)
                #     audio = audio_full[center - half_interval : center + half_interval]
                #     specgram = librosa.feature.melspectrogram(audio,
                #         sr=self.sr, n_fft=512, win_length=480, hop_length=240, n_mels=80)
                #     specgram = librosa.power_to_db(specgram, ref=np.max)
                #     specgrams.append(specgram)
                # specgrams = torch.from_numpy(np.array(specgrams)).unsqueeze(1)  # (B, T, 1, H, W)
        
        frame_tensors = self.image_transforms(frames)  # (T, C, H, W)
        mask_tensors = self.mask_transforms(masks)  # (T, C, H, W)
        specgram_tensors = self.specgram_transforms(specgrams) if len(specgrams) > 0 else list()  # (T, 1, H, W)
        # data_dict = {'frames': frame_tensors, 'masks': mask_tensors, 'specgrams': specgram_tensors}
        return frame_tensors, mask_tensors, specgram_tensors

    def get_frame_index(self, video_length, n_refs):
        if random.uniform(0, 1) > 0.5:
            frame_index = random.sample(range(video_length), n_refs)
            frame_index.sort()
        else:
            pivot = random.randint(0, video_length - n_refs)
            frame_index = [pivot + i for i in range(n_refs)]
        return frame_index


class AVEInpaintingTest(Dataset):
    def __init__(self, dataset_args: dict):
        self.dataset_args = dataset_args
        self.video_dir = dataset_args['video_dir']
        self.image_dir = os.path.join(self.video_dir, 'test', 'png')
        self.mask_dir = dataset_args['mask_dir']
        self.audio_dir = os.path.join(self.video_dir, 'test', 'wav')
        self.fps = dataset_args['fps']
        self.audio_sr = dataset_args['audio_sr']

        self.video_num_frames = dict()
        for video_id in os.listdir(self.image_dir):
            self.video_num_frames[video_id] = len(os.listdir(os.path.join(self.image_dir, video_id)))
        self.video_id_list = sorted(list(self.video_num_frames.keys()))

        self.image_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])
        self.mask_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.audio_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(-40.0,), std=(40.0,))
        ])

    def __len__(self):
        return len(self.video_id_list)
    
    def __getitem__(self, index: int):
        video_id = self.video_id_list[index]
        num_frames = self.video_num_frames[video_id]

        images = list()
        masks = list()
        specgrams = list()
        
        # All frames
        for i in range(1, num_frames + 1):
            image_path = os.path.join(self.image_dir, video_id, f'{str(i).zfill(3)}.png')
            image = Image.open(image_path)
            image = self.image_transforms(image).unsqueeze(0)
            images.append(image)
        images = torch.cat(images, dim=0)

        # All masks
        mask_path = os.path.join(self.mask_dir, f'{str(4007).zfill(5)}.png')
        mask = Image.open(mask_path)
        mask = self.mask_transforms(mask).unsqueeze(0)
        masks = [mask] * num_frames
        masks = torch.cat(masks, dim=0)

        return images, masks, video_id
        
        # # All spectrograms
        # audio_path = os.path.join(self.audio_dir, f'{video_id}.wav')
        # audio_full, sr = librosa.load(audio_path, sr=self.audio_sr)
        # half_sec = self.audio_sr // 2
        # win_length, hop_length = int(self.audio_sr * 0.01), int(self.audio_sr * 0.005)
        # for i in range(1, num_frames + 1):
        #     center = int((i + 0.5) / self.fps * self.audio_sr)
        #     center = np.clip(center, half_sec, int(num_frames / self.fps) * self.audio_sr - half_sec)
        #     pos_audio = audio_full[center - half_sec : center + half_sec]
        #     specgram = librosa.feature.melspectrogram(
        #         pos_audio, sr=self.audio_sr, n_fft=256, win_length=win_length, hop_length=hop_length, n_mels=80
        #     )
        #     specgram = librosa.power_to_db(specgram, ref=np.max)
        #     specgram = self.audio_transforms(specgram).unsqueeze(0)
        #     specgrams.append(specgram)
        # specgrams = torch.cat(specgrams, dim=0)

        # return images, specgrams, video_id