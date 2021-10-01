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


class AVEInpainting(Dataset):
    def __init__(self, split: str, dataset_args: dict, mask_type: str, get_audio=False):
        self.split = split
        assert self.split in ['train', 'val', 'test']
        self.video_dir = dataset_args['video_dir']
        self.image_dir = os.path.join(self.video_dir, split, 'png')
        self.mask_dir = dataset_args['mask_dir']
        self.audio_dir = os.path.join(self.video_dir, split, 'wav')
        self.n_refs = dataset_args['n_refs']
        self.fps = dataset_args['fps']
        self.sr = dataset_args['audio_sr']
        self.mask_type = mask_type
        self.get_audio = get_audio

        self.video_num_frames = dict()
        for video_id in os.listdir(self.image_dir):
            self.video_num_frames[video_id] = len(os.listdir(os.path.join(self.image_dir, video_id)))
        self.video_id_list = sorted(list(self.video_num_frames.keys()))

        self.hflipper = transforms.RandomHorizontalFlip(1.)
        self.image_transforms = transforms.Compose([
            GroupRandomCrop((224, 224)),
            GroupRandomHorizontalFlip(0.5 if mask_type == 'I' else 0.),
            Stack(),
            ToTorchTensor(),
            Normalize()
        ])
        self.specgram_transforms = transforms.Compose([
            Normalize(mean=-40., std=40.)
        ])
        self.mask_transforms = transforms.Compose([
            GroupRandomHorizontalFlip(0.5 if mask_type == 'I' else 0.),
            Stack(),
            ToTorchTensor()
        ])

    def __len__(self):
        return len(self.video_id_list)

    def __getitem__(self, index):  # (B, T, C, H, W)
        video_id = self.video_id_list[index]
        num_frames = self.video_num_frames[video_id]
        frame_index = self.get_frame_index(num_frames, self.n_refs)  # sample temporal index
        
        frames = list()
        masks = list()  # hole: 1 (white), visible: 0 (black)
        specgrams = list()

        # Pick random mask
        if self.split == 'train':
            if self.mask_type == 'I':
                mask_path = os.path.join(self.mask_dir, f'{str(random.randrange(0, 6000)).zfill(5)}.png')
                mask = Image.open(mask_path).resize((224, 224)).convert('L')
                masks = [mask] * len(frame_index)
            elif self.mask_type == 'S':
                mask_dir = os.path.join(self.mask_dir, self.split, video_id)
                for i in frame_index:
                    mask_path = os.path.join(mask_dir, f'{str(i + 1).zfill(3)}.png')
                    mask = Image.open(mask_path).resize((224, 224)).convert('L')
                    masks.append(mask)

        # Frame and masks
        if self.split == 'train':
            for i in frame_index:
                image_path = os.path.join(self.image_dir, video_id, f'{str(i + 1).zfill(3)}.png')
                image = Image.open(image_path).resize((224, 224))
                frames.append(image)

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
                    center = np.clip(center, half_sec, int(num_frames / self.fps) * self.sr - half_sec)
                    audio = audio_full[center - half_sec : center + half_sec]
                    specgram = librosa.feature.melspectrogram(
                        audio, sr=self.sr, n_fft=256, win_length=win_length, hop_length=hop_length, n_mels=80
                    )
                    specgram = librosa.power_to_db(specgram, ref=np.max)
                    specgrams.append(specgram)
                specgrams = torch.from_numpy(np.array(specgrams)).unsqueeze(1)  # (T, 1, H, W)
        
        frame_tensors = self.image_transforms(frames)  # (T, C, H, W)
        mask_tensors = self.mask_transforms(masks)  # (T, C, H, W)
        if self.get_audio:
            specgram_tensors = self.specgram_transforms(specgrams)  # (T, 1, H, W)
            return frame_tensors, mask_tensors, specgram_tensors
        else:
            return frame_tensors, mask_tensors

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
        mask_path = os.path.join(self.mask_dir, f'{str(4706).zfill(5)}.png')
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


class MUSICSoloInpainting(Dataset):
    def __init__(self, split: str, dataset_args: dict, mask_type: str, get_audio=False):
        self.split = split
        assert self.split in ['train', 'val', 'test']
        self.video_dir = dataset_args['video_dir']
        self.image_dir = os.path.join(self.video_dir, split, 'png')
        self.mask_dir = dataset_args['mask_dir']
        self.audio_dir = os.path.join(self.video_dir, split, 'wav')
        self.n_refs = dataset_args['n_refs']
        self.fps = dataset_args['fps']
        self.sr = dataset_args['audio_sr']
        self.mask_type = mask_type
        self.get_audio = get_audio

        self.video_num_frames = dict()
        for video_id in os.listdir(self.image_dir):
            self.video_num_frames[video_id] = len(os.listdir(os.path.join(self.image_dir, video_id)))
        self.video_id_list = sorted(list(self.video_num_frames.keys()))

        self.hflipper = transforms.RandomHorizontalFlip(1.)
        self.image_transforms = transforms.Compose([
            GroupRandomCrop((224, 224)),
            GroupRandomHorizontalFlip(0.5 if mask_type == 'I' else 0.),
            Stack(),
            ToTorchTensor(),
            Normalize()
        ])
        self.specgram_transforms = transforms.Compose([
            Normalize(mean=-40., std=40.)
        ])
        self.mask_transforms = transforms.Compose([
            GroupRandomHorizontalFlip(0.5 if mask_type == 'I' else 0.),
            Stack(),
            ToTorchTensor()
        ])

    def __len__(self):
        return len(self.video_id_list)

    def __getitem__(self, index):  # (B, T, C, H, W)
        video_id = self.video_id_list[index]
        num_frames = self.video_num_frames[video_id]
        frame_index = self.get_frame_index(num_frames, self.n_refs)  # sample temporal index
        
        frames = list()
        masks = list()  # hole: 1 (white), visible: 0 (black)
        specgrams = list()

        # Pick random mask
        if self.split == 'train':
            if self.mask_type == 'I':
                mask_path = os.path.join(self.mask_dir, f'{str(random.randrange(0, 6000)).zfill(5)}.png')
                mask = Image.open(mask_path).resize((224, 224)).convert('L')
                masks = [mask] * len(frame_index)
            elif self.mask_type == 'S':
                mask_dir = os.path.join(self.mask_dir, self.split, video_id)
                for i in frame_index:
                    mask_path = os.path.join(mask_dir, f'{str(i + 1).zfill(5)}.png')
                    mask = Image.open(mask_path).resize((224, 224)).convert('L')
                    masks.append(mask)

        # Frame and masks
        if self.split == 'train':
            for i in frame_index:
                image_path = os.path.join(self.image_dir, video_id, f'{str(i + 1).zfill(5)}.png')
                image = Image.open(image_path).resize((224, 224))
                frames.append(image)

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
                    center = np.clip(center, half_sec, int(num_frames / self.fps) * self.sr - half_sec)
                    audio = audio_full[center - half_sec : center + half_sec]
                    specgram = librosa.feature.melspectrogram(
                        audio, sr=self.sr, n_fft=256, win_length=win_length, hop_length=hop_length, n_mels=80
                    )
                    specgram = librosa.power_to_db(specgram, ref=np.max)
                    specgrams.append(specgram)
                specgrams = torch.from_numpy(np.array(specgrams)).unsqueeze(1)  # (T, 1, H, W)
        
        frame_tensors = self.image_transforms(frames)  # (T, C, H, W)
        mask_tensors = self.mask_transforms(masks)  # (T, C, H, W)
        if self.get_audio:
            specgram_tensors = self.specgram_transforms(specgrams)  # (T, 1, H, W)
            return frame_tensors, mask_tensors, specgram_tensors
        else:
            return frame_tensors, mask_tensors

    def get_frame_index(self, video_length, n_refs):
        if random.uniform(0, 1) > 0.5:
            frame_index = random.sample(range(video_length), n_refs)
            frame_index.sort()
        else:
            pivot = random.randint(0, video_length - n_refs)
            frame_index = [pivot + i for i in range(n_refs)]
        return frame_index