import random
import numbers

from PIL import Image
import numpy as np
import torch


# ================================================================================
# Image and spectrogram transform utils
# ================================================================================


class GroupRandomCrop:
    """Randomly crop the given PIL.Image.
    Input is the list of PIL.Images
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group: list):
        w, h = img_group[0].size
        th, tw = self.size
        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class GroupRandomHorizontalFlip:
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5.
    Input is the list of PIL.Images
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_group: list):
        v = random.random()
        if v < self.p:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            return ret
        else:
            return img_group


class Stack:
    """ Converts a list of RGB or grayscale PIL.Image
    into numpy.ndarray (T x H x W x C) in the range of [0, 255].
    """
    def __init__(self):
        pass
    
    def __call__(self, images: list) -> list:
        image_type = images[0].mode
        if image_type == "RGB":
            return np.stack(images, axis=0)
        elif image_type == "L":
            images = [np.expand_dims(image, axis=2) for image in images]
            return np.stack(images, axis=0)
        else:
            raise NotImplementedError(f"Unknown type of image: {image_type}")


class ToTorchTensor:
    """ Converts numpy.ndarray images (T x H x W x C) from the range of [0, 255]
    into torch.FloatTensor images of shape (T x C x H x W) in the range of [0.0, 1.0].
    """
    def __init__(self):
        pass

    def __call__(self, images: np.ndarray) -> torch.FloatTensor:  # (T, H, W, C)
        images = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)
        images = images.float().div(255)
        return images


class Normalize:
    """ Normalize torch.FloatTensor images in the range of [0.0, 1.0]
    with given mean and standard deviation.
    """
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std
    
    def __call__(self, images: torch.FloatTensor) -> torch.FloatTensor:
        return (images - self.mean) / self.std


# class MinMaxNormalize:
#     """ Min-max normalize torch.FloatTensor Mel-spectrograms in the range of [min, 0.0]
#     into the range of [0.0, 1.0]
#     """
#     def __init__(self):
#         pass

#     def __call__(self, specgrams: torch.FloatTensor) -> torch.FloatTensor:
#         if specgrams.min() == 0:
#             return specgrams
#         return torch.clamp((specgrams - specgrams.min()) / -specgrams.min(), 0, 1)


class ToImage:
    """ Converts normalized torch.FloatTensor images (N x C x H x W) in the range of [-1.0, 1.0]
    into numpy.ndarray images (N x H x W x C) in the range of [0, 255].
    """
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std
    
    def __call__(self, images) -> np.ndarray:
        images = images.cpu() if images.is_cuda else images
        images = images * self.std + self.mean
        images = images.permute(0, 2, 3, 1)
        images *= 255
        return images.numpy().astype(np.uint8)

