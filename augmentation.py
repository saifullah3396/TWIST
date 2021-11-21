# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
import ocrodeg
import numpy as np
import torch
import torchvision
from das.data.transforms.dict_transform import DictTransform
from das.data.transforms.grayscale_to_rgb import GrayScaleToRGB
from PIL import Image, ImageFilter, ImageOps
from timm.data import create_transform
from torchvision import transforms
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms.transforms import ToPILImage
import torchvision.transforms.functional as F

class RandomResizedCropCustom(RandomResizedCrop):
    @staticmethod
    def get_params(img, scale, ratio, region_mask):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        width, height = F._get_image_size(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                # mask = region_mask[h//2:height-h//2, w//2:width-w//2]
                pixel_list = region_mask.nonzero()
                if len(pixel_list) == 0:
                    i = torch.randint(0, height - h + 1, size=(1,)).item()
                    j = torch.randint(0, width - w + 1, size=(1,)).item()
                else:
                    p_idx = torch.randint(0, len(pixel_list), size=(1, )).item()
                    i = pixel_list[p_idx][0] - h // 2
                    j = pixel_list[p_idx][1] - w // 2
                i = int(torch.clip(i, min=0, max=height - h - 1))
                j = int(torch.clip(j, min=0, max=width - w - 1))
                    
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img, pixel_list):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio, pixel_list)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)



class RandomResizedMaskedCrop(object):
    def __init__(self, img_size):
        self.t = RandomResizedCropCustom((img_size,img_size), scale=(0.05, 0.15))

    def get_black_and_white_regions_mask(self, image_tensor):
        black_and_white_threshold = 0.5
        c, h, w = image_tensor.shape
        ky = 8
        kx = 8
        black_and_white_regions_fast = (image_tensor[0].unfold(
            0, ky, kx).unfold(1, ky, kx) < black_and_white_threshold).any(dim=2).any(dim=2)
        black_and_white_regions_fast = black_and_white_regions_fast.repeat_interleave(
            ky, dim=0).repeat_interleave(kx, dim=1)
        black_and_white_regions_fast = torchvision.transforms.functional.resize(
            black_and_white_regions_fast.unsqueeze(0), [h, w]).squeeze()
        return ((black_and_white_regions_fast).float())
        
    def __call__(self, img):
        return self.t(img, self.get_black_and_white_regions_mask(img))

class Blotches(object):
    def __call__(self, img):
        if len(img.shape) != 2:
            img = img.squeeze()
        return torch.from_numpy(ocrodeg.random_blotches(np.array(img), 3e-4, 1e-4)).unsqueeze(0)

class BinaryBlur(object):
    def __call__(self, img):
        if len(img.shape) != 2:
            img = img.squeeze()
        return torch.from_numpy(ocrodeg.binary_blur(np.array(img), 0.5)).unsqueeze(0)


class DocumentAugmentations2(object):
    def __init__(self, args):
        self.aug1 = transforms.Compose([
            RandomResizedMaskedCrop(args.img_size),
            transforms.RandomApply([BinaryBlur()], p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            GrayScaleToRGB(),
            transforms.ToPILImage(),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, .5])], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.aug2 = transforms.Compose([
            RandomResizedMaskedCrop(args.img_size),            
            transforms.RandomApply([Blotches()], p=0.2),
            transforms.RandomApply([BinaryBlur()], p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            GrayScaleToRGB(),
            transforms.ToPILImage(),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, .5])], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.aug1(image))
        crops.append(self.aug2(image))
        return crops

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

def get_augmentations(args):
    if args.aug == 'moco':
        return MocoAugmentations(args)
    if args.aug == 'barlow':
        return \
            transforms.Compose([
                DictTransform(['image'], BarlowtwinsAugmentations(args))])
    if args.aug == 'docs':
        return \
            transforms.Compose([
                DictTransform(['image'], DocumentAugmentations2(args))])
    if args.aug == 'multicrop':
        return MultiCropAugmentation(args)
    if args.aug == 'multicropeval':
        return MultiCropEvalAugmentation(args)
    if args.aug == 'rand':
        return RandAugmentation(args)

class RandAugmentation(object):
    def __init__(self, args):
        self.aug = create_transform(
                input_size=224,
                is_training=True,
                color_jitter=0.4,
                auto_augment='rand-m9-mstd0.5-inc1',
                interpolation='bicubic',
                re_prob=0.25,
                re_mode='pixel',
                re_count=1,
            )

    def __call__(self, image):
        crops = []
        crops.append(self.aug(image))
        crops.append(self.aug(image))
        return crops

class DocumentAugmentations(object):
    def __init__(self, args):
        self.aug1 = transforms.Compose([
            GrayScaleToRGB(),
            transforms.ToPILImage(),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomApply(
                [   
                    transforms.RandomAffine((-5, 5))], 
                p=0.5
            ),
            transforms.RandomApply(
                [   
                    transforms.RandomAffine(0, translate=(0.2, 0.2))], 
                p=0.5
            ),
            transforms.RandomApply(
                [   
                    transforms.RandomAffine(0, scale=(0.8, 1.0))], 
                p=0.5
            ),
            transforms.RandomApply(
                [   
                    transforms.RandomAffine(0, shear=(-5, 5))], 
                p=0.5
            ),

            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, .5])], p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.aug2 = transforms.Compose([
            GrayScaleToRGB(),
            transforms.ToPILImage(),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomApply(
                [   
                    transforms.RandomAffine((-5, 5))], 
                p=0.5
            ),
            transforms.RandomApply(
                [   
                    transforms.RandomAffine(0, translate=(0.2, 0.2))], 
                p=0.5
            ),
            transforms.RandomApply(
                [   
                    transforms.RandomAffine(0, scale=(0.8, 1.0))], 
                p=0.5
            ),
            transforms.RandomApply(
                [   
                    transforms.RandomAffine(0, shear=(-5, 5))], 
                p=0.5
            ),

            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, .5])], p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.aug1(image))
        crops.append(self.aug2(image))
        return crops
        
class MocoAugmentations(object):
    def __init__(self, args):
        self.aug = transforms.Compose([
            DictTransform(['image'], 
                transforms.RandomResizedCrop(args.img_size, scale=(0.2, 1.), interpolation=Image.BICUBIC)),
            DictTransform(['image'], 
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)),
            DictTransform(['image'], 
                transforms.RandomGrayscale(p=0.2)),
            DictTransform(['image'], 
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5)),
            DictTransform(['image'], 
                transforms.RandomHorizontalFlip()),
            DictTransform(['image'], 
                transforms.ToTensor()),
            DictTransform(['image'], 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])),]) 

    def __call__(self, image):
        crops = []
        crops.append(self.aug(image))
        crops.append(self.aug(image))
        return crops



# class RandomResizedCropThreshold(object):
#     def __init__(self, img_size):
#         self.img_size = img_size
#         self.t = transforms.Resize((img_size, img_size))
#         self.grid_xy = 4

#     def __call__(self, img):
#         c, h, w = img.shape
#         size_y = h // self.grid_xy # patch size
#         size_x = w // self.grid_xy  # patch stride
#         patches = img.unfold(1, size_y, size_y).unfold(2, size_x, size_x)
#         patches = patches.reshape(c, self.grid_xy * self.grid_xy, size_y, size_x).permute(1, 0, 2, 3)
#         valid_patches = []
#         for i in range(patches.shape[0]):
#             if len(patches[i][patches[i] < 0.5]) > 0:
#                 valid_patches.append(self.t(patches[i].squeeze()).float())
#             if len(valid_patches) == 2:
#                 break
#             # else:
#                 # valid_patches.append(torch.zeros((c, self.img_size, self.img_size)))
#         return torch.stack(valid_patches)

# class DocumentAugmentations2(object):
#     def __init__(self, args):
#         self.aug = transforms.Compose([
#             GrayScaleToRGB(),
#             transforms.RandomApply([
#                 transforms.RandomRotation((90, 90), expand=True)], p=0.5),
#             transforms.ConvertImageDtype(torch.float),
#             RandomResizedCropThreshold(args.img_size),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#     def __call__(self, image):
#         return self.aug(image)

class BarlowtwinsAugmentations(object):
    def __init__(self, args):
        self.aug1 = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.aug2 = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.aug1(image))
        crops.append(self.aug2(image))
        return crops

class MultiCropAugmentation(object):
    def __init__(self, args):
        global_crops_scale = args.global_crops_scale
        local_crops_scale  = args.local_crops_scale
        local_crops_number = args.local_crops_number

        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            #utils.GaussianBlur(1.0),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            #utils.GaussianBlur(0.1),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            #utils.GaussianBlur(p=0.5),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

