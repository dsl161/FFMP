
import json
import os
import random
from PIL import Image
from torch.utils.data import Dataset
from datasets.utils import NormalSample, jpg2id
import datasets.transforms as T
import logging

import cv2
import numpy as np
import torch

import torchvision.transforms as transforms


logger = logging.getLogger("global")


class BaseTransform(object):
    """
    Resize image, density, and boxes
    """

    def __init__(
        self, size_rsz, gamma=False, gray=False, hflip=True, vflip=False):
        self.size_rsz = size_rsz  # [h, w]
        self.hflip = hflip
        self.vflip = vflip
        self.gamma = gamma
        self.gray = gray

    def __call__(self, image, boxes, points):

        # gamma
        if self.gamma:
            gamma_range = [0.8, 1.25]
            prob = 0.5
            gamma_fn = T.Gamma(gamma_range, prob)
            image = gamma_fn(image)
        image = Image.fromarray(image)

        # hflip, vflip, gray
        if self.gray:
            prob = 0.5
            transform_fn = T.RandomGrayscale(prob)
            image = transform_fn(image)

        if self.hflip:
            prob = 0.5
            transform_fn = T.RandomHFlip(self.size_rsz, prob)
            image, boxes, points = transform_fn(
                image,  boxes, points
            )
        if self.vflip:
            prob = 0.5
            transform_fn = T.RandomVFlip(self.size_rsz, prob)
            image, boxes, points = transform_fn(
                image,  boxes, points
            )


        return image, boxes, points


class FSC147(Dataset):
    def __init__(
        self,
        root_dir,
        mode,
    ):
        super().__init__()
        self.root_dir = root_dir

        with open(os.path.join(root_dir, 'Train_Test_Val_FSC_147.json')) as f:
            imglist = json.load(f)[mode]
        self.imgids = [jpg2id(imgf) for imgf in imglist]
        with open(os.path.join(root_dir, 'fsc147_384x576.json')) as f:
            samples = json.load(f)
        self.samples = {idx: samples[idx] for idx in self.imgids}
        random.shuffle(self.imgids)

        self.it2cat = dict()
        with open(os.path.join(root_dir, 'ImageClasses_FSC147.txt')) as f:
            catdict = dict()
            for line in f.read().strip().split('\n'):
                a, b = line.split('.jpg')
                a, b = a.strip(), b.strip()
                if b not in catdict:
                    catdict[b] = len(catdict) + 1
                self.it2cat[a] = catdict[b]

        if mode == 'train':
            self.transform_fn = BaseTransform((384, 576), True, True, True, True)
            self.colorjitter_fn = T.RandomColorJitter.from_params(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, prob=0.5)
        else:
            self.transform_fn = None
            self.colorjitter_fn = None

        self.root_path = root_dir
        self.normalfunc = NormalSample()

    def __getitem__(self, index):
        imgid = self.imgids[index]

        sample = self.getSample(imgid)

        return (*sample, imgid)

    def __len__(self):
        return len(self.imgids)

    def getSample(self, imgid):
        sample = self.samples[imgid]
        image_path = os.path.join(self.root_path, sample['imagepath'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        points = sample['points'] # N x (w, h)
        boxes = sample['boxes'][:3]  # 3 x ((left, top), (right, bottom))


        # transform
        if self.transform_fn:
            image, boxes, points = self.transform_fn(
                image, boxes, points,
            )
        if self.colorjitter_fn:
            image = self.colorjitter_fn(image)

        points = torch.tensor(points).round().long()
        boxes = torch.clip(torch.tensor(boxes).view(3, 4).round().long(), min=0)
        dotmap = np.zeros((1, h, w), dtype=np.float32)
        points[:, 1] = torch.clip(points[:, 1], min=0, max=h - 1)
        points[:, 0] = torch.clip(points[:, 0], min=0, max=w - 1)
        dotmap[0, points[:, 1], points[:, 0]] = 1
        image, dotmap = self.normalfunc(image, dotmap)
        for i, box in enumerate(boxes):
            l, t, r, b = box
            b, r = max(t + 1, b), max(l + 1, r)
            boxes[i] = torch.tensor([l, t, r, b])
        return image, boxes, dotmap

    @staticmethod
    def collate_fn(samples):
        images, boxes, dotmaps, imgids = zip(*samples)
        images = torch.stack(images, dim=0)
        index = torch.arange(images.size(0)).view(-1, 1).repeat(1, 3).view(-1, 1)
        boxes = torch.cat([index, torch.cat(boxes, dim=0)], dim=1)
        dotmaps = torch.stack(dotmaps, dim=0)
        return images, boxes, dotmaps, imgids


if __name__ == '__main__':

    dataset = FSC147('/home/mgx/project/Bilinear-Matching-Network-main/dataset', mode='train')

    import tqdm
    for image, boxes, dotmap, id in tqdm.tqdm(dataset):
        # imgid:str
        print(image.shape)  # torch.Size([3, 384, 384])
        print(boxes.shape)  # torch.Size([3, 4])

        print(dotmap.shape)  # torch.Size([1, 384, 384])
        print(id)
        break

