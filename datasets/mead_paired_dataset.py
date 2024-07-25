import os
import math
import random
import torch
import json
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset
from glob import glob
import numpy as np


class MEADPairedDataset(Dataset):
    def __init__(self, opts, mode='train', max_num=15000):
        super.__init__()
        self.opts = opts
        self.data_root = opts.data_root
        self.source_actors = opts.source_actors
        self.driving_actors = opts.driving_actors
        self.emotions = opts.emotions
        self.mode = mode
        self.source_paths = []
        self.source_emotions = []
        self.paired_imgs = []
        self.paired_data_dict = {}
        self.data_dict = {}
        
        self.use_pair_data = True
        self.pair_data_percent = 0.5

        all_actors = set(self.source_actors).union(set(self.driving_actors))
        for actor in all_actors:
            self.paired_data_dict[actor] = {emo: [] for emo in self.emotions}
            self.data_dict[actor] = {emo: [] for emo in self.emotions}
            with open(os.path.join(self.data_root, mode, actor, 'videos/_frame_info.txt')) as f:
                frame_infos = f.read().splitlines()
            
            paired_imgs = []

            if (mode == 'train') and (actor in self.source_actors):
                for emo in self.emotions:
                    imgs = sorted(glob(os.path.join(self.data_root, 'generated', emo, actor, '*.png')))
                    self.paired_data_dict[actor][emo] = imgs

                for i in range(len(self.paired_data_dict[actor]['neutral'])):
                    paired_img = {}
                    for emo in self.emotions:
                        paired_img[emo] = self.paired_data_dict[actor][emo][i]
                    paired_imgs.append(paired_img)

                if len(paired_imgs) > max_num:
                    paired_imgs = random.sample(paired_imgs, max_num)
                
                self.paired_imgs += paired_imgs

            frame_infos = random.sample(
                frame_infos, min(len(frame_infos), max_num))
            for info in frame_infos:
                emo, global_idx = info.split(' ')
                emo = emo.split('_', 1)[0]
                global_idx = int(global_idx)
                if mode == 'train':
                    seq_idx = global_idx // 50
                    img_path = os.path.join(
                        self.data_root, mode, actor, 'images_aligned', f'{seq_idx:06d}', f'{global_idx:06d}.png')
                else:
                    img_path = os.path.join(
                        self.data_root, mode, actor, 'images_aligned', f'{global_idx:06d}.png')

                if not os.path.exists(img_path):
                    continue

                self.data_dict[actor][emo].append(img_path)
                if actor in self.source_actors:
                    self.source_paths.append(img_path)
                    self.source_emotions.append(emo)

        self.transform_crop = T.Compose([
            T.Resize((opts.loadSize, opts.loadSize)),
            T.CenterCrop((opts.cropSize, opts.cropSize)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.transform_no_crop = T.Compose([
            T.Resize((opts.cropSize, opts.cropSize)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.to_tensor = T.ToTensor()

    def __len__(self):
       return len(self.source_paths)*2 if self.mode == 'train' else len(self.source_paths)

    def __getitem__(self, index):
        if self.mode == 'train':
            if self.use_pair_data and random.random() <= self.pair_data_percent:
                return self.__get_pair_data(index % len(self.paired_imgs))
            else:
                return self.__get_unpair_data(index // 2)
        else:
            return self.__get_unpair_data(index)

    def __get_pair_data(self, index):
        paired_img = self.paired_imgs[index]
        source_emo = random.choice(self.emotions)
        source = paired_img[source_emo]
        source_actor = source.rsplit('/', 2)[1]

        driving_actor = random.choice(
            list(set(self.driving_actors)-set([source_actor])))
        driving_emo = random.choice(self.emotions)

        target = paired_img[driving_emo]

        img_D1 = random.choice(self.data_dict[source_actor][driving_emo])
        img_D2 = random.choice(self.data_dict[driving_actor][driving_emo])

        # lm_S = source.replace(
        #     'images_aligned', 'eye_landmarks_aligned').replace('png', 'txt')
        # lm_D1 = img_D1.replace(
        #     'images_aligned', 'eye_landmarks_aligned').replace('png', 'txt')
        # lm_D2 = img_D2.replace(
        #     'images_aligned', 'eye_landmarks_aligned').replace('png', 'txt')
        # lm_T = target.replace(
        #     'images_aligned', 'eye_landmarks_aligned').replace('png', 'txt')

        # mask_D1 = img_D1.replace('images_aligned', 'masks_aligned')
        # mask_T = target.replace('images_aligned', 'masks_aligned')

        source = Image.open(source).convert('RGB')
        source = self.transform_no_crop(source)

        img_T1 = Image.open(target).convert('RGB')
        img_T1 = self.transform_no_crop(img_T1)

        img_D1 = Image.open(img_D1).convert('RGB')
        img_D1 = self.transform_crop(img_D1)

        img_D2 = Image.open(img_D2).convert('RGB')
        img_D2 = self.transform_crop(img_D2)

        # mask_D1 = Image.open(mask_D1).convert('L')
        # mask_D1 = self.to_tensor(mask_D1)

        # mask_T = Image.open(mask_T).convert('L')
        # mask_T = self.to_tensor(mask_T)

        # lm_S = torch.from_numpy(np.loadtxt(lm_S, np.float32).reshape((-1, 2)))
        # lm_D1 = torch.from_numpy(np.loadtxt(
        #     lm_D1, np.float32).reshape((-1, 2)))
        # lm_D2 = torch.from_numpy(np.loadtxt(
        #     lm_D2, np.float32).reshape((-1, 2)))
        # lm_T = torch.from_numpy(np.loadtxt(
        #     lm_T, np.float32).reshape((-1, 2)))

        # if self.opts.loadSize != 256:
        #     ratio = self.opts.loadSize / 256
        #     shift = (self.opts.loadSize - self.opts.cropSize) / 2
        #     lm_S = lm_S * ratio - shift
        #     lm_D1 = lm_D1 * ratio - shift
        #     lm_D2 = lm_D2 * ratio - shift

        data = {
            "img_S": source,
            "img_D1": img_D1,
            "img_D2": img_D2,
            "img_T1": img_T1,
            "emo_S": source_emo,
            "emo_D": driving_emo,
            "actor_S": source_actor,
            "actor_D2": driving_actor,
            #"lm_S": lm_S,
            #"lm_D1": lm_D1,
            #"lm_D2": lm_D2,
            #"lm_T": lm_T,
            #"mask_D1": mask_D1,
            #"mask_T": mask_T,
            "is_paired": 1
        }

        return data

    def __get_unpair_data(self, index):
        source = self.source_paths[index]
        source_actor = source.split('/images_aligned/')[0].rsplit('/', 1)[-1]
        source_emo = self.source_emotions[index]

        driving_actor = random.choice(
            list(set(self.driving_actors)-set([source_actor])))
        driving_emo = random.choice(self.emotions)

        img_D1 = random.choice(self.data_dict[source_actor][driving_emo])
        img_D2 = random.choice(self.data_dict[driving_actor][driving_emo])

        # lm_S = source.replace(
        #     'images_aligned', 'eye_landmarks_aligned').replace('png', 'txt')
        # lm_D1 = img_D1.replace(
        #     'images_aligned', 'eye_landmarks_aligned').replace('png', 'txt')
        # lm_D2 = img_D2.replace(
        #     'images_aligned', 'eye_landmarks_aligned').replace('png', 'txt')
        
        # mask_D1 = img_D1.replace('images_aligned', 'masks_aligned')

        source = Image.open(source).convert('RGB')
        source = self.transform_crop(source)

        img_D1 = Image.open(img_D1).convert('RGB')
        img_D1 = self.transform_crop(img_D1)

        img_D2 = Image.open(img_D2).convert('RGB')
        img_D2 = self.transform_crop(img_D2)

        # mask_D1 = Image.open(mask_D1).convert('L')
        # mask_D1 = self.to_tensor(mask_D1)

        img_T1 = img_D1.clone()
        # mask_T = mask_D1.clone()
        is_paired = 0

        # lm_S = torch.from_numpy(np.loadtxt(lm_S, np.float32).reshape((-1, 2)))
        # lm_D1 = torch.from_numpy(np.loadtxt(
        #     lm_D1, np.float32).reshape((-1, 2)))
        # lm_D2 = torch.from_numpy(np.loadtxt(
        #     lm_D2, np.float32).reshape((-1, 2)))
        # lm_T = lm_S.clone()

        # if self.opts.loadSize != 256:
        #     ratio = self.opts.loadSize / 256
        #     shift = (self.opts.loadSize - self.opts.cropSize) / 2
        #     lm_S = lm_S * ratio - shift
        #     lm_D1 = lm_D1 * ratio - shift
        #     lm_D2 = lm_D2 * ratio - shift

        data = {
            "img_S": source,
            "img_D1": img_D1,
            "img_D2": img_D2,
            "img_T1": img_T1,
            "emo_S": source_emo,
            "emo_D": driving_emo,
            "actor_S": source_actor,
            "actor_D2": driving_actor,
            # "lm_S": lm_S,
            # "lm_D1": lm_D1,
            # "lm_D2": lm_D2,
            # "lm_T": lm_T,
            # "mask_D1": mask_D1,
            # "mask_T": mask_T,
            "is_paired": is_paired
        }

        return data


if __name__ == '__main__':
    import torch
    from options.train_options import TrainOptions

    opts = TrainOptions().parse()
    dataset = MEADPairedDataset(opts)
    print(len(dataset))
    data = dataset[0]
    for k, v in data.items():
        print(k, v.shape if isinstance(v, torch.Tensor) else v)

    # TODO return lm_T
