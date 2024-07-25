import os
import random
import torch
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset
from glob import glob
import numpy as np


class MEADDataset(Dataset):
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
        self.data_dict = {}

        all_actors = set(self.source_actors).union(set(self.driving_actors))
        for actor in all_actors:
            self.data_dict[actor] = {emo: [] for emo in self.emotions}
            with open(os.path.join(self.data_root, mode, actor, 'videos/_frame_info.txt')) as f:
                frame_infos = f.read().splitlines()
            frame_infos = random.sample(frame_infos, min(len(frame_infos), max_num))
            for info in frame_infos:
                emo, global_idx = info.split(' ')
                emo = emo.split('_', 1)[0]
                global_idx = int(global_idx)
                if mode == 'train':
                    seq_idx = global_idx // 50
                    img_path = os.path.join(self.data_root, mode, actor, 'images_aligned', f'{seq_idx:06d}', f'{global_idx:06d}.png')
                else:
                    img_path = os.path.join(self.data_root, mode, actor, 'images_aligned', f'{global_idx:06d}.png')
                
                if not os.path.exists(img_path):
                    continue

                self.data_dict[actor][emo].append(img_path)
                if actor in self.source_actors:
                    self.source_paths.append(img_path)
                    self.source_emotions.append(emo)

        self.transform = T.Compose([
            T.Resize((opts.loadSize, opts.loadSize)),
            T.CenterCrop((opts.cropSize, opts.cropSize)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        source = self.source_paths[index]
        source_actor = source.split('/images_aligned/')[0].rsplit('/', 1)[-1]
        source_emo = self.source_emotions[index]

        driving_actor = random.choice(list(set(self.driving_actors)-set([source_actor])))
        driving_emo = random.choice(self.emotions)

        img_D1 = random.choice(self.data_dict[source_actor][source_emo]) # driving_emo
        img_D2 = random.choice(self.data_dict[driving_actor][driving_emo])

        lm_S = source.replace('images_aligned', 'eye_landmarks_aligned').replace('png', 'txt')
        lm_D1 = img_D1.replace('images_aligned', 'eye_landmarks_aligned').replace('png', 'txt')
        lm_D2 = img_D2.replace('images_aligned', 'eye_landmarks_aligned').replace('png', 'txt')

        mask_D1 = img_D1.replace('images_aligned', 'masks_aligned')

        source = Image.open(source).convert('RGB')
        source = self.transform(source)

        img_D1 = Image.open(img_D1).convert('RGB')
        img_D1 = self.transform(img_D1)

        img_D2 = Image.open(img_D2).convert('RGB')
        img_D2 = self.transform(img_D2)

        mask_D1 = Image.open(mask_D1).convert('L')
        mask_D1 = self.to_tensor(mask_D1)

        lm_S = torch.from_numpy(np.loadtxt(lm_S, np.float32).reshape((-1, 2)))
        lm_D1 = torch.from_numpy(np.loadtxt(lm_D1, np.float32).reshape((-1, 2)))
        lm_D2 = torch.from_numpy(np.loadtxt(lm_D2, np.float32).reshape((-1, 2)))

        if self.opts.loadSize != 256:
            ratio = self.opts.loadSize / 256
            shift = (self.opts.loadSize - self.opts.cropSize) / 2
            lm_S = lm_S * ratio - shift
            lm_D1 = lm_D1 * ratio - shift
            lm_D2 = lm_D2 * ratio - shift

        data = {
            "img_S": source,
            "img_D1": img_D1,
            "img_D2": img_D2,
            "emo_S": source_emo,
            "emo_D": driving_emo,
            "actor_S": source_actor,
            "actor_D2": driving_actor,
            "lm_S": lm_S,
            "lm_D1": lm_D1,
            "lm_D2": lm_D2,
            "mask_D1": mask_D1
        }

        return data


class MEADTestDataset(Dataset):
    def __init__(self, source, interID, crossID, opts):
        self.opts = opts
        self.source_imgs = list(sorted(glob(os.path.join(source, 'images_aligned/*'))))
        self.interID_imgs = list(sorted(glob(os.path.join(interID, 'images_aligned/*'))))
        self.crossID_imgs = list(sorted(glob(os.path.join(crossID, 'images_aligned/*'))))

        self.r1 = len(self.source_imgs)/len(self.interID_imgs)
        self.r2 = len(self.source_imgs)/len(self.crossID_imgs)

        self.transform = T.Compose([
            T.Resize((opts.loadSize, opts.loadSize)),
            T.CenterCrop((opts.cropSize, opts.cropSize)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.source_imgs)
    
    def __getitem__(self, index):
        src = self.source_imgs[index]
        ref1 = self.interID_imgs[int(index/self.r1)]
        ref2 = self.crossID_imgs[int(index/self.r2)]

        src_img = Image.open(src).convert('RGB')
        src_img = self.transform(src_img)

        img_D1 = Image.open(ref1).convert('RGB')
        img_D1 = self.transform(img_D1)

        img_D2 = Image.open(ref2).convert('RGB')
        img_D2 = self.transform(img_D2)

        lm_S = src.replace('images_aligned', 'eye_landmarks_aligned').replace('png', 'txt')
        lm_D1 = ref1.replace('images_aligned', 'eye_landmarks_aligned').replace('png', 'txt')
        lm_D2 = ref2.replace('images_aligned', 'eye_landmarks_aligned').replace('png', 'txt')

        lm_S = torch.from_numpy(np.loadtxt(lm_S, np.float32).reshape((-1, 2)))
        lm_D1 = torch.from_numpy(np.loadtxt(lm_D1, np.float32).reshape((-1, 2)))
        lm_D2 = torch.from_numpy(np.loadtxt(lm_D2, np.float32).reshape((-1, 2)))

        if self.opts.loadSize != 256:
            ratio = self.opts.loadSize / 256
            shift = (self.opts.loadSize - self.opts.cropSize) / 2
            lm_S = lm_S * ratio - shift
            lm_D1 = lm_D1 * ratio - shift
            lm_D2 = lm_D2 * ratio - shift

        return src_img, img_D1, img_D2, lm_S, lm_D1, lm_D2, ref1



class MEADTestIntraDataset(Dataset):
    def __init__(self, source,opts):
        super().__init__()
        self.opts = opts
        self.source_imgs = list(sorted(glob(os.path.join(source, 'images_aligned/*'))))

        self.transform = T.Compose([
            T.Resize((opts.loadSize, opts.loadSize)),
            T.CenterCrop((opts.cropSize, opts.cropSize)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.source_imgs)
    
    def __getitem__(self, index):
        src = self.source_imgs[index]
        src_img = Image.open(src).convert('RGB')
        src_img = self.transform(src_img)

        lm_S = src.replace('images_aligned', 'eye_landmarks_aligned').replace('png', 'txt')
        lm_S = torch.from_numpy(np.loadtxt(lm_S, np.float32).reshape((-1, 2)))

        if self.opts.loadSize != 256:
            ratio = self.opts.loadSize / 256
            shift = (self.opts.loadSize - self.opts.cropSize) / 2
            lm_S = lm_S * ratio - shift

        return src_img, lm_S, src


class MEADSeqDataset(Dataset):
    def __init__(self, opts, mode='train', max_num=15000):
        self.opts = opts
        self.data_root = opts.data_root
        self.source_actors = opts.source_actors
        self.driving_actors = opts.driving_actors
        self.emotions = opts.emotions
        self.mode = mode
        self.source_paths = []
        self.source_emotions = [] 
        self.frame_infos = []
        self.data_dict = {}

        all_actors = set(self.source_actors).union(set(self.driving_actors))
        for actor in all_actors:
            self.data_dict[actor] = {emo: [] for emo in self.emotions}
            with open(os.path.join(self.data_root, mode, actor, 'videos/_frame_info.txt')) as f:
                frame_infos = f.read().splitlines()
            #frame_infos = frame_infos[:min(len(frame_infos), max_num)]
            for info in frame_infos:
                emo, global_idx = info.split(' ')
                emo, v, fid = emo.split('_')
                global_idx = int(global_idx)
                if mode == 'train':
                    seq_idx = global_idx // 50
                    img_path = os.path.join(self.data_root, mode, actor, 'images_aligned', f'{seq_idx:06d}', f'{global_idx:06d}.png')
                else:
                    img_path = os.path.join(self.data_root, mode, actor, 'images_aligned', f'{global_idx:06d}.png')
                
                if not os.path.exists(img_path):
                    continue

                self.data_dict[actor][emo].append(img_path)
                if actor in self.source_actors:
                    self.source_paths.append(img_path)
                    self.source_emotions.append(emo)
                    self.frame_infos.append((emo+v, int(fid)))

        self.transform = T.Compose([
            T.Resize((opts.loadSize, opts.loadSize)),
            T.CenterCrop((opts.cropSize, opts.cropSize)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.source_paths) // 2

    def __get_once(self, index, driving_emo=None):
        source = self.source_paths[index]
        source_actor = source.split('/images_aligned/')[0].rsplit('/', 1)[-1]
        source_emo = self.source_emotions[index]

        driving_actor = random.choice(list(set(self.driving_actors)-set([source_actor])))
        #driving_emo = driving_emo or random.choice(self.emotions)
        if driving_emo is None:
            driving_emo = source_emo if random.random() >= 0.5 else random.choice(self.emotions)

        img_D1 = random.choice(self.data_dict[source_actor][driving_emo])
        img_D2 = random.choice(self.data_dict[driving_actor][driving_emo])

        lm_S = source.replace('images_aligned', 'eye_landmarks_aligned').replace('png', 'txt')
        lm_D1 = img_D1.replace('images_aligned', 'eye_landmarks_aligned').replace('png', 'txt')
        lm_D2 = img_D2.replace('images_aligned', 'eye_landmarks_aligned').replace('png', 'txt')

        source = Image.open(source).convert('RGB')
        source = self.transform(source)

        img_D1 = Image.open(img_D1).convert('RGB')
        img_D1 = self.transform(img_D1)

        img_D2 = Image.open(img_D2).convert('RGB')
        img_D2 = self.transform(img_D2)

        lm_S = torch.from_numpy(np.loadtxt(lm_S, np.float32).reshape((-1, 2)))
        lm_D1 = torch.from_numpy(np.loadtxt(lm_D1, np.float32).reshape((-1, 2)))
        lm_D2 = torch.from_numpy(np.loadtxt(lm_D2, np.float32).reshape((-1, 2)))

        if self.opts.loadSize != 256:
            ratio = self.opts.loadSize / 256
            shift = (self.opts.loadSize - self.opts.cropSize) / 2

            lm_S = lm_S * ratio - shift
            lm_D1 = lm_D1 * ratio - shift
            lm_D2 = lm_D2 * ratio - shift

        data = {
            "img_S": source,
            "img_D1": img_D1,
            "img_D2": img_D2,
            "emo_S": source_emo,
            "emo_D": driving_emo,
            "actor_S": source_actor,
            "actor_D2": driving_actor,
            "lm_S": lm_S,
            "lm_D1": lm_D1,
            "lm_D2": lm_D2
        }

        return data, driving_emo
    
    def merge_data(self, data1, data2):
        data = {}
        for k in data1.keys():
            data[k] = [data1[k], data2[k]]
            if isinstance(data1[k], torch.Tensor):
                data[k] = torch.stack(data[k], dim=0)
        return data
    
    def __getitem__(self, index):
        i = index * 2
        j = i + 1
        data1, driving_emo = self.__get_once(i)
        v, fid = self.frame_infos[i]
        if j >= len(self.frame_infos) or self.frame_infos[j][1] != fid+1:
            j = i - 1
            if self.frame_infos[j][0] == v and self.frame_infos[j][1] == fid - 1:
                data0, _ = self.__get_once(j, driving_emo)
            else:
                data0 = data1
            data = self.merge_data(data0, data1)
        else:
            if self.frame_infos[j][0] == v and self.frame_infos[j][1] == fid + 1:
                data2, _ = self.__get_once(j, driving_emo)
            else:
                data2 = data1
            data = self.merge_data(data1, data2)
        
        return data


if __name__ == '__main__':
    import torch
    from options.train_options import TrainOptions

    opts = TrainOptions().parse()
    dataset = MEADDataset(opts)
    print(len(dataset))
    data = dataset[0]
    for k, v in data.items():
        print(k, v.shape if isinstance(v, torch.Tensor) else v)

