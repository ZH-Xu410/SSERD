import os
import torch
import shutil
import numpy as np
import torch.nn.functional as F
from models.psp_no_attn import pSp
from PIL import Image
from glob import glob
from tqdm import tqdm
import torchvision.transforms as T
from options.train_options import TrainOptions
from utils.common import tensor2im


transform = T.Compose([
    T.Resize((320, 320)),
    T.CenterCrop((256, 256)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

@torch.no_grad()
def inversion(model, images):
    codes, _, _ = model.encoder(
        images, return_feat=True, return_full=True)  # modified
    codes = codes + model.latent_avg.repeat(codes.shape[0], 1, 1)

    return codes


def get_data(images):
    batch = []
    for img in images:
        img = Image.open(img)
        img = transform(img)
        batch.append(img)
    return torch.stack(batch, dim=0).cuda()


@torch.no_grad()
def _edit(model, x, edit_w):
    codes, feats, P_S = model.encoder(
        x, return_feat=True, return_full=True)  # modified
    codes = codes + model.latent_avg.repeat(codes.shape[0], 1, 1)

    first_layer_feats = feats[0:2]  # use f
    skip_layer_feats = feats[2:]  # use skipped encoder feature
    # use fusion layer to fuse encoder feature and decoder feature.
    fusion = model.encoder.fusion

    if edit_w.ndim == 2:
        edit_w = edit_w.unsqueeze(0).repeat(codes.shape[0], 1, 1)

    image, result_latent = model.decoder([codes],
                                         input_is_latent=True,
                                         randomize_noise=False,
                                         return_latents=True,
                                         first_layer_feature=first_layer_feats,
                                         first_layer_feature_ind=0,
                                         skip_layer_feature=skip_layer_feats,
                                         fusion_block=fusion,
                                         zero_noise=True,
                                         editing_w=edit_w)
    image = F.interpolate(image, (256, 256), mode='bilinear')
    return image


def edit(model, editing_w, images, save_dir):
    img_id = 0
    for i in tqdm(range(0, len(images), batch_size)):
        imgs = images[i:min(i+batch_size, len(images))]
        batch = get_data(imgs)
        preds = _edit(model, batch, editing_w)
        for p in preds:
            tensor2im(p).save(os.path.join(save_dir, f'{img_id:06d}.png'))
            img_id += 1


if __name__ == '__main__':
    opts = TrainOptions().parse()
    model = pSp(opts).cuda().eval()
    batch_size = 2
    alpha = 15


    save_dir = 'data/MEAD/generated'
    for actor in ['M003', 'M009', 'W029', 'M012', 'M030', 'W015']: 
        img_list = sorted(glob(f'data/MEAD/train/{actor}/images_aligned/**/*.png', recursive=True))
        with open(f'data/MEAD/train/{actor}/videos/_frame_info.txt') as f:
            frame_infos = f.read().splitlines()
        idx2emo = {}
        for x in frame_infos:
            emo, idx = x.split(' ')
            emo = emo.split('_', 1)[0]
            idx2emo[int(idx)] = emo
    
        images = []
        for img in img_list:
            idx = int(os.path.basename(img).split('.')[0])
            emo = idx2emo[idx]
            if emo == 'neutral':
                images.append(img)
        
        actor_folder = os.path.join(save_dir, 'neutral', actor)
        os.makedirs(actor_folder, exist_ok=True)
        for i, img in enumerate(images):
            img = transform(Image.open(img))
            tensor2im(img).save(os.path.join(actor_folder, f'{i:06d}.png'))

        for emo in ['angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised']:
            print(actor, emo)
            editing_w = torch.load(f'exp/emo_direction/{emo}_{actor}.pth', 'cuda') * alpha
            actor_folder = os.path.join(save_dir, emo, actor)
            os.makedirs(actor_folder, exist_ok=True)
            edit(model, editing_w, images, actor_folder)
    

