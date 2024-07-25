import os
import torch
import random
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from utils.common import tensor2im
from models.psp_no_attn import pSp
from options.train_options import TrainOptions
from glob import glob
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm


transform = T.Compose([
    T.Resize((320, 320)),
    T.CenterCrop((256, 256)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


def fit(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = svm.SVC(kernel='linear')

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print("Accuracy on test set:", accuracy_score(y_test, y_pred))

    coefficients = model.coef_[0].reshape(18, 512)  

    edit_direction = coefficients / np.linalg.norm(coefficients, axis=1, keepdims=True)

    return edit_direction


@torch.no_grad()
def edit(model, x, edit_w):
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



def main(model, actor, emotion):
    img_list = sorted(glob(f'data/MEAD/train/{actor}/images_aligned/**/*.png', recursive=True))
    with open(f'data/MEAD/train/{actor}/videos/_frame_info.txt') as f:
        frame_infos = f.read().splitlines()
    idx2emo = {}
    for x in frame_infos:
        emo, idx = x.split(' ')
        emo = emo.split('_', 1)[0]
        idx2emo[int(idx)] = emo
    
    img_list1 = []
    img_list2 = []
    for i, img in enumerate(img_list):
        emo = idx2emo[i]
        if emo == 'neutral':
            img_list1.append(img)
        elif emo == emotion:
            img_list2.append(img)
        
    img_list = img_list1 + img_list2
    labels = np.concatenate([np.zeros([len(img_list1)]), np.ones([len(img_list2)])])

    latent_list = []
    for i in img_list:
        latent = np.load(i.replace('images_aligned', 'codes').replace('.png', '.npy'))
        latent_list.append(latent)
    latent_list = np.stack(latent_list, axis=0)

    direction = fit(latent_list.reshape(latent_list.shape[0], -1), labels)
    direction = torch.from_numpy(direction).float().cuda()
    torch.save(direction, f'exp/emo_direction/{emotion}_{actor}.pth')

    vis_dir = f'exp/emo_direction/vis_{emotion}_{actor}'
    os.makedirs(vis_dir, exist_ok=True)

    for i in tqdm(range(5)):
        image1 = random.choice(img_list1)
        image2 = random.choice(img_list2)
        image1 = transform(Image.open(image1)).unsqueeze(0).cuda()
        image2 = transform(Image.open(image2)).unsqueeze(0).cuda()

        tensor2im(image1[0]).save(f'{vis_dir}/{i}_x.png')
        #tensor2im(image2[0]).save(f'{vis_dir}/{i}_y.png')

        for alpha in range(12,22,2):
            out = edit(model, image1, direction*alpha)
            tensor2im(out[0]).save(f'{vis_dir}/{i}_x_edit_{alpha}.png')

        #out = edit(model, image2, -direction*alpha)
        #tensor2im(out[0]).save(f'{vis_dir}/{i}_y_edit_-{alpha}.png')


if __name__ == '__main__':
    opts = TrainOptions().parse()
    model = pSp(opts).cuda().eval()


    for actor in ['M003', 'M009', 'W029', 'M012', 'M030', 'W015']:
        for emotion in ['angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised']:
            print(actor, emotion)
            main(model, actor, emotion)

