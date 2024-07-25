import shutil
from models.psp import pSp
from options.test_options import TestOptions
from utils.common import tensor2im
from datasets.mead_dataset import MEADTestIntraDataset
import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys

sys.path.append(".")


def calc_lip_dist(landmark):
    d1 = ((landmark[:, 27]-landmark[:, 33])**2).sum(1) ** 0.5
    d2 = ((landmark[:, 28]-landmark[:, 32])**2).sum(1) ** 0.5
    d3 = ((landmark[:, 29]-landmark[:, 31])**2).sum(1) ** 0.5
    return (d1+d2+d3)/3


def get_lip_coeff_targets(lm_S, lm_D1, lm_D2):
    dist_S = calc_lip_dist(lm_S)
    dist_D1 = calc_lip_dist(lm_D1)
    dist_D2 = calc_lip_dist(lm_D2)

    targets_D1 = (((dist_S - dist_D1)/30).clamp_(-1, 1) + 1) / 2
    targets_D2 = (((dist_S - dist_D2)/30).clamp_(-1, 1) + 1) / 2

    return targets_D1, targets_D2


def run():
    test_opts = TestOptions().parse()
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
        
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    opts = Namespace(**opts)

    if os.path.exists(test_opts.exp_dir):
        shutil.rmtree(test_opts.exp_dir)

    net = pSp(opts, ckpt).eval().cuda()

    for actor in ('M003 M009 W029 M012 M030 W015'.split(' ')):
        for emo in opts.emotions:
            out_path_gen = os.path.join(
                test_opts.exp_dir, 'gen_intra', emo, actor)
            out_path_real = os.path.join(test_opts.exp_dir, 'real_intra', emo, actor)
            os.makedirs(out_path_gen)
            os.makedirs(out_path_real)

            dataset = MEADTestIntraDataset(
                os.path.join(test_opts.data_path, 'test', emo, actor),
                opts
            )
            dataloader = DataLoader(dataset,
                                    batch_size=opts.test_batch_size,
                                    shuffle=False,
                                    num_workers=int(opts.test_workers),
                                    drop_last=False)

            print(f'testing {emo} {actor}')
            i = 0
            with torch.no_grad():
                for batch in tqdm(dataloader):
                    S, lm_S, path = batch
                    S = S.cuda()
                    lm_S = lm_S.cuda()

                    SD1, _ = net(S, S, None, first_layer_feature_ind=opts.feat_ind, use_skip=opts.use_skip)
                    bs = S.shape[0]
                    for j in range(bs):
                        img = tensor2im(SD1[j])
                        img.save(os.path.join(out_path_gen, f'{i:06d}.png'))
                        real_path = os.path.join(out_path_real, os.path.basename(path[j]))
                        if not os.path.exists(real_path):
                            tensor2im(S[j]).save(real_path)
                        i += 1


if __name__ == '__main__':
    run()
