import shutil
from models.psp import pSp
from options.test_options import TestOptions
from utils.common import tensor2im
from datasets.mead_dataset import MEADTestDataset
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


crossID_map = {
    'M003': 'W015',
    'M009': 'M003',
    'W029': 'M009',
    'M012': 'W029',
    'M030': 'M012',
    'W015': 'M030'
}



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
            out_path_cross = os.path.join(
                test_opts.exp_dir, 'gen_cross', emo, actor)
            out_path_real = os.path.join(test_opts.exp_dir, 'real_cross', emo, actor)
            os.makedirs(out_path_cross)
            os.makedirs(out_path_real)

            dataset = MEADTestDataset(
                os.path.join(test_opts.data_path, 'test/neutral', actor),
                os.path.join(test_opts.data_path, 'test', emo, actor),
                os.path.join(test_opts.data_path, 'test',
                             emo, crossID_map[actor]),
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
                    S, D1, D2, lm_S, lm_D1, lm_D2, path = batch
                    S = S.cuda()
                    D1 = D1.cuda()
                    D2 = D2.cuda()
                    lm_S = lm_S.cuda()
                    lm_D1 = lm_D1.cuda()
                    lm_D2 = lm_D2.cuda()

                    _, SD2 = net(S, None, D2, first_layer_feature_ind=opts.feat_ind, use_skip=opts.use_skip)
                    bs = S.shape[0]
                    for j in range(bs):
                        img_cross = tensor2im(SD2[j])
                        img_cross.save(os.path.join(out_path_cross, f'{i:06d}.png'))
                        real_path = os.path.join(out_path_real, os.path.basename(path[j]))
                        if not os.path.exists(real_path):
                            tensor2im(D1[j]).save(real_path)
                        i += 1


if __name__ == '__main__':
    run()
