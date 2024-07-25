import os
import cv2
from skimage import io, img_as_float32, img_as_ubyte
from skimage.measure import label
import torch
import numpy as np
import argparse
from tqdm import tqdm
from glob import glob
from preprocessing.segmentation.simple_unet import UNet
from preprocessing.segment_face import get_face_masks, print_args
from postprocessing.image_blending.image_blender import SoftErosion
from postprocessing.image_blending.image_blender import Blend


def unalign(align_transform, mask, face_img):
    face = cv2.warpAffine(face_img, align_transform, (face_img.shape[1], face_img.shape[0]), flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LANCZOS4)
    face[np.where(mask==0)] = 0

    return face


def blend(blender, imgA, imgB, mask):
    imgA = img_as_float32(imgA)
    imgB = img_as_float32(imgB)
    mask = img_as_float32(mask)
    new_img = blender(imgA, imgB, mask)

    return img_as_ubyte(np.clip(new_img,0,1))


def main():
    print('---------- Face segmentation --------- \n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='Negative value to use CPU, or greater or equal than zero for GPU id.')
    parser.add_argument('--celeb_dir', type=str, default='/data/2022_stu/zihua/datasets/MEAD-6ID', help='Path to exp folder.')
    parser.add_argument('--exp_dir', type=str, default='exp/v7.1/test', help='Path to exp folder.')
    parser.add_argument('--resize_first', action='store_true', help='If specified, first resize image, then blend, else reversely')
    parser.add_argument('--save_images', action='store_true', help='If specified, save the cropped blended images, apart from the full frames')
    parser.add_argument('--method', type=str, default='pyramid', choices = ['copy_paste', 'pyramid', 'poisson'], help='Blending method')
    parser.add_argument('--n_levels', type=int, default=4, help='Number of levels of the laplacian pyramid, if pyramid blending is used')
    parser.add_argument('--n_levels_copy', type=int, default=0, help='Number of levels at the top of the laplacian pyramid to copy from image A')
    args = parser.parse_args()

    # Figure out the device
    gpu_id = int(args.gpu_id)
    if gpu_id < 0:
        device = 'cpu'
    elif torch.cuda.is_available():
        if gpu_id >= torch.cuda.device_count():
            device = 'cuda:0'
        else:
            device = 'cuda:' + str(gpu_id)
    else:
        print('GPU device not available. Exit')
        exit(0)

    # Print Arguments
    print_args(parser, args)

    # Load pretrained face segmenter
    # segmenter_path = 'preprocessing/segmentation/lfw_figaro_unet_256_2_0_segmentation_v1.pth'
    # checkpoint = torch.load(segmenter_path)
    # predictor = UNet(n_classes=3,feature_scale=1).to(device)
    # predictor.load_state_dict(checkpoint['state_dict'])
    # smooth_mask = SoftErosion(kernel_size=21, threshold=0.6).to(device)

    blender = Blend(method=args.method, n_levels=args.n_levels, n_levels_copy=args.n_levels_copy, device = device)

    for emo in 'neutral angry disgusted fear happy sad surprised'.split(' '):
        for actor in ('M003 M009 W029 M012 M030 W015'.split(' ')):
            for mode in ('inter', 'cross'):
                img_dir = os.path.join(args.exp_dir, 'gen_'+mode, emo, actor)
                imgs = sorted(glob(img_dir + '/*'))
                out_dir = os.path.join(args.exp_dir, 'gen2_'+mode, emo, actor)
                os.makedirs(out_dir, exist_ok=True)
                for i, img in enumerate(tqdm(imgs, desc=f'{emo} {actor} {mode}')):
                    img = cv2.imread(img)
                    mask = cv2.imread(os.path.join(args.celeb_dir, 'test/neutral', actor, 'masks', f'{i:06d}.png'))
                    align_transform = np.loadtxt(os.path.join(args.celeb_dir, 'test/neutral', actor, 'align_transforms', f'{i:06d}.txt'))
                    new_img = np.zeros((320, 320, 3), dtype=np.uint8)
                    new_img[32:320-32, 32:320-32, :] = img
                    new_img = cv2.resize(new_img, (256, 256), interpolation=cv2.INTER_LINEAR)
                    new_img = unalign(align_transform, mask, new_img)
                    src_img = cv2.imread(os.path.join(args.celeb_dir, 'test/neutral', actor, 'images', f'{i:06d}.png'))
                    out_img = blend(blender, src_img, new_img, mask)
                    cv2.imwrite(os.path.join(out_dir, f'{i:06d}.png'), out_img)

    print('DONE!')


if __name__=='__main__':
    main()





# 1. seg & pad -> aligned face & mask
# 2. unalign -> face & mask
# 3. blend -> images