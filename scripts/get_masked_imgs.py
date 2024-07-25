import os
import sys
import cv2
from tqdm import tqdm
from glob import glob


def exe_once(img, mask, save_path):
    img = cv2.imread(img)
    mask = cv2.imread(mask)
    mask = cv2.resize(mask, (320, 320))
    m = 32
    mask = mask[m:320-m,m:320-m,0]
    img[mask==0] = 0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)


if __name__ == '__main__':
    test_dir = sys.argv[1]
    if test_dir.endswith('/'):
        test_dir = test_dir[:-1]
    celeb_dir = 'data/MEAD'
    save_dir = test_dir + '_masked'

    for subdir in ('gen_intra', 'real_intra', 'gen_cross', 'real_cross'):
        if not os.path.exists(os.path.join(test_dir, subdir)):
            continue
        imgs = glob(os.path.join(test_dir, subdir, '**/*.png'), recursive=True)
        for img in tqdm(imgs):
            emo, actor, name = img.split(subdir+'/')[1].split('/')
            if 'self' in subdir:
                mask = os.path.join(celeb_dir, 'test', emo, actor, 'masks_aligned', name)
            else:
                mask = os.path.join(celeb_dir, 'test/neutral', actor, 'masks_aligned', name)
            save_path = os.path.join(save_dir, subdir, emo, actor, name)
            exe_once(img, mask, save_path)
