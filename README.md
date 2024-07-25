# SSERD - Official Implementation
## Self-Supervised Emotion Representation Disentanglement for Speech-Preserving Facial Expression Manipulation (ACM MM 2024)

## 1. Installation

Python3.8, install the requirements via pip.
  ```bash
  pip install -r requirements.txt
  ```  

## 2. Download Pretrained Models

See the Release Page.

## 3. Prepare Dataset

Download the [MEAD](https://wywu.github.io/projects/MEAD/MEAD.html) or [RAVDESS](https://zenodo.org/records/1188976) dataset. 


Place the segmentor weights "lfw_figaro_unet_256_2_0_segmentation_v1.pth" under "preprocessing/segmentation/".

To train or test the method on a specific subject, first create a folder for this subject and place the video(s) of this subject into a "videos" subfolder. 

For example, for testing the method on M003's clip, a structure similar to the following must be created:
```
M003 ----- videos ----- 001.mp4
```

To preprocess the video (face detection, segmentation, landmark detection, alignment) run:
```bash
./scripts/preprocess.sh <celeb_path>
```
- ```<celeb_path>``` is the path to the folder used for this actor.
- ```<mode>``` is one of ```{train, test, reference}``` for each of the above cases respectively.

After successfull execution, the following structure will be created:

```
<celeb_path> ----- videos -----video.mp4 (e.g. "001.mp4")
                   |        |
                   |        ---video.txt (e.g. "001.txt", stores the per-frame bounding boxes, created only if mode=test)
                   |
                   --- images (cropped and resized images)
                   |
                   --- full_frames (original frames of the video, created only if mode=test or mode=reference)
                   |
                   --- eye_landmarks (landmarks for the left and right eyes, created only if mode=train or mode=test)
                   |
                   --- eye_landmarks_aligned (same as above, but aligned)
                   |
                   --- align_transforms (similarity transformation matrices, created only if mode=train or mode=test)
                   |
                   --- faces (segmented images of the face, created only if mode=train or mode=test)
                   |
                   --- faces_aligned (same as above, but aligned)
                   |
                   --- masks (binary face masks, created only if mode=train or mode=test)
                   |
                   --- masks_aligned (same as above, but aligned)
```


## 4. Training

- finetune stage one.
  ```bash
  python scripts/finetune_stage1.py --exp_dir exp/stage1 \
    --start_from_latent_avg \
    --use_skip --use_att 1 \
    --learning_rate 0.0001 \
    --max_steps 50000 \
    --val_interval 5000 \
    --checkpoint_path pretrained_models/pretrain.pt \
  ```

- generate paired data.
  ```bash
  # find emotion direction
  python scripts/get_emotion_direction.py \
    --start_from_latent_avg \
    --use_skip --use_att 1 \
    --checkpoint_path exp/stage1/checkpoints/best_model.pt
  
  # edit images
  python scripts/gen_edit_images.py \
    --start_from_latent_avg \
    --use_skip --use_att 1 \
    --checkpoint_path exp/stage1/checkpoints/best_model.pt
  ```

- finetune stage two.
  ```bash
  python scripts/finetune_stage2.py --exp_dir exp/stage2 \
    --start_from_latent_avg \
    --use_skip --use_att 1 \
    --learning_rate 0.0001 \
    --max_steps 50000 \
    --val_interval 5000 \
    --checkpoint_path exp/stage1/checkpoints/best_model.pt
  ```

## 5. Test
```bash
# for intra-ID
python scripts/test_intra.py --exp_dir exp/test --checkpoint_path exp/stage2/checkpoints/best_model.pt --data_path data/MEAD
# for cross-ID
python scripts/test_cross.py --exp_dir exp/test --checkpoint_path exp/stage2/checkpoints/best_model.pt --data_path data/MEAD
```
For quantitative evaluation, we use the masked face images. Use "scripts/get_masked_imgs.py" to obtain masked faces.
## Acknowledgements

Codes are Codes are mainly developed based on [StyleGANEX](https://github.com/williamyang1991/StyleGANEX).  We thank them for their wonderful work.