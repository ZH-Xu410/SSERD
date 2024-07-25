from models.stylegan2.model import Discriminator  # modified
from torch import nn, autograd  # modified
from datasets.mead_dataset import MEADDataset
from training.ranger import Ranger
from models.psp_no_attn import pSp
from criteria.lpips.lpips import LPIPS
from criteria import id_loss, w_norm
from utils import common, train_utils
from utils.latent_pool import LatentCodePool
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch
from torchvision.ops import roi_align
import os
import math
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.psp import FacialComponentDiscriminator
import torchvision.transforms as T
from copy import deepcopy

matplotlib.use('Agg')


class Coach:
    def __init__(self, opts):
        self.opts = opts

        self.global_step = 0

        self.device = 'cuda'  # TODO: Allow multiple GPU? currently using CUDA_VISIBLE_DEVICES
        self.opts.device = self.device

        if self.opts.use_wandb:
            from utils.wandb_utils import WBLogger
            self.wb_logger = WBLogger(self.opts)

        # Initialize network
        self.net = pSp(self.opts).to(self.device)
        self.net_clone = deepcopy(self.net)
        for p in self.net_clone.parameters():
            p.requires_grad = False

        # self.lip_direction = torch.load(self.opts.lip_direction, self.device)

        self.latent_pool = LatentCodePool(
            8, emotions=self.opts.emotions, actors=self.opts.source_actors)

        if self.opts.checkpoint_path is not None:
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            if 'latent_pool' in ckpt:
                self.latent_pool.load(ckpt['latent_pool'])
        else:
            ckpt = None

        if self.opts.adv_lambda > 0:  # modified, add discriminator
            self.discriminator = Discriminator(
                1024, channel_multiplier=2, img_channel=3)
            if ckpt is not None:
                if 'discriminator' in ckpt.keys():
                    print('Loading discriminator from checkpoint: {}'.format(
                        self.opts.checkpoint_path))
                    self.discriminator.load_state_dict(
                        ckpt['discriminator'], strict=False)
            self.discriminator.to(self.device)
            self.left_eye_disc = FacialComponentDiscriminator().to(self.device)
            self.right_eye_disc = FacialComponentDiscriminator().to(self.device)
            self.mouth_disc = FacialComponentDiscriminator().to(self.device)
            self.discriminator_optimizer = torch.optim.Adam(
                list(self.discriminator.parameters()), lr=self.opts.learning_rate)
            self.left_eye_disc_optimizer = torch.optim.Adam(
                list(self.left_eye_disc.parameters()), lr=self.opts.learning_rate)
            self.right_eye_disc_optimizer = torch.optim.Adam(
                list(self.right_eye_disc.parameters()), lr=self.opts.learning_rate)
            self.mouth_disc_optimizer = torch.optim.Adam(
                list(self.mouth_disc.parameters()), lr=self.opts.learning_rate)

        # Estimate latent_avg via dense sampling if latent_avg is not available
        if self.net.latent_avg is None:
            self.net.latent_avg = self.net.decoder.mean_latent(int(1e5))[
                0].detach()

        self.mse_loss = nn.MSELoss().to(self.device)
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type='vgg').to(self.device).eval()
        if self.opts.id_lambda > 0:
            self.id_loss = id_loss.IDLoss(self.opts.ir_se50_weights).to(self.device).eval()
        if self.opts.w_norm_lambda > 0:
            self.w_norm_loss = w_norm.WNormLoss(
                start_from_latent_avg=self.opts.start_from_latent_avg)

        # Initialize optimizer
        self.optimizer = self.configure_optimizers()

        # Initialize dataset
        self.train_dataset, self.test_dataset = self.configure_datasets()
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.opts.batch_size,
                                           shuffle=True,
                                           num_workers=int(self.opts.workers),
                                           drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.opts.test_batch_size,
                                          shuffle=False,
                                          num_workers=int(
                                              self.opts.test_workers),
                                          drop_last=True)

        # Initialize logger
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

    def train(self):
        self.net.train()
        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()

                # ************************ Data Preparation **************************
                x = batch['img_S'].cuda()
                y = batch['img_D1'].cuda()
                lm_S = batch['lm_S'].cuda()
                emo_S = batch['emo_S']
                actor_S = batch['actor_S']

                # ************************ forward **************************

                # y_hat is the output image, latent is the extracted w+
                recon_x, latents_x = self.net(x, resize=True, zero_noise=self.opts.zero_noise,
                                           first_layer_feature_ind=self.opts.feat_ind,
                                           use_skip=self.opts.use_skip, return_latents=True)
                _, latents_y = self.net(y, resize=True, zero_noise=self.opts.zero_noise,
                                           first_layer_feature_ind=self.opts.feat_ind,
                                           use_skip=self.opts.use_skip, return_latents=True)
                edit_x, _ = self.net(x, editing_w=latents_y-latents_x, resize=True, zero_noise=self.opts.zero_noise,
                                           first_layer_feature_ind=self.opts.feat_ind,
                                           use_skip=self.opts.use_skip, return_latents=True)

                if self.global_step < self.opts.contrastive_start_step:
                    self.latent_pool.add_latent(emo_S, actor_S, latents_x)

                if self.opts.use_facial_disc:
                    src_facial_component = self.get_roi_regions(x, lm_S)
                    gen_facial_component = self.get_roi_regions(recon_x, lm_S)
                    facial_component = (
                        src_facial_component, gen_facial_component)
                else:
                    facial_component = None

                # adversarial loss
                if self.opts.adv_lambda > 0:
                    loss_dict = self.train_discriminator(
                        x, recon_x, facial_component)
                else:
                    loss_dict = {}

                # calculate losses
                loss, G_loss_dict, id_logs = self.calc_loss(
                    x, y, recon_x, edit_x, actor_S, emo_S, latents_x, facial_component)

                loss_dict.update(G_loss_dict)

                loss.backward()
                self.optimizer.step()

                # ************************ logging and saving model**************************

                if self.global_step % self.opts.image_interval == 0 or (self.global_step < 1000 and self.global_step % 25 == 0):
                    self.parse_and_log_images(
                        id_logs, x, y, x, y, recon_x, edit_x, title='images/train/faces')
                    #self.parse_and_log_images(
                    #    id_logs, x, pseudo_D1, pseudo_D2, SD1_pi, SD2_pi, title='images/train/faces2')
                    if facial_component is not None:
                        self.parse_and_log_facial_components(*(facial_component[0]+facial_component[1]),
                                                             title='images/train/components',
                                                             subscript='{:04d}'.format(batch_idx))
                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')

                # Log images of first batch to wandb
                # if self.opts.use_wandb and batch_idx == 0:
                #     self.wb_logger.log_images_to_wandb(
                #         x, D1, D2, SD1, SD2, id_logs, prefix="train", step=self.global_step, opts=self.opts)

                # Validation related
                val_loss_dict = None
                if (self.global_step + 1) % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                    val_loss_dict = self.validate()
                    if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
                        self.best_val_loss = val_loss_dict['loss']
                        self.checkpoint_me(val_loss_dict, is_best=True)

                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)

                if self.global_step == self.opts.max_steps:
                    print('Training finished.')
                    break

                self.global_step += 1

    @torch.no_grad()
    def validate(self):
        self.net.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(tqdm(self.test_dataloader)):
            x = batch['img_S'].cuda()
            y = batch['img_D1'].cuda()
            lm_S = batch['lm_S'].cuda()
            emo_S = batch['emo_S']
            actor_S = batch['actor_S']


            # ************************ forward **************************

            # y_hat is the output image, latent is the extracted w+
            recon_x, latents_x = self.net(x, resize=True, zero_noise=self.opts.zero_noise,
                                          first_layer_feature_ind=self.opts.feat_ind,
                                          use_skip=self.opts.use_skip, return_latents=True)
            _, latents_y = self.net(y, resize=True, zero_noise=self.opts.zero_noise,
                                           first_layer_feature_ind=self.opts.feat_ind,
                                           use_skip=self.opts.use_skip, return_latents=True)
            edit_x, _ = self.net(x, editing_w=latents_y-latents_x, resize=True, zero_noise=self.opts.zero_noise,
                                           first_layer_feature_ind=self.opts.feat_ind,
                                           use_skip=self.opts.use_skip, return_latents=True)

            if self.opts.use_facial_disc:
                src_facial_component = self.get_roi_regions(x, lm_S)
                gen_facial_component = self.get_roi_regions(recon_x, lm_S)
                facial_component = (
                    src_facial_component, gen_facial_component)
            else:
                facial_component = None

            # adversarial loss
            cur_d_loss_dict = {}
            if self.opts.adv_lambda > 0:
                cur_d_loss_dict = self.validate_discriminator(
                    x, recon_x, facial_component)

            loss, cur_loss_dict, id_logs = self.calc_loss(
                x, y, recon_x, edit_x, actor_S, emo_S, latents_x, facial_component)

            #cur_loss_dict = {**cur_d_loss_dict, **cur_loss_dict}

            agg_loss_dict.append(cur_loss_dict)

            # self.parse_and_log_images(id_logs, x, D1, D2, SD1, SD2,
            #                           title='images/test/faces',
            #                           subscript='{:04d}'.format(batch_idx))
            # if facial_component is not None:
            #     self.parse_and_log_facial_components(*facial_component[0],
            #                               title='images/test/components',
            #                               subscript='{:04d}'.format(batch_idx))

            # Log images of first batch to wandb
            # if self.opts.use_wandb and batch_idx == 0:
            #     self.wb_logger.log_images_to_wandb(
            #         x, D1, D2, SD1, SD2, id_logs, prefix="test", step=self.global_step, opts=self.opts)

            # For first step just do sanity test on small amount of data
            if self.global_step == 0 and batch_idx >= 4:
                self.net.train()
                return None  # Do not log, inaccurate in first batch

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.train()
        return loss_dict

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(
                    f'**Best**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n')
                if self.opts.use_wandb:
                    self.wb_logger.log_best_model()
            else:
                f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

    def configure_optimizers(self):
        # if hasattr(self.opts, 'pretrain_model') and self.opts.pretrain_model == 'input_label_layer':  # modified
        #     params = list(self.net.encoder.input_label_layer.parameters())
        # else:
        #     params = list(self.net.encoder.parameters())

        params = list(self.net.encoder.parameters())
        #params += list(self.net.cross_attention.parameters())
        #params += list(self.net.transformer.parameters())

        # if self.opts.use_sft:
        #     params += list(self.net.condition_scale.parameters())
        #     params += list(self.net.condition_shift.parameters())

        # if self.opts.train_decoder:
        #     params += list(self.net.decoder.parameters())

        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate)
        return optimizer

    def configure_datasets(self):
        train_dataset = MEADDataset(self.opts, mode='train')
        test_dataset = MEADDataset(self.opts, mode='train', max_num=100)
        if self.opts.use_wandb:
            self.wb_logger.log_dataset_wandb(
                train_dataset, dataset_name="Train")
            self.wb_logger.log_dataset_wandb(test_dataset, dataset_name="Test")
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of test samples: {len(test_dataset)}")
        return train_dataset, test_dataset

    def calc_loss(self, x, y, recon_x, edit_x, actors, emotions, latents, facial_component=None):
        loss_dict = {}
        loss = 0.0
        id_logs = None

        if self.opts.id_lambda > 0:
            loss_id, sim_improvement, id_logs = self.id_loss(recon_x, x, x)
            loss_id += self.id_loss(edit_x, y, y)[0]
            loss_id *= self.opts.id_lambda
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss += loss_id

        if self.opts.l2_lambda > 0: 
            loss_l2 = (F.mse_loss(recon_x, x) + F.mse_loss(edit_x, y)) * self.opts.l2_lambda
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2
            
        if self.opts.lpips_lambda > 0:
            loss_lpips = (self.lpips_loss(recon_x, x).mean() + self.lpips_loss(edit_x, y).mean()) * self.opts.lpips_lambda
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips

        if self.opts.adv_lambda > 0:
            pred = torch.cat([recon_x, edit_x], dim=0)
            loss_g = F.softplus(-self.discriminator(pred)
                                ).mean() * self.opts.adv_lambda
            loss_dict['loss_g'] = float(loss_g)
            loss += loss_g

        if self.opts.emo_contrastive > 0 and self.global_step >= self.opts.contrastive_start_step:
            cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
            tau = 0.1
            result = []
            for i in range(latents.shape[0]):
                try:
                    w_query = self.latent_pool.query(emotions[i], actors[i])
                except:
                    continue
                sim = cos(
                    latents[i:i+1].repeat(w_query.shape[0], 1, 1), w_query).mean(1).unsqueeze(0)
                value = F.cross_entropy(
                    sim / tau, target=torch.tensor([0], device=self.device))
                result.append(value)
            if len(result) > 0:
                loss_emo_contrastive = torch.stack(
                result, dim=0).mean() * self.opts.emo_contrastive
            else:
                loss_emo_contrastive = 0
            loss_dict['loss_emo_contrastive'] = float(loss_emo_contrastive)
            loss += loss_emo_contrastive

            if self.net.training:
                self.latent_pool.add_latent(emotions, actors, latents)

        if facial_component is not None:  # (SD1, SD2)
            # only use D1 for feature loss
            left_eye_gt = facial_component[0][0]
            right_eye_gt = facial_component[0][1]
            mouth_gt = facial_component[0][2]
            left_eye_pred = facial_component[1][0]
            right_eye_pred = facial_component[1][1]
            mouth_pred = facial_component[1][2]

            left_eye_fake, left_eye_fake_feat = self.left_eye_disc(
                left_eye_pred, return_feats=True)
            right_eye_fake, right_eye_fake_feat = self.right_eye_disc(
                right_eye_pred, return_feats=True)
            mouth_fake, mouth_fake_feat = self.mouth_disc(
                mouth_pred, return_feats=True)

            loss_g_left_eye = F.softplus(-left_eye_fake).mean() * \
                self.opts.adv_lambda
            loss_g_right_eye = F.softplus(-right_eye_fake).mean() * \
                self.opts.adv_lambda
            loss_g_mouth = F.softplus(-mouth_fake).mean() * \
                self.opts.adv_lambda

            loss += loss_g_left_eye + loss_g_right_eye + loss_g_mouth

            loss_dict['loss_g_left_eye'] = float(loss_g_left_eye)
            loss_dict['loss_g_right_eye'] = float(loss_g_right_eye)
            loss_dict['loss_g_mouth'] = float(loss_g_mouth)

            if self.opts.facial_feat_lambda > 0:
                _, left_eye_real_feat = self.left_eye_disc(
                    left_eye_gt, return_feats=True)
                _, right_eye_real_feat = self.right_eye_disc(
                    right_eye_gt, return_feats=True)
                _, mouth_real_feat = self.mouth_disc(
                    mouth_gt, return_feats=True)

                left_eye_feat_loss = self.facial_feature_loss(
                    [left_eye_fake_feat[0], left_eye_fake_feat[1]], left_eye_real_feat)
                right_eye_feat_loss = self.facial_feature_loss(
                    [right_eye_fake_feat[0], right_eye_fake_feat[1]], right_eye_real_feat)
                mouth_feat_loss = self.facial_feature_loss(
                    [mouth_fake_feat[0], mouth_fake_feat[1]], mouth_real_feat)

                loss += (left_eye_feat_loss + right_eye_feat_loss + mouth_feat_loss) * \
                    self.opts.facial_feat_lambda
                loss_dict['loss_left_eye_feat'] = float(
                    left_eye_feat_loss) * self.opts.facial_feat_lambda
                loss_dict['loss_right_eye_feat'] = float(
                    right_eye_feat_loss) * self.opts.facial_feat_lambda
                loss_dict['loss_mouth_feat'] = float(
                    mouth_feat_loss) * self.opts.facial_feat_lambda

        loss_dict['loss'] = float(loss)

        return loss, loss_dict, id_logs

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

    def facial_feature_loss(self, feat, feat_gt):
        return F.l1_loss(self._gram_mat(feat[0]), self._gram_mat(feat_gt[0].detach())) * 0.5 + F.l1_loss(self._gram_mat(feat[1]), self._gram_mat(feat_gt[1].detach()))

    def get_mouth_center(self, landmarks):
        pts = landmarks[:, 14:]
        mouth_center = torch.median(pts, dim=1).values
        return mouth_center

    def get_eyes_center(self, landmarks):
        left_pts = landmarks[:, 0:6]
        right_pts = landmarks[:, 6:12]
        left_eye_center = torch.median(left_pts, dim=1).values
        right_eye_center = torch.median(right_pts, dim=1).values
        return left_eye_center, right_eye_center

    def get_roi_regions(self, images, landmarks, eye_out_size=60, mouth_out_size=80):
        rois_eyes = []
        rois_mouths = []
        left_eye_center, right_eye_center = self.get_eyes_center(landmarks)
        mouth_center = self.get_mouth_center(landmarks)
        left_eye_bbox = torch.cat(
            [left_eye_center-eye_out_size//2, left_eye_center+eye_out_size//2], dim=1)
        right_eye_bbox = torch.cat(
            [right_eye_center-eye_out_size//2, right_eye_center+eye_out_size//2], dim=1)
        mouth_bbox = torch.cat(
            [mouth_center-mouth_out_size//2, mouth_center+mouth_out_size//2], dim=1)

        for b in range(images.size(0)):
            # left eye and right eye
            img_inds = torch.full(
                (2, 1), b, dtype=torch.float32, device=self.device)
            bbox = torch.stack(
                [left_eye_bbox[b, :], right_eye_bbox[b, :]], dim=0)  # shape: (2, 4)
            rois = torch.cat([img_inds, bbox], dim=1)  # shape: (2, 5)
            rois_eyes.append(rois)
            # mouse
            img_inds = torch.full(
                (1, 1), b, dtype=torch.float32, device=self.device)
            # shape: (1, 5)
            rois = torch.cat([img_inds, mouth_bbox[b:b + 1, :]], dim=1)
            rois_mouths.append(rois)

        rois_eyes = torch.cat(rois_eyes, 0).to(self.device)
        rois_mouths = torch.cat(rois_mouths, 0).to(self.device)

        all_eyes = roi_align(images, boxes=rois_eyes, output_size=eye_out_size)
        left_eyes = all_eyes[0::2, :, :, :]
        right_eyes = all_eyes[1::2, :, :, :]
        mouths = roi_align(images, boxes=rois_mouths,
                           output_size=mouth_out_size)

        return left_eyes, right_eyes, mouths
    
    @staticmethod
    def get_gaussian_filter(kernel_size=15, sigma=7, channels=3):
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    
        mean = (kernel_size - 1) / 2
        variance = sigma**2
    
        gaussian_kernel = (1 / (2 * math.pi * variance)) * torch.exp(-torch.sum((xy_grid - mean) ** 2, dim=-1) / (2 * variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    
        gaussian_filter = nn.Conv2d(channels, channels, kernel_size, groups=channels, bias=False, padding=kernel_size//2)
        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False
    
        return gaussian_filter
    
    def blur_mouth(self, images, landmarks):
        pts = landmarks[:, 14:]
        mouth_center = torch.median(pts, dim=1).values
        mouth_out_size = torch.tensor([80, 60]).to(self.device).unsqueeze(0)
        bboxes = torch.cat([mouth_center-mouth_out_size//2, mouth_center+mouth_out_size//2], dim=1).long()
        margin = 15
        
        blured_mouth = []
        blured_imgs = torch.zeros_like(images).requires_grad_(False)
        masks = blured_imgs.clone()
        for i, bbox in enumerate(bboxes):
            blured_mouth.append(self.gaussian_filter(images[i:i+1, :, bbox[1]-margin:bbox[3]+margin, bbox[0]-margin:bbox[2]+margin]))
            masks[i, :, bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
        blured_mouth = torch.cat(blured_mouth, dim=0)
        masks = masks.bool()
        blured_imgs[masks] = blured_mouth[:, :, margin:-margin, margin:-margin].reshape(-1)
        blured_imgs[~masks] = images[~masks]

        return blured_imgs

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)
        if self.opts.use_wandb:
            self.wb_logger.log(prefix, metrics_dict, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print(f'Metrics for {prefix}, step {self.global_step}')
        for key, value in metrics_dict.items():
            print(f'\t{key} = ', value)

    def parse_and_log_images(self, id_logs, x, D1, D2, T1, SD1, SD2, title, subscript=None, display_count=2):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'source': common.log_input_image(x[i], self.opts),
                'driving1': common.log_input_image(D1[i], self.opts),
                'driving2': common.log_input_image(D2[i], self.opts),
                'target1': common.log_input_image(T1[i], self.opts),
                'output1': common.tensor2im(SD1[i]),
                'output2': common.tensor2im(SD2[i]),
            }
            if id_logs is not None:
                for key in id_logs[i]:
                    cur_im_data[key] = id_logs[i][key]
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, subscript=subscript)

    def parse_and_log_facial_components(self, le_t, re_t, m_t, le_p, re_p, m_p, title, subscript=None, display_count=2):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'left_eye_gt': common.tensor2im(le_t[i]),
                'right_eye_gt': common.tensor2im(re_t[i]),
                'mouth_gt': common.tensor2im(m_t[i]),
                'left_eye_pred': common.tensor2im(le_p[i]),
                'right_eye_pred': common.tensor2im(re_p[i]),
                'mouth_pred': common.tensor2im(m_p[i]),
            }
            im_data.append(cur_im_data)

        fig = common.vis_facial_component(im_data)
        step = self.global_step
        if subscript:
            path = os.path.join(self.logger.log_dir, title,
                                f'{subscript}_{step:04d}.jpg')
        else:
            path = os.path.join(self.logger.log_dir, title, f'{step:04d}.jpg')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def log_images(self, name, im_data, subscript=None, log_latest=False):
        fig = common.vis_faces(im_data)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.logger.log_dir, name,
                                f'{subscript}_{step:04d}.jpg')
        else:
            path = os.path.join(self.logger.log_dir, name, f'{step:04d}.jpg')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.net.state_dict(),
            'latent_pool': self.latent_pool.pool,
            'opts': vars(self.opts)
        }
        if self.opts.adv_lambda > 0:  # modified
            save_dict['discriminator'] = self.discriminator.state_dict()
            save_dict['left_eye_disc'] = self.left_eye_disc.state_dict()
            save_dict['right_eye_disc'] = self.right_eye_disc.state_dict()
            save_dict['mouth_disc'] = self.mouth_disc.state_dict()
        if self.opts.editing_w_path is not None:
            save_dict['editing_w'] = self.editing_w.cpu()
        # save the latent avg in state_dict for inference if truncation of w was used during training
        if self.opts.start_from_latent_avg:
            save_dict['latent_avg'] = self.net.latent_avg
        return save_dict

    # modified
    @staticmethod
    def discriminator_loss(real_pred, fake_pred, loss_dict=None, lambd=1):
        real_loss = F.softplus(-real_pred).mean() * lambd
        fake_loss = F.softplus(fake_pred).mean() * lambd

        if loss_dict is not None:
            loss_dict['loss_d_real'] = float(real_loss)
            loss_dict['loss_d_fake'] = float(fake_loss)

        return real_loss + fake_loss

    @staticmethod
    def discriminator_r1_loss(real_pred, real_w):
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_w, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(
            grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

    @staticmethod
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def train_discriminator(self, real_img, fake_img, facial_component=None):
        loss_dict = {}
        self.requires_grad(self.discriminator, True)

        real_pred = self.discriminator(real_img)
        fake_pred = self.discriminator(fake_img.detach())
        loss = self.discriminator_loss(
            real_pred, fake_pred, loss_dict, self.opts.adv_lambda)
        loss_dict['loss_d'] = float(loss)

        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()

        # r1 regularization
        d_regularize = self.global_step % self.opts.d_reg_every == 0
        if d_regularize:
            real_img = real_img.detach()
            real_img.requires_grad = True
            real_pred = self.discriminator(real_img)
            r1_loss = self.discriminator_r1_loss(real_pred, real_img)

            self.discriminator.zero_grad()
            r1_final_loss = self.opts.r1 / 2 * r1_loss * \
                self.opts.d_reg_every + 0 * real_pred[0]
            r1_final_loss.backward()
            self.discriminator_optimizer.step()
            loss_dict['loss_r1'] = float(r1_final_loss)

        # Reset to previous state
        self.requires_grad(self.discriminator, False)

        if facial_component is not None:  # (SD1, SD2)
            left_eye_gt = facial_component[0][0]
            right_eye_gt = facial_component[0][1]
            mouth_gt = facial_component[0][2]
            left_eye_pred = facial_component[1][0]
            right_eye_pred = facial_component[1][1]
            mouth_pred = facial_component[1][2]

            self.requires_grad(self.left_eye_disc, True)
            self.requires_grad(self.right_eye_disc, True)
            self.requires_grad(self.mouth_disc, True)

            left_eye_real = self.left_eye_disc(left_eye_gt)
            left_eye_fake = self.left_eye_disc(left_eye_pred.detach())
            right_eye_real = self.right_eye_disc(right_eye_gt)
            right_eye_fake = self.right_eye_disc(right_eye_pred.detach())
            mouth_real = self.mouth_disc(mouth_gt)
            mouth_fake = self.mouth_disc(mouth_pred.detach())

            loss_d_left_eye = self.discriminator_loss(
                left_eye_real, left_eye_fake, lambd=self.opts.adv_lambda)
            loss_d_right_eye = self.discriminator_loss(
                right_eye_real, right_eye_fake, lambd=self.opts.adv_lambda)
            loss_d_mouth = self.discriminator_loss(
                mouth_real, mouth_fake, lambd=self.opts.adv_lambda)

            loss_dict['loss_d_left_eye'] = float(loss_d_left_eye)
            loss_dict['loss_d_right_eye'] = float(loss_d_right_eye)
            loss_dict['loss_d_mouth'] = float(loss_d_mouth)

            self.left_eye_disc_optimizer.zero_grad()
            self.right_eye_disc_optimizer.zero_grad()
            self.mouth_disc_optimizer.zero_grad()
            loss_d_left_eye.backward()
            loss_d_right_eye.backward()
            loss_d_mouth.backward()
            self.left_eye_disc_optimizer.step()
            self.right_eye_disc_optimizer.step()
            self.mouth_disc_optimizer.step()

            self.requires_grad(self.left_eye_disc, False)
            self.requires_grad(self.right_eye_disc, False)
            self.requires_grad(self.mouth_disc, False)

        return loss_dict

    @torch.no_grad()
    def validate_discriminator(self, real_img, fake_img, facial_component=None):
        loss_dict = {}
        real_pred = self.discriminator(real_img)
        fake_pred = self.discriminator(fake_img.detach())
        loss = self.discriminator_loss(
            real_pred, fake_pred, loss_dict, self.opts.adv_lambda)
        loss_dict['loss_d'] = float(loss)

        if facial_component is not None:
            left_eye_gt = facial_component[0][0]
            right_eye_gt = facial_component[0][1]
            mouth_gt = facial_component[0][2]
            left_eye_pred = facial_component[1][0]
            right_eye_pred = facial_component[1][1]
            mouth_pred = facial_component[1][2]

            left_eye_real = self.left_eye_disc(left_eye_gt)
            left_eye_fake = self.left_eye_disc(left_eye_pred.detach())
            right_eye_real = self.right_eye_disc(right_eye_gt)
            right_eye_fake = self.right_eye_disc(right_eye_pred.detach())
            mouth_real = self.mouth_disc(mouth_gt)
            mouth_fake = self.mouth_disc(mouth_pred.detach())

            loss_d_left_eye = self.discriminator_loss(
                left_eye_real, left_eye_fake, lambd=self.opts.adv_lambda)
            loss_d_right_eye = self.discriminator_loss(
                right_eye_real, right_eye_fake, lambd=self.opts.adv_lambda)
            loss_d_mouth = self.discriminator_loss(
                mouth_real, mouth_fake, lambd=self.opts.adv_lambda)

            loss_dict['loss_d_left_eye'] = float(loss_d_left_eye)
            loss_dict['loss_d_right_eye'] = float(loss_d_right_eye)
            loss_dict['loss_d_mouth'] = float(loss_d_mouth)

        return loss_dict
