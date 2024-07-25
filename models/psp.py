"""
This file defines the core research contribution
"""
from models.transformer import CrossAttention, Transformer
import torch.nn.functional as F
from models.stylegan2.model import Generator
from models.encoders import psp_encoders
from basicsr.archs.stylegan2_arch import ConvLayer
from torch import nn
import torch
import math
import matplotlib
matplotlib.use('Agg')


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items()
              if k[:len(name)] == name}
    return d_filt


class pSp(nn.Module):

    def __init__(self, opts, ckpt=None):
        super(pSp, self).__init__()
        self.set_opts(opts)
        # compute number of style inputs based on the output resolution
        self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
        # Define architecture
        self.encoder = self.set_encoder()
        self.decoder = Generator(self.opts.output_size, 512, 8)
        self.cross_attention = CrossAttention(
            d_model=512, nhead=4, dim_feedforward=1024)
        self.transformer = Transformer(
            dim=512, depth=2, heads=8, dim_head=64, mlp_dim=1, dropout=0)  # 64,1
        # Load weights if needed
        self.load_weights(ckpt, ckpt is not None)
        self.encoder_fixed = False

    def set_encoder(self):
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(
                50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(
                50, 'ir_se', self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(
                self.opts.encoder_type))
        return encoder

    def load_weights(self, ckpt=None, strict=False):
        if self.opts.checkpoint_path is not None:
            print('Loading pSp from checkpoint: {}'.format(
                self.opts.checkpoint_path))
            if ckpt is None:
                ckpt = torch.load(self.opts.checkpoint_path,
                                  map_location='cpu')
            self.encoder.load_state_dict(
                get_keys(ckpt, 'encoder'), strict=strict)
            self.decoder.load_state_dict(
                get_keys(ckpt, 'decoder'), strict=strict)
            self.__load_latent_avg(ckpt)
            if len(get_keys(ckpt, 'cross_attention').keys()) > 0:
                self.cross_attention.load_state_dict(
                    get_keys(ckpt, 'cross_attention'), strict=strict)
                self.transformer.load_state_dict(
                    get_keys(ckpt, 'transformer'), strict=strict)
        else:
            print('Loading encoders weights from irse50!')
            encoder_ckpt = torch.load(self.opts.ir_se50_weights)
            # if input to encoder is not an RGB image, do not load the input layer weights
            if self.opts.label_nc != 0:
                encoder_ckpt = {
                    k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
            self.encoder.load_state_dict(encoder_ckpt, strict=False)
            print('Loading decoder weights from pretrained!')
            ckpt = torch.load(self.opts.stylegan_weights)
            self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
            if self.opts.learn_in_w:
                self.__load_latent_avg(ckpt, repeat=1)
            else:
                self.__load_latent_avg(ckpt, repeat=self.opts.n_styles)

    def encode(self, x, use_skip=True, use_feature=True):
        feats = None  # f and the skipped encoder features
        codes, feats, P_S = self.encoder(
            x, return_feat=True, return_full=use_skip)  # modified
        # normalize with respect to the center of an average face
        if self.opts.start_from_latent_avg:
            if self.opts.learn_in_w:
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
            else:
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        first_layer_feats, skip_layer_feats, fusion = None, None, None  # modified
        if use_feature:  # modified
            first_layer_feats = feats[0:2]  # use f
            if use_skip:  # modified
                skip_layer_feats = feats[2:]  # use skipped encoder feature
                # use fusion layer to fuse encoder feature and decoder feature.
                fusion = self.encoder.fusion

        return codes, feats, first_layer_feats, skip_layer_feats, fusion

    def get_emotion_latents(self, codes, D, use_skip=True):  # modified
        codes_D, _, P_D = self.encoder(
            D, return_feat=True, return_full=use_skip)
        delta_D = self.cross_attention(
            codes.permute(1, 0, 2), codes_D.permute(1, 0, 2))
        emotion_latents_D = self.transformer(delta_D.permute(1, 0, 2))

        if self.opts.start_from_latent_avg:
            if self.opts.learn_in_w:
                codes_D = codes_D + self.latent_avg.repeat(codes_D.shape[0], 1)
            else:
                codes_D = codes_D + \
                    self.latent_avg.repeat(codes_D.shape[0], 1, 1)
        return emotion_latents_D, codes_D

    def forward(self, x, D1=None, D2=None, resize=True, randomize_noise=True,
                return_latents=False, use_feature=True, zero_noise=False,
                first_layer_feature_ind=0, use_skip=False):  # modified

        codes, feats, first_layer_feats, skip_layer_feats, fusion = self.encode(
            x, use_skip, use_feature)

        images_SD1, images_SD2 = None, None
        codes_D1, codes_D2 = None, None
        emotion_latents_D1, emotion_latents_D2 = None, None

        if D1 is not None:
            emotion_latents_D1, codes_D1 = self.get_emotion_latents(codes, D1, use_skip)

            images_SD1, result_latent_SD1 = self.decoder([codes],
                                                         input_is_latent=True,
                                                         randomize_noise=randomize_noise,
                                                         return_latents=return_latents,
                                                         first_layer_feature=first_layer_feats,
                                                         first_layer_feature_ind=first_layer_feature_ind,
                                                         skip_layer_feature=skip_layer_feats,
                                                         fusion_block=fusion,
                                                         zero_noise=zero_noise,
                                                         editing_w=emotion_latents_D1)
            if resize:
                images_SD1 = F.interpolate(
                    images_SD1, (256, 256), mode='bilinear')

        if D2 is not None:
            emotion_latents_D2, codes_D2 = self.get_emotion_latents(codes, D2, use_skip)

            images_SD2, result_latent_SD2 = self.decoder([codes],
                                                         input_is_latent=True,
                                                         randomize_noise=randomize_noise,
                                                         return_latents=return_latents,
                                                         first_layer_feature=first_layer_feats,
                                                         first_layer_feature_ind=first_layer_feature_ind,
                                                         skip_layer_feature=skip_layer_feats,
                                                         fusion_block=fusion,
                                                         zero_noise=zero_noise,
                                                         editing_w=emotion_latents_D2)

            if resize:
                images_SD2 = F.interpolate(
                    images_SD2, (256, 256), mode='bilinear')

        if (not D1 is None) or (not D2 is None):
            images = (images_SD1, images_SD2)
            latents = (codes, codes_D1, codes_D2)
            emo_latents = (emotion_latents_D1, emotion_latents_D2)

            if return_latents:
                return images, latents, emo_latents
            else:
                return images

        else:
            images, result_latent = self.decoder([codes],
                                                 input_is_latent=True,
                                                 randomize_noise=randomize_noise,
                                                 return_latents=return_latents,
                                                 first_layer_feature=first_layer_feats,
                                                 first_layer_feature_ind=first_layer_feature_ind,
                                                 skip_layer_feature=skip_layer_feats,
                                                 fusion_block=fusion,
                                                 zero_noise=zero_noise,
                                                 editing_w=None)

            if resize:
                images = F.interpolate(images, (256, 256), mode='bilinear')
            latents = result_latent

            if return_latents:
                return images, latents
            else:
                return images

    def set_opts(self, opts):
        self.opts = opts

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].cuda()
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None

    def _fix_encoder(self):
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.fusion.train()
        for p in self.encoder.fusion.parameters():
            p.requires_grad = True

    def fix_encoder(self):
        self._fix_encoder()
        self.encoder_fixed = True

    def train(self, mode=True):
        super().train(mode)
        if mode and self.encoder_fixed:
            self._fix_encoder()
        return self


class FacialComponentDiscriminator(nn.Module):
    """Facial component (eyes, mouth, noise) discriminator used in GFPGAN.
    """

    def __init__(self):
        super(FacialComponentDiscriminator, self).__init__()
        # It now uses a VGG-style architectrue with fixed model size
        self.conv1 = ConvLayer(3, 64, 3, downsample=False, resample_kernel=(
            1, 3, 3, 1), bias=True, activate=True)
        self.conv2 = ConvLayer(64, 128, 3, downsample=True, resample_kernel=(
            1, 3, 3, 1), bias=True, activate=True)
        self.conv3 = ConvLayer(128, 128, 3, downsample=False, resample_kernel=(
            1, 3, 3, 1), bias=True, activate=True)
        self.conv4 = ConvLayer(128, 256, 3, downsample=True, resample_kernel=(
            1, 3, 3, 1), bias=True, activate=True)
        self.conv5 = ConvLayer(256, 256, 3, downsample=False, resample_kernel=(
            1, 3, 3, 1), bias=True, activate=True)
        self.final_conv = ConvLayer(256, 1, 3, bias=True, activate=False)

    def forward(self, x, return_feats=False, **kwargs):
        """Forward function for FacialComponentDiscriminator.

        Args:
            x (Tensor): Input images.
            return_feats (bool): Whether to return intermediate features. Default: False.
        """
        feat = self.conv1(x)
        feat = self.conv3(self.conv2(feat))
        rlt_feats = []
        if return_feats:
            rlt_feats.append(feat.clone())
        feat = self.conv5(self.conv4(feat))
        if return_feats:
            rlt_feats.append(feat.clone())
        out = self.final_conv(feat)

        if return_feats:
            return out, rlt_feats
        else:
            return out


if __name__ == '__main__':
    from options.train_options import TrainOptions

    opts = TrainOptions().parse()
    model = pSp(opts).cuda()

    x = torch.randn(2, 3, 256, 256, device='cuda')
    y1 = torch.randn(2, 3, 256, 256, device='cuda')
    y2 = torch.randn(2, 3, 256, 256, device='cuda')

    images, latents = model(x, y1, y2, return_latents=True)
    print(images[0].shape)
    print(latents[0].shape)
