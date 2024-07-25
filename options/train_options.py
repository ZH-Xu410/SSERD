from argparse import ArgumentParser


class TrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')
        self.parser.add_argument('--encoder_type', default='GradualStyleEncoder', type=str, help='Which encoder to use') 
        self.parser.add_argument('--input_nc', default=3, type=int, help='Number of input image channels to the psp encoder')
        self.parser.add_argument('--label_nc', default=0, type=int, help='Number of input label channels to the psp encoder')
        self.parser.add_argument('--output_size', default=1024, type=int, help='Output size of generator')
        self.parser.add_argument('--feat_ind', default=0, type=int, help='Layer index of G to accept the first-layer feature')
        self.parser.add_argument('--max_pooling', action="store_true", help='Apply max pooling or average pooling')
        self.parser.add_argument('--use_skip', action="store_true", help='Using skip connection from the encoder to the styleconv layers of G')
        self.parser.add_argument('--use_skip_torgb', action="store_true", help='Using skip connection from the encoder to the toRGB layers of G.')
        self.parser.add_argument('--skip_max_layer', default=7, type=int, help='Layer used for skip connection. 1,2,3,4,5,6,7 correspond to 4,8,16,32,64,128,256')
        self.parser.add_argument('--use_att', default=0, type=int, help='Layer of MLP used for attention, 0 not use attention')
        self.parser.add_argument('--zero_noise', action="store_true", help='Whether using zero noises')
 
        self.parser.add_argument('--data_root', default='data/MEAD', type=str, help='dataset root')
        self.parser.add_argument('--source_actors', default=['M003', 'M009', 'W029', 'M012', 'M030', 'W015'], nargs='+', help='source actors')
        self.parser.add_argument('--driving_actors', default=['M003', 'M009', 'W029', 'M012', 'M030', 'W015'], nargs='+', help='driving actors')
        self.parser.add_argument('--emotions', default=['neutral', 'angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised'], nargs='+', help='emotions')
        self.parser.add_argument('--loadSize', default=320, type=int, help='load image size')
        self.parser.add_argument('--cropSize', default=256, type=int, help='crop image size')

        self.parser.add_argument('--batch_size', default=2, type=int, help='Batch size for training')
        self.parser.add_argument('--test_batch_size', default=1, type=int, help='Batch size for testing and inference')
        self.parser.add_argument('--workers', default=2, type=int, help='Number of train dataloader workers')
        self.parser.add_argument('--test_workers', default=1, type=int, help='Number of test/inference dataloader workers')

        self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='Optimizer learning rate')
        self.parser.add_argument('--max_steps', default=50000, type=int, help='Maximum number of training steps')
        self.parser.add_argument('--lr_steps', default=[50000], type=int, nargs='+', help='lr decay steps')
        self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')
        self.parser.add_argument('--train_decoder', default=False, type=bool, help='Whether to train the decoder model')
        self.parser.add_argument('--start_from_latent_avg', action='store_true', help='Whether to add average latent vector to generate codes from encoder.')
        self.parser.add_argument('--learn_in_w', action='store_true', help='Whether to learn in w space instead of w+')

        self.parser.add_argument('--lpips_lambda', default=0.8, type=float, help='LPIPS loss multiplier factor')
        self.parser.add_argument('--id_lambda', default=0.2, type=float, help='ID loss multiplier factor')
        self.parser.add_argument('--l2_lambda', default=1.0, type=float, help='L2 loss multiplier factor')
        self.parser.add_argument('--w_norm_lambda', default=0, type=float, help='W-norm loss multiplier factor')
        self.parser.add_argument('--adv_lambda', default=0.1, type=float, help='Adversarial loss multiplier factor')
        self.parser.add_argument('--d_reg_every', default=16, type=int, help='Interval of the applying r1 regularization')
        self.parser.add_argument('--r1', default=1, type=float, help="weight of the r1 regularization")
        self.parser.add_argument('--emo_contrastive', default=0.1, type=float, help='emotion contrastive loss multiplier factor')
        self.parser.add_argument('--contrastive_start_step', default=100, type=int, help='when to start calcualte contrastive loss')
        self.parser.add_argument('--use_facial_disc', action="store_true", help='Using facial component discriminator')
        self.parser.add_argument('--facial_feat_lambda', default=0, type=float, help='facial component feature loss weight')
        
        self.parser.add_argument('--ir_se50_weights', default="pretrain_weights/model_ir_se50.pth", type=str, help='Path to ir_se50 model weights')
        self.parser.add_argument('--stylegan_weights', default=None, type=str, help='Path to StyleGAN model weights')
        self.parser.add_argument('--checkpoint_path', default="pretrain_weights/pretrain.pth", type=str, help='Path to model checkpoint')

        self.parser.add_argument('--image_interval', default=100, type=int, help='Interval for logging train images during training')
        self.parser.add_argument('--board_interval', default=50, type=int, help='Interval for logging metrics to tensorboard')
        self.parser.add_argument('--val_interval', default=10000, type=int, help='Validation interval')
        self.parser.add_argument('--save_interval', default=5000, type=int, help='Model checkpoint interval')
        self.parser.add_argument('--use_wandb', action="store_true", help='Whether to use Weights & Biases to track experiment.')



    def parse(self):
        opts = self.parser.parse_args()
        return opts
