import torch
from .base_model import BaseModel
from . import networks
from copy import deepcopy
from models import create_model


class Pix2PixBrainModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_L2', type=float, default=0.5, help='weight for tumour tissue over rest of brain. Range [0,1]')
            parser.add_argument('--TPN', action='store_true', help='Use the Time Prediction Network (TPN) in the loss')
            parser.add_argument('--gamma', type=float, default=100.0, help='weight for time loss, when TPN is set to True')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # False during testing or if training andf not enabled
        self.TPN_enabled = False

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc + 1, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            if self.opt.TPN:
                self.TPN_enabled = True
                self.loss_names = ['G_GAN', 'G_L1', 'G_TPN', 'D_real', 'D_fake']

                # Store final gamma value and then set it to 0
                self.final_gamma = deepcopy(opt.gamma)
                opt.gamma = 0

                # Initiliaze m and c and None
                self.update_m = None
                self.update_c = None

                # Setup TPN if set to True
                print("\nSetting up TPN\n")
                opt_TPN = deepcopy(opt) # copy train options and change later
                opt_TPN.model = 'time_predictor'
                opt_TPN.name = 'time_prediction_10'
                opt_TPN.netD = 'time_input'
                opt_TPN.ndf = 16
                # hard-code some parameters for test
                opt_TPN.display_id = -1   # no visdom display;
                opt_TPN.isTrain = False
                print("Options TPN: {}\n\n".format(opt_TPN))
                self.TPN = create_model(opt_TPN)      # create a model given opt_TPN.model and other options
                self.TPN.setup(opt_TPN)               # regular setup: load

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # Check if lambda_L2 is in range [0,1]
            assert (0 <= self.opt.lambda_L2 <= 1)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.true_time = input['time_period'][0]
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.TPN_enabled:
            self.img_time_tensor = torch.ones(self.real_A.shape) * self.true_time
            img_with_time = torch.cat((self.img_time_tensor, self.real_A), 1) 
            self.fake_B = self.netG(img_with_time)
            # Predict the time between real image A and generated image B
            self.TPN.real_A = self.real_A
            self.TPN.real_B = self.fake_B
            self.TPN.forward()
            self.fake_time = self.TPN.prediction
        else:
            self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        ### ORIGINAL
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        ### END ORIGINAL

        # Weighted L1 Loss
        # fake_B_tumour = self.fake_B.clone().detach()
        # real_B_tumour = self.real_B.clone().detach()
        # fake_B_tumour[fake_B_tumour < 0.5] = 0
        # real_B_tumour[fake_B_tumour < 0.5] = 0
        # self.loss_G_L1 = self.opt.lambda_L1 * (self.criterionL1(self.fake_B, self.real_B) * (1 - self.opt.lambda_L2) + \
        #                  self.criterionL1(fake_B_tumour, real_B_tumour) * self.opt.lambda_L2)


        # TPN Loss
        if self.TPN_enabled:
            true_time_tensor = torch.ones(self.fake_time.shape) * self.true_time
            self.loss_G_TPN = self.criterionL1(true_time_tensor, self.fake_time.cpu()) * self.opt.gamma
            # combine loss and calculate gradients
            self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_TPN.to(self.device)
        else:
            # combine loss and calculate gradients
            self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate gradients for G
        self.optimizer_G.step()             # udpate G's weights

    def update_current_gamma(self, epoch):
        ''' Update gamma value for TPN from opt, depending on the epoch '''
        start_epoch = 100
        end_epoch = 150

        # Values should be None only at the first call
        if self.update_m == None and self.update_c == None:
            self.update_m = self.final_gamma / (end_epoch - start_epoch)
            self.update_c = -self.update_m * start_epoch

        if epoch < start_epoch:
            self.opt.gamma = 0
        elif start_epoch < epoch < end_epoch:
            # Linearly update gamma
            self.opt.gamma = self.update_m * epoch + self.update_c
        else: # epoch > end_epoch
            self.opt_gamma = self.final_gamma

        print('gamma = %.7f' % self.opt.gamma)
