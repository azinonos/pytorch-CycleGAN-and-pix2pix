import torch
from .base_model import BaseModel
from . import networks
from copy import deepcopy
from models import create_model
import numpy as np


class TimePredictorModel(BaseModel):
    """ This class implements the time_predictor model, for learning the time difference between two given MRI images.

    The model training requires '--dataset_mode brain' dataset.
    By default, it uses a '--netD time_input' discriminator,

    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The training objective is: M(True_Time, Predicted_Time), where M is a Distance metric
        By default, we use UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', dataset_mode='aligned', netD='time_input')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the time_predictor class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_real']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A','real_B', 'diff_map']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['D']
        # define network
        # opt.netD can be ['time_input', 'time_diffmap', time_hist', 'time_autoenc']
        self.Dtype = opt.netD
        input_channel_size = opt.input_nc + opt.output_nc

        # These type of discriminators handle 1D data, so change
        # the corresponding sizes
        if opt.netD == 'time_diffmap':
            input_channel_size = opt.input_nc
        if opt.netD == 'time_hist' or opt.netD == 'time_autoenc':
            opt.norm = 'batch_1d'
            input_channel_size = opt.input_nc

        if opt.netD == 'time_autoenc':
            # Setup Autoencoder if given in arguments
            print("\nSetting up AutoEncoder\n")
            opt_autoenc = deepcopy(opt) # copy train options and change later
            opt_autoenc.model = 'auto_encoder'
            opt_autoenc.name = 'autoenc_02' # change to trained autoencoder model name
            opt_autoenc.netD = 'autoenc'
            opt_autoenc.norm = 'batch'
            # hard-code some parameters for test
            opt_autoenc.num_threads = 0   # test code only supports num_threads = 1
            opt_autoenc.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
            opt_autoenc.isTrain = False
            print("Options Autoenc: {}\n\n".format(opt_autoenc))
            self.autoencoder = create_model(opt_autoenc)      # create a model given opt_autoenc.model and other options
            self.autoencoder.setup(opt_autoenc)               # regular setup: load

        self.netD = networks.define_D(input_channel_size, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionL2 = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.diff_map = input['diff_map'].to(self.device)
        self.hist_diff = input['hist_diff'].float().to(self.device)
        self.true_time = input['time_period'][0]
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.Dtype == 'time_input':
            real_AB = torch.cat((self.real_A, self.real_B), 1) # we need to feed both input and output to the network
            self.prediction = self.netD(real_AB)
        elif self.Dtype == 'time_diffmap':
            self.prediction = self.netD(self.diff_map)
        elif self.Dtype == 'time_hist':
            self.prediction = self.netD(self.hist_diff)
        elif self.Dtype == 'time_autoenc':
            self.autoencoder.diff_map = self.diff_map # Bypass Input and just store the diff map in the object
            latent_vector = self.autoencoder.forward_getVector() 
            self.prediction = self.netD(latent_vector[np.newaxis, :])

    def backward_D(self):
        # Calculate Loss for D
        # Store scalar in a 1x1 matrix so that we can use it for the loss
        true_time_tensor = torch.ones(self.prediction.shape) * self.true_time
        self.loss_D_real = self.criterionL2(true_time_tensor, self.prediction.cpu())
        self.loss_D = self.loss_D_real
        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
