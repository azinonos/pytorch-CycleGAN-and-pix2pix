import torch
from .base_model import BaseModel
from . import networks


class AutoEncoderModel(BaseModel):
    """ This class implements an autoencoder model, for learning the encoding/decoding of an image

    The model training requires '--dataset_mode aligned' dataset.

    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The training objective is: L1(Input, Output)
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', dataset_mode='aligned')
        return parser

    def __init__(self, opt):
        """Initialize the autoencoder class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['AE_real']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A','real_B', 'diff_map', 'recreated_diff_map']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['AE']
        # define network
        self.netAE = networks.define_D(opt.input_nc, opt.ndf, 'autoenc',
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_AE = torch.optim.Adam(self.netAE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_AE)

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
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.recreated_diff_map = self.netAE(self.diff_map)

    def forward_getVector(self):
        ''' Run forward pass but only on encoder, to return latent vector '''
        with torch.no_grad():
            latent_vector = self.netAE.forward_vectorOnly(self.diff_map)
        return latent_vector

    def backward_AE(self):
        # Calculate Loss for AE
        self.loss_AE_real = self.criterionL1(self.diff_map, self.recreated_diff_map)
        self.loss_AE = self.loss_AE_real
        self.loss_AE.backward()

    def optimize_parameters(self):
        self.forward()
        # update AE
        self.set_requires_grad(self.netAE, True)  # enable backprop for AE
        self.optimizer_AE.zero_grad()     # set AE's gradients to zero
        self.backward_AE()                # calculate gradients for AE
        self.optimizer_AE.step()          # update AE's weights
