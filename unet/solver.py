import csv
import os

import torch.nn.functional as F
from torch import distributed
from torch import optim
from torch.nn import BCELoss
from tqdm.auto import tqdm

from unet.evaluation import *
from unet.evaluator import BinaryFilterEvaluator
from unet.logger import LoggerScalar
from unet.loss import DCAndBCELoss, DCAndCELoss
from unet.loss.soft_dice import MemoryEfficientSoftDiceLoss
from unet.network import get_network


class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):
        self.config = config
        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.network.img_ch
        self.output_ch = config.network.output_ch
        self.image_size = config.dataset.image_size
        self.is_ddp = distributed.is_available() and distributed.is_initialized()
        if config.has_multiple_label:
            self.criterion = DCAndBCELoss(
                bce_kwargs={},
                soft_dice_kwargs={'batch_dice': config.dataset.batch_dice,
                                  'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
            )
        else:
            # self.criterion = DCAndCELoss(
            #     soft_dice_kwargs={'batch_dice': config.dataset.batch_dice,
            #                       'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            #     ce_kwargs={},
            #     weight_ce=1, weight_dice=1,
            #     ignore_label=config.ignore_label,
            #     dice_class=MemoryEfficientSoftDiceLoss
            # )
            self.criterion = BCELoss()

        print("Loss class:", str(self.criterion.__class__.__name__))
        self.augmentation_prob = config.dataset.augmentation_prob

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.dataset.batch_size

        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step

        # Path
        self.model_path = config.network.model_path
        self.tensorboard_path = config.tensorboard_path
        self.result_path = config.result_path
        self.mode = config.mode

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.network.model_type
        self.log = LoggerScalar(os.path.join(self.tensorboard_path, self.model_type))
        self.build_model()

    def build_model(self):
        """Build generator and discriminator."""
        self.unet = get_network(self.config.network)

        self.optimizer = optim.Adam(list(self.unet.parameters()),
                                    self.lr, (self.beta1, self.beta2))
        self.unet.to(self.device)

    # self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

    def compute_accuracy(self, SR, GT):
        SR_flat = SR.view(-1)
        GT_flat = GT.view(-1)

        acc = GT_flat.data.cpu() == (SR_flat.data.cpu() > 0.5)

    def tensor2img(self, x):
        img = (x[:, 0, :, :] > x[:, 1, :, :]).float()
        img = img * 255
        return img

    def train(self):
        """Train encoder, generator and discriminator."""

        # ====================================== Training ===========================================#
        # ===========================================================================================#

        unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' % (
            self.model_type, self.num_epochs, self.lr, self.num_epochs_decay, self.augmentation_prob))

        # U-Net Train
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
            return

        # Train for Encoder
        lr = self.lr
        best_unet_score = 0.
        best_epoch = 0
        for epoch in range(self.num_epochs):
            evaluator = BinaryFilterEvaluator(epoch=epoch, total_epoch=self.num_epochs, type='train')
            self.unet.train(True)
            for i, (images, GT) in enumerate(
                    tqdm.tqdm(self.train_loader, desc=f"{self.model_type} Epoch {epoch} Training Processing")):
                # GT : Ground Truth

                images = images.to(self.device)
                GT = GT.to(self.device)

                # SR : Segmentation Result
                SR = self.unet(images)

                SR_probs = F.sigmoid(SR)

                SR_flat = SR_probs.view(SR_probs.size(0), -1)

                GT_flat = GT.view(GT.size(0), -1)
                loss = self.criterion(SR_flat, GT_flat)
                current_loss = loss.item()

                # Backprop + optimize
                self.reset_grad()
                loss.backward()
                self.optimizer.step()

                evaluator.evaluate(SR_probs, GT, images.size(0), current_loss)

            evaluator.calculate()

            # Print the log info
            print(evaluator.to_log())
            my_fantastic_logging = evaluator.to_tensorboard()

            # Decay learning rate
            if (epoch + 1) > (self.num_epochs - self.num_epochs_decay):
                lr -= (self.lr / float(self.num_epochs_decay))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                print('Decay learning rate to lr: {}.'.format(lr))
                my_fantastic_logging['lr'] = lr
            self.log.plot_data(my_fantastic_logging=my_fantastic_logging)
            # ===================================== Validation ====================================#

            unet_score = self._valid_(True, epoch=epoch)
            # Save Best U-Net model
            if unet_score > best_unet_score:
                best_unet_score = unet_score
                best_epoch = epoch
                best_unet = self.unet.state_dict()
                print('Best %s model score : %.4f' % (self.model_type, best_unet_score))
                torch.save(best_unet, unet_path)

        # ===================================== Test ====================================#
        del self.unet
        del best_unet
        self.build_model()
        self.unet.load_state_dict(torch.load(unet_path))
        acc, SE, SP, PC, F1, JS, DC = self._valid_(False)
        with open(os.path.join(self.result_path, 'result.csv'), 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            wr.writerow([self.model_type, acc, SE, SP, PC, F1, JS, DC, self.lr, best_epoch, self.num_epochs,
                         self.num_epochs_decay, self.augmentation_prob])
            f.close()

    def _valid_(self, isValid=True, epoch=None):
        valid_evaluator = BinaryFilterEvaluator(epoch=epoch, total_epoch=self.num_epochs, type='valid')
        self.unet.train(False)
        self.unet.eval()

        for i, (images, GT) in enumerate(self.valid_loader):
            images = images.to(self.device)
            GT = GT.to(self.device)
            SR = F.sigmoid(self.unet(images))
            valid_evaluator.evaluate(SR, GT, images.size(0), 0)

        valid_evaluator.calculate()
        unet_score = valid_evaluator.JS + valid_evaluator.DC

        if not isValid:
            return valid_evaluator.acc, valid_evaluator.SE, valid_evaluator.SP, valid_evaluator.PC, valid_evaluator.F1, valid_evaluator.JS, valid_evaluator.DC

        print(valid_evaluator.to_log())
        self.log.plot_data(my_fantastic_logging=valid_evaluator.to_tensorboard())
        '''
        torchvision.utils.save_image(images.data.cpu(),
                                    os.path.join(self.result_path,
                                                '%s_valid_%d_image.png'%(self.model_type,epoch+1)))
        torchvision.utils.save_image(SR.data.cpu(),
                                    os.path.join(self.result_path,
                                                '%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
        torchvision.utils.save_image(GT.data.cpu(),
                                    os.path.join(self.result_path,
                                                '%s_valid_%d_GT.png'%(self.model_type,epoch+1)))
        '''
        return unet_score
