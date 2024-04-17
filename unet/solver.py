import csv
import os
from os.path import join

import torch
import torch.nn.functional as F
import torchvision
from termcolor import colored
from torch import distributed
from torch import optim
from torch.nn import BCELoss
from tqdm import tqdm

from unet.evaluator import BinaryFilterEvaluator
from unet.logger import LoggerScalar
from unet.loss import DCAndBCELoss, DCAndCELoss
from unet.loss.soft_dice import MemoryEfficientSoftDiceLoss
from unet.network import get_network, get_cached_pretrained_model


class Solver(object):
    def __init__(self, config, train_loader, valid_loader):
        self.config = config
        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader

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

        self.save_interval = config.save_interval
        self.current_epoch = self.get_cached_pretrained_epoch()
        self.build_model()

    def get_cache_model_path(self):
        joined_path = os.path.join(self.config.cache_base_path, get_cached_pretrained_model(self.config))
        if not os.path.exists(joined_path):
            os.makedirs(joined_path)
        return joined_path

    def get_cached_pretrained_epoch(self):
        if not self.config.need_record:
            return 0

        joined_path = self.get_cache_model_path()

        # 获取文件夹列表
        folders = os.listdir(joined_path)
        # 过滤出仅包含数字名称的文件夹
        numeric_folders = [folder for folder in folders if folder.isdigit()]

        if numeric_folders:
            # 将文件夹名称转换为整数并找到最大的一个
            max_folder = max(map(int, numeric_folders))
        else:
            max_folder = 0  # 如果不存在数字文件夹，返回0
        return max_folder

    def build_model(self, load_current_epoch=True):
        """Build generator and discriminator."""
        self.unet = get_network(
            config=self.config.network,
            dataset_config=self.config.dataset,
            device=self.device,
            load_pretrained_model=self.current_epoch == 0,
        )

        self.optimizer = optim.Adam(list(self.unet.parameters()),
                                    self.lr, (self.beta1, self.beta2))

        if self.current_epoch == 0:
            print("No cached model found.")
            return

        if not load_current_epoch:
            return

        cache_path = self.get_cache_model_path()
        network_path = os.path.join(cache_path, str(self.current_epoch), 'network.pth')
        optimizer_path = os.path.join(cache_path, str(self.current_epoch), 'optimizer.pth')
        print(colored('Loaded network param.', "light_green",
                      attrs=["bold"]))
        self.unet.load_state_dict(torch.load(network_path))
        self.optimizer.load_state_dict(torch.load(optimizer_path))

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

    def tensor2img(self, x):
        img = (x[:, 0, :, :] > x[:, 1, :, :]).float()
        img = img * 255
        return img

    def train(self):
        """Train encoder, generator and discriminator."""

        # ====================================== Training ===========================================#
        # ===========================================================================================#
        cache_path = self.get_cache_model_path()
        best_network_path = os.path.join(cache_path, 'best_network')

        # Train for Encoder
        lr = self.lr
        best_unet_score = 0.

        if self.current_epoch != 0:
            with open(join(best_network_path, 'best_data.txt'), 'r') as f:
                text = f.read().split(',')
                best_unet_score = float(text[1])

        for epoch in range(self.current_epoch, self.num_epochs):
            evaluator = BinaryFilterEvaluator(epoch=epoch, total_epoch=self.num_epochs, type='train')
            self.unet.train(True)
            for i, (images, GT, _) in enumerate(
                    tqdm(
                        iterable=self.train_loader,
                        desc=f"{self.model_type} Epoch {epoch} Training Processing"
                    )
            ):
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
                best_unet = self.unet.state_dict()
                best_unet_optimizer = self.optimizer.state_dict()
                print(colored('Best %s model score : %.4f' % (self.model_type, best_unet_score), "light_green",
                              attrs=["bold"]))
                if not os.path.exists(best_network_path):
                    os.makedirs(best_network_path)
                torch.save(best_unet, join(best_network_path, 'network.pth'))
                torch.save(best_unet_optimizer, join(best_network_path, 'optimizer.pth'))
                # save best epoch
                with open(join(best_network_path, 'best_data.txt'), 'w') as f:
                    f.write(f'{epoch},{best_unet_score}')

            if epoch % self.save_interval == self.save_interval - 1:
                if not os.path.exists(join(cache_path, str(epoch))):
                    os.makedirs(join(cache_path, str(epoch)))
                torch.save(self.unet.state_dict(), join(cache_path, str(epoch), 'network.pth'))
                torch.save(self.optimizer.state_dict(), join(cache_path, str(epoch), 'optimizer.pth'))

        # ===================================== Test ====================================#
        self.test()

    def test(self):
        cache_path = self.get_cache_model_path()
        best_network_path = os.path.join(cache_path, 'best_network')
        with open(join(best_network_path, 'best_data.txt'), 'r') as f:
            text = f.read().split(',')
            best_epoch = int(text[0])
        del self.unet
        self.build_model(load_current_epoch=False)
        self.unet.load_state_dict(torch.load(join(best_network_path, 'network.pth')))
        valid_evaluator = self._valid_(False)
        result_path = os.path.join(self.result_path, 'result.csv')

        first_create = os.path.exists(result_path)

        with open(os.path.join(self.result_path, 'result.csv'), 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            if not first_create:
                wr.writerow([
                    'Model',
                    'Miou', 'F1_score', 'Accuracy', 'Specificity',
                    'Sensitivity', 'DSC', 'AP', 'AUC',
                    'Jaccard Similarity', 'Precision',
                    'LR', 'Best Epoch', 'Total Epoch',
                    'Decay Epoch', 'Augmentation Prob'
                ])

            wr.writerow([
                get_cached_pretrained_model(self.config),
                valid_evaluator.MIOU * 100, valid_evaluator.F1 * 100, valid_evaluator.acc * 100, valid_evaluator.SP * 100,
                valid_evaluator.SE * 100, valid_evaluator.DC * 100, valid_evaluator.AP * 100, valid_evaluator.AUC * 100,
                valid_evaluator.JS * 100, valid_evaluator.PC * 100,
                self.lr, best_epoch, self.num_epochs,
                self.num_epochs_decay, self.augmentation_prob]
            )
            f.close()

    def _valid_(self, isValid=True, epoch=None):
        valid_evaluator = BinaryFilterEvaluator(
            epoch=epoch,
            total_epoch=self.num_epochs,
            type='valid',
            threshold=self.config.threshold
        )
        self.unet.train(False)
        self.unet.eval()

        for i, (images, GT, origin_image_name) in enumerate(
                tqdm(
                    iterable=self.valid_loader,
                    desc=f"{self.model_type} Epoch {epoch} Validation Processing"
                )
        ):
            images = images.to(self.device)
            GT = GT.to(self.device)
            SR = F.sigmoid(self.unet(images))
            valid_evaluator.evaluate(SR, GT, images.size(0), 0)
            if isValid:
                # 将SR输出的概率值转换为二值化的图像
                SR = SR > valid_evaluator.threshold
                SR = SR * 255
                # 写入图像
                torchvision.utils.save_image(
                    SR.data.cpu(),
                    os.path.join(
                        self.result_path,
                        f'{origin_image_name}.png'
                    )
                )

        valid_evaluator.calculate()
        unet_score = valid_evaluator.JS + valid_evaluator.DC

        if not isValid:
            return valid_evaluator
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
