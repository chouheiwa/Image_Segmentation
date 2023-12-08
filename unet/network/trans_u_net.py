import logging
from os.path import join

import numpy as np
import torch
from scipy import ndimage
from torch import nn
from torch.nn.functional import interpolate, grid_sample

from unet.network.module.transformer import Transformer, DecoderCup, SegmentationHead
from unet.network.network_type import NetworkType
from unet.utils import np2th

logger = logging.getLogger(__name__)


class TransUNet(NetworkType):

    @classmethod
    def create_model(
            cls,
            config,
            dataset_config,
            device,
            **kwargs
    ):
        transformer_config = config.transformer

        unet = TransUNet(
            config=transformer_config,
            img_size=dataset_config.processed_image.size,
            num_classes=config["num_classes"],
            zero_head=config["zero_head"],
            vis=config["vis"]
        )
        unet.base_config = config
        unet.to(device)

        if 'load_pretrained_model' in kwargs and kwargs['load_pretrained_model']:
            unet.load_from(
                np.load(join(config["pretrained_directory_path"], transformer_config.pretrained_model_name + '.npz')))

        return unet

    def __init__(self, config, img_size, num_classes=21843, zero_head=False, vis=False):
        super(TransUNet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.img_size = img_size
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                print("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                ntok_new = posemb_new.size(1)
                gs_new = int(np.sqrt(ntok_new))
                if gs_new * gs_new == ntok_new:
                    posemb = self.zoom_reshape(posemb_grid, posemb_new)
                else:
                    posemb = self.resize_position_embedding(posemb, posemb_new)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

    @classmethod
    def cache_model_name(cls, base_config) -> str:
        return join(super().cache_model_name(base_config), base_config.network.transformer.pretrained_model_name)

    def zoom_reshape(self, posemb_grid, posemb_new: nn.Parameter):
        ntok_new = posemb_new.size(1)
        gs_old = int(np.sqrt(len(posemb_grid)))
        gs_new = int(np.sqrt(ntok_new))
        print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
        posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
        zoom = (gs_new / gs_old, gs_new / gs_old, 1)
        posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
        return posemb_grid.reshape(1, gs_new * gs_new, -1)

    def resize_position_embedding(self, posemb, posemb_new):
        n_posemb = posemb.size(1)
        gs_old = int(np.sqrt(n_posemb - 1))
        posemb_grid = posemb[:, 1:].transpose(1, 2).reshape(1, -1, gs_old, gs_old)
        grid_size = self.config.patches.grid
        patch_size = (self.img_size[0] // 16 // grid_size[0], self.img_size[1] // 16 // grid_size[1])
        patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
        gs_new_h = self.img_size[0] // patch_size_real[0]
        gs_new_w = self.img_size[1] // patch_size_real[1]
        x = torch.linspace(-1, 1, gs_new_w)
        y = torch.linspace(-1, 1, gs_new_h)
        x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
        grid = torch.stack((x_grid, y_grid), -1).unsqueeze(0)
        posemb_grid_new = grid_sample(posemb_grid, grid, align_corners=True)
        posemb_new_resized = posemb_grid_new.reshape(1, posemb_new.size(2), -1).transpose(1, 2)
        return posemb_new_resized.numpy()
