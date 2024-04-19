import argparse
import os

from unet.data_loader.loader_mapper import get_data_loader
from unet.network import get_support_list

from unet.solver import Solver
from torch.backends import cudnn

from unet.utils import YamlReader


def main(config):
    cudnn.benchmark = True
    if config.network is None:
        print('yaml file must contain network section or network_path must be specified')
        return
    model_type = config.network.model_type
    support_list = get_support_list()
    if model_type not in support_list:
        print(f'ERROR!! model_type should be selected in {"/".join(support_list)}')
        print('Your input for model_type was %s' % config.network.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(config.network.model_path):
        os.makedirs(config.network.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path, config.dataset.type, model_type)
    config.tensorboard_path = os.path.join(config.tensorboard_path, config.dataset.type, model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    print(config)

    train_loader, valid_loader = get_data_loader(config.dataset)
    solver = Solver(config, train_loader, valid_loader)

    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--yaml_path', type=str, default='config/full-config/local/TransUNet.yaml',
                        help='If you set the yaml_path, the config will be read from the yaml file, '
                             'all the other arguments will be ignored.')

    config = parser.parse_args()
    if config.yaml_path is not None:
        config = YamlReader(config.yaml_path).yaml

    main(config)
