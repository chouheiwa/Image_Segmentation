from os.path import join, dirname, abspath


def get_root_dir():
    """Returns the root directory."""
    return dirname(dirname(dirname(abspath(__file__))))


def get_pre_train_model(model: str) -> str:
    return join(get_root_dir(), 'model', 'vit_checkpoint', 'imagenet21k', f'{model}.npz')
