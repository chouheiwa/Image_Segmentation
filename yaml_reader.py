from os.path import dirname, join

import yaml


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, d=None, **kwargs):
        super(DotDict, self).__init__()
        if d is not None:
            for key, value in d.items():
                self[key] = DotDict(value) if isinstance(value, dict) else value
        for key, value in kwargs.items():
            self[key] = DotDict(value) if isinstance(value, dict) else value


def is_relative(path):
    return not path.startswith('/')


def read_yaml(path):
    with open(path, 'r') as f:
        current_dir = DotDict(yaml.safe_load(f))
        if 'parent_path' not in current_dir:
            return current_dir

        parent_path = current_dir['parent_path']
        if is_relative(parent_path):
            parent_path = join(dirname(path), parent_path)

        parent_dir = read_yaml(parent_path)
        for key in parent_dir:
            if key not in current_dir:
                current_dir[key] = parent_dir[key]
        del current_dir['parent_path']
        return current_dir


if __name__ == '__main__':
    print(read_yaml('config/config_default.yaml'))
    print(read_yaml(join('config', 'autodl', 'default.yaml')).train_path)
