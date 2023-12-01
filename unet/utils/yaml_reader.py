from os.path import dirname, join
from ml_collections import ConfigDict
import yaml
from unet.utils import get_root_dir

suffix = '_path'


def is_relative(path):
    return not path.startswith('/')


def read_yaml(path: str) -> ConfigDict:
    """
    This function create a config from a yaml file.
    @param path: The yaml file path.
    @return: The config dict.
    """
    config_dic = read_yaml_template(path)
    return config_dic


def bind_definitions_to_yaml(config_dic: ConfigDict, definitions: ConfigDict, base_yaml_path: str) -> ConfigDict:
    for key in definitions:
        if definitions[key] is None:
            continue
        yaml_path = definitions[key]
        if is_relative(yaml_path):
            yaml_path = join(dirname(base_yaml_path), yaml_path)
        current_yaml = read_yaml_template(yaml_path, should_resolve_definitions=True)
        key = key.removesuffix(suffix)
        config_dic[key] = current_yaml
    return config_dic


def read_yaml_template(path, parent_key: str = 'parent_path', should_resolve_definitions: bool = True) -> ConfigDict:
    with open(path, 'r') as f:
        current_dir = ConfigDict(yaml.safe_load(f))
        current_definitions = current_dir.definitions if 'definitions' in current_dir else ConfigDict({})
        if parent_key not in current_dir:
            if should_resolve_definitions:
                current_dir = bind_definitions_to_yaml(current_dir, current_definitions, path)
            if 'definitions' in current_dir:
                del current_dir['definitions']
            return current_dir

        parent_path = current_dir[parent_key]
        if is_relative(parent_path):
            parent_path = join(dirname(path), parent_path)
        current_definitions = current_definitions if current_definitions is not None else ConfigDict({})
        parent_dir = read_yaml_template(parent_path, parent_key, should_resolve_definitions=False)
        parent_definitions = parent_dir.definitions if 'definitions' in parent_dir else ConfigDict({})
        if 'definitions' in parent_dir:
            del parent_dir['definitions']
        parent_definitions = parent_definitions if parent_definitions is not None else ConfigDict({})
        for key in parent_definitions:
            if key in current_definitions and current_definitions[key] is not None:
                continue
            current_definitions[key] = parent_definitions[key]

        for key in parent_dir:
            if key not in current_dir or current_dir[key] is None:
                current_dir[key] = parent_dir[key]
        del current_dir[parent_key]

        if should_resolve_definitions:
            if 'definitions' in current_dir:
                del current_dir['definitions']
            current_dir = bind_definitions_to_yaml(current_dir, current_definitions, path)
        else:
            current_dir.definitions = current_definitions
        return current_dir


if __name__ == '__main__':
    print(read_yaml(join(get_root_dir(), 'config', 'own_server', 'AttU_Net.yaml')))
