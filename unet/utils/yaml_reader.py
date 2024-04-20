import re
from os.path import dirname, join
from ml_collections import ConfigDict
import yaml
from unet.utils import get_root_dir

suffix = '_path'


def is_relative(path):
    # 如果以/或者~或者\w:\开头，那么就是绝对路径
    # linux文件开头是/或者~
    if path.startswith('/') or path.startswith('~'):
        return False
    # Windows绝对路径开头是类似C:\的
    if re.match(r'^\w:', path):
        return False
    return True


class YamlReader:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise RuntimeError("YamlReader is not initialized")
        return cls._instance

    def __init__(self, path: str):
        self.path = path
        self.yaml = self.read_yaml()
        YamlReader._instance = self

    def read_yaml(self) -> ConfigDict:
        """
        This function create a config from a yaml file.
        @param path: The yaml file path.
        @return: The config dict.
        """
        config_dic = self.read_yaml_template(self.path)
        static_read_yaml = ConfigDict(config_dic)
        return static_read_yaml

    def merge_dicts_deep(self, dict1, dict2):
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge_dicts_deep(result[key], value)
            else:
                result[key] = value
        return result

    def bind_definitions_to_yaml(self, config_dic: ConfigDict, definitions: ConfigDict,
                                 base_yaml_path: str) -> ConfigDict:
        base_dict = {}
        for key in definitions:
            if definitions[key] is None:
                continue
            yaml_path = definitions[key]
            if is_relative(yaml_path):
                yaml_path = join(dirname(base_yaml_path), yaml_path)
            current_yaml = self.read_yaml_template(yaml_path, should_resolve_definitions=True)
            key = key.removesuffix(suffix)
            base_dict[key] = current_yaml
        return self.merge_dicts_deep(base_dict, config_dic)

    def read_yaml_template(self, path, parent_key: str = 'parent_path',
                           should_resolve_definitions: bool = True) -> ConfigDict:
        with open(path, 'r') as f:
            current_dir = yaml.safe_load(f)
            current_definitions = current_dir['definitions'] if 'definitions' in current_dir else {}
            if parent_key not in current_dir:
                if should_resolve_definitions:
                    current_dir = self.bind_definitions_to_yaml(current_dir, current_definitions, path)
                if 'definitions' in current_dir:
                    del current_dir['definitions']
                return current_dir

            parent_path = current_dir[parent_key]
            if is_relative(parent_path):
                parent_path = join(dirname(path), parent_path)
            current_definitions = current_definitions if current_definitions is not None else {}
            parent_dir = self.read_yaml_template(parent_path, parent_key, should_resolve_definitions=False)
            parent_definitions = parent_dir['definitions'] if 'definitions' in parent_dir else {}
            if 'definitions' in parent_dir:
                del parent_dir['definitions']
            parent_definitions = parent_definitions if parent_definitions is not None else {}
            current_definitions = self.merge_dicts_deep(parent_definitions, current_definitions)
            current_dir = self.merge_dicts_deep(parent_dir, current_dir)
            del current_dir[parent_key]

            if should_resolve_definitions:
                if 'definitions' in current_dir:
                    del current_dir['definitions']
                current_dir = self.bind_definitions_to_yaml(current_dir, current_definitions, path)
            else:
                current_dir['definitions'] = current_definitions
            return current_dir


if __name__ == '__main__':
    # print(merge_dicts_deep({'a': 1, 'b': {'c': 1}}, {'a': 3, 'b': {'c': 4}}))
    print(YamlReader(join(get_root_dir(), 'config', 'full-config', 'own_server_data', 'TransUNet.yaml')).yaml)
