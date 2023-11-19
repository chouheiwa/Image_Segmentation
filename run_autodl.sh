CONFIG_PATH='config/autodl/'

# U_Net/R2U_Net/AttU_Net/R2AttU_Net

python main.py --yaml_path=$CONFIG_PATH'U_Net.yaml'

python main.py --yaml_path=$CONFIG_PATH'R2U_Net.yaml'

python main.py --yaml_path=$CONFIG_PATH'AttU_Net.yaml'

python main.py --yaml_path=$CONFIG_PATH'R2AttU_Net.yaml'