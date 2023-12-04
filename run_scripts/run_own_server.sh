CONFIG_PATH='config/full-config/own_server/'

# U_Net/R2U_Net/AttU_Net/R2AttU_Net

python main.py --yaml_path=$CONFIG_PATH'UNet.yaml'

python main.py --yaml_path=$CONFIG_PATH'R2AttUNet.yaml'

python main.py --yaml_path=$CONFIG_PATH'R2UNet.yaml'

python main.py --yaml_path=$CONFIG_PATH'AttUNet.yaml'