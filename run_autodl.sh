TRAIN_PATH='/root/autodl-tmp/ISIC2018/Train/ISIC2018_Task1-2_Training_Input'
VERIFY_PATH='/root/autodl-tmp/ISIC2018/Validation/ISIC2018_Task1-2_Validation_Input'
TEST_PATH='/root/autodl-tmp/ISIC2018/Test/ISIC2018_Task1-2_Test_Input'
TENSORBOARD_PATH='/root/tf-logs'

# U_Net/R2U_Net/AttU_Net/R2AttU_Net

python -m main.py \
  --train_path=$TRAIN_PATH \
  --verify_path=$VERIFY_PATH \
  --test_path=$TEST_PATH \
  --tensorboard_path=$TENSORBOARD_PATH \
  --model_type='U_Net'

python -m main.py \
  --train_path=$TRAIN_PATH \
  --verify_path=$VERIFY_PATH \
  --test_path=$TEST_PATH \
  --tensorboard_path=$TENSORBOARD_PATH \
  --model_type='R2U_Net'

python -m main.py \
  --train_path=$TRAIN_PATH \
  --verify_path=$VERIFY_PATH \
  --test_path=$TEST_PATH \
  --tensorboard_path=$TENSORBOARD_PATH \
  --model_type='AttU_Net'

python -m main.py \
  --train_path=$TRAIN_PATH \
  --verify_path=$VERIFY_PATH \
  --test_path=$TEST_PATH \
  --tensorboard_path=$TENSORBOARD_PATH \
  --model_type='R2AttU_Net'