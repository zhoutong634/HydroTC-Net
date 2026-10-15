import os
import time

import ml_collections
import torch

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
seed = 666
os.environ['PYTHONHASHSEED'] = str(seed)

cosineLR = True  # whether use cosineLR or not
# Flood prediction inputs: 36 rain channels + 1 DEM channel + 1 time channel = 38
n_channels = 38
n_labels = 1
epochs = 500
img_size = (560, 624)  # Lujiazui DEM: 549 rows x 610 cols, padded to multiples of 16
print_frequency = 10
save_frequency = 5000
vis_frequency = 20
early_stopping_patience =80

pretrain = False
task_name = 'FloodPrediction'
learning_rate = 3e-5
batch_size = 4
flood_threshold = 0.05  # meters
max_water_depth = 10.0  # meters, aligned with dataset clipping
rain_norm_factor = 320.0  # mm/h, fixed from train-set max (~303.17) with margin

# Composite flood loss:
# total = depth + flood extent (BCE + Dice) + spatial gradient
loss_depth_weight = 1.0
loss_flood_bce_weight = 0.3
loss_flood_dice_weight = 0.3
loss_gradient_weight = 0.1
loss_depth_alpha = 2.0
loss_depth_beta = 2.0
loss_underestimate_weight = 0.5
loss_extent_temperature = 30.0
loss_smooth_l1_beta = 0.05 

#model_name = 'HydroTC_Net'
model_name = 'UNet'

project_root = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(project_root, 'data')
train_dataset = os.path.join(data_root, 'train')
val_dataset = os.path.join(data_root, 'val')
test_dataset = os.path.join(data_root, 'test')
dem_path = os.path.join(data_root, 'static', 'dem.input')
pad_multiple = 16
session_name = 'FloodPrediction' + '_' + time.strftime('%m.%d_%Hh%M')
save_path = task_name + '/' + model_name + '/' + session_name + '/'
model_path = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path = save_path + session_name + ".log"
visualize_path = save_path + 'visualize_val/'


##########################################################################
# CTrans configs
##########################################################################
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads = 4
    config.transformer.num_layers = 4
    config.expand_ratio = 4
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 64
    config.n_classes = 1
    return config


# used in testing phase, copy the session name in training phase
test_session = "Test_session_07.03_20h39"
