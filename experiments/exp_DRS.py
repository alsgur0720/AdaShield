import os


log_dir1='log'
raw_training_data_dir = './data/Client4/raw_data'
# raw_training_data_dir = './data/raw_data'
process_data_dir = './data/Client4/process_data'
# process_data_dir = './data/process_data'
experiment_name = 'DRS'
main_model_root = 'model/federated-DRS'
detection_model_path = 'loss_train/epoch_200_lr_1e-4_derivate/best_model.pth'
checkpoint_model_dir = os.path.join(main_model_root, experiment_name)
aggregation_method = 'AdaShiled-FL'
n_slice = 21
image_size = 288
train_batch_size=4
validation_batch_size=4
segt_class=4
learning_rate=0.001
lr_decay_rate=0.99
train_epoch=100
continue_training=False
is_drs = True
is_acdc = False
