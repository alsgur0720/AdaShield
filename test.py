import os, time
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from load_data import *
from network import DRS  
from loss import *

n_slice = 21
image_size = 288
client_id = 1

model_path = f'model/federated-learning/DRS/model.pth'

testdataset_dir = f'./data_2/Client{client_id}/raw_data/train'
predict_testdata_dir = 'data_2/process_data/csn_test_segmentation'
if not os.path.exists(predict_testdata_dir):
    os.makedirs(predict_testdata_dir)

testdata_list = select_test_data(testdataset_dir)
data_len = len(testdata_list['testing'])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = DRS().to(device)

model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.eval()




if __name__ == '__main__':
    start_time = time.time()
    total_samples = 1
    criterion = torch_loss_accuracy_3D
    avg_accuracy = 0.0
    loss_avg = 0.0
    for i in range(data_len):  # 테스트 데이터 순회
        print(f"Processing test image {i+1}")
        image_names = testdata_list['testing'][i]
        image_name, gt_name = image_names[0], image_names[1]
        

        nim1 = nib.load(image_name)
        gt_seg = nib.load(gt_name)
        image_volume = nim1.get_fdata()
        gt_seg_volume_ori = gt_seg.get_fdata()
        image_shape=image_volume.shape

        output_dir = "visualized_images"
        os.makedirs(output_dir, exist_ok=True)

        raw_crop_image_volume=image_volume.copy()
        

        raw_shape=raw_crop_image_volume.shape
        crop_image_volume=raw_crop_image_volume

        crop_image_volume = normalization(crop_image_volume, (1.0, 99.0))
        crop_image_volume = crop_image_volume.astype(np.float32)

        gt_seg_volume = normalization(gt_seg_volume_ori, (1.0, 99.0))
        gt_seg_volume = gt_seg_volume.astype(np.float32)

        # pad to 288 288 21
        X, Y, Z = crop_image_volume.shape
        cx, cy, cz = X / 2, Y / 2, Z / 2
        crop_image_volume_pad_3D,firstrow,firstcolumn,firstslice,lastrow,lastcolumn,lastslice = pad_image_for_csn(crop_image_volume, cx, cy, cz, image_size, n_slice)
    
        
        gt_seg_volume_pad_3D,firstrow,firstcolumn,firstslice,lastrow,lastcolumn,lastslice = pad_image_for_csn(gt_seg_volume, cx, cy, cz, image_size, n_slice)

        image_volume_pred = np.zeros((image_size, image_size,n_slice))

        image_volume_pred=crop_image_volume_pad_3D

        image_volume_pred = np.expand_dims(image_volume_pred, axis=0)
        image_volume_gt = np.expand_dims(gt_seg_volume_pad_3D, axis=0)
        
        image_volume_padded_tensor = torch.tensor(image_volume_pred, dtype=torch.float32).to(device)

        crop_image = image_volume_padded_tensor[:, :, :, :]
        crop_image = crop_image.permute(0, 3, 1, 2)
        
        segt_pl = torch.tensor(gt_seg_volume_pad_3D, dtype=torch.long).to(device).squeeze(0)
        
        
        
        crop_segt_pl = segt_pl[:, :, :]
        

        logits_segt, cov, b, output_rec, b_rec = model(crop_image)
        
        
        loss_segt, accuracy_segt, dice_0, dice_1, dice_2, dice_3, dice_all,pred_segt = criterion(logits_segt, segt_pl.unsqueeze(0), 4)
        
        loss_avg += loss_segt.item()
         
        
    
    
    avg_accuracy /= data_len
    print(f"Average accuracy: {avg_accuracy}")
        