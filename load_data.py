import os

import nibabel as nib
import math, random
import numpy as np
import matplotlib.pyplot as plt
import h5py
import gc
import pandas as pd
import re

epsilon = 0.09

def extract_numbers(strings):
    numbers = []
    for s in strings:
        nums = re.findall(r'\d+', s)
        numbers.extend(nums)
    return numbers

def select_training_data(dataset_dir, is_acdc, is_adver=False):
    
    data_list = {}
    for k in ['train', 'validation']:

        subset_dir = os.path.join(dataset_dir, k)
        data_list[k] = []

        
        for patient in sorted(os.listdir(subset_dir)):

            patient_dir = os.path.join(subset_dir, patient)  # F:/ACDC/training/train/patient001

            last_dir = os.path.basename(patient_dir)
            
            # image_name = '{0}/{1}_4d.nii.gz'.format(patient_dir, last_dir)
            # segt_name = '{0}/{1}_sa_gt.nii.gz'.format(patient_dir, last_dir)
            # data_list[k] += [[image_name, segt_name, patient]]
            
            
            for file in sorted(os.listdir(patient_dir)):
                # print(file)
                # exit()
                
                if file[-8] == 't' and file[-21:-18]!='038' and file[-21:-18]!='057' and is_acdc:
                    
                    if is_adver:
                        image_name = '{0}/{1}_frame{2}_adver_example_{3}.nii.gz'.format(patient_dir, patient, file[-12:-10], epsilon)
                        segt_name = '{0}/{1}_frame{2}_adver_result_{3}.nii.gz'.format(patient_dir, patient, file[-12:-10], epsilon)
                    
                    else:
                        image_name = '{0}/{1}_frame{2}.nii.gz'.format(patient_dir, patient, file[-12:-10])
                        segt_name = '{0}/{1}_frame{2}_gt.nii.gz'.format(patient_dir, patient, file[-12:-10])
                    
                    
                    if os.path.exists(image_name) and os.path.exists(segt_name):
                        data_list[k] += [[image_name, segt_name, patient]]
                        
                elif not is_acdc and is_adver and k == 'train':
                    match = re.search(r'frame(\d+)', file)

                    if match:
                        frame_num = match.group(1)
                        
                    # print(frame_num)
                    # exit()
                    # image_name = '{0}/{1}_frame{2}_sa.nii.gz'.format(patient_dir, last_dir, frame_num)
                    # segt_name = '{0}/{1}_frame{2}_adver_result_sa.nii.gz'.format(patient_dir, last_dir, frame_num)
                    image_name = '{0}/{1}_frame{2}_sa_adver_example_{3}.nii.gz'.format(patient_dir, last_dir, frame_num, epsilon)
                    segt_name = '{0}/{1}_frame{2}_sa_adver_result_{3}.nii.gz'.format(patient_dir, last_dir, frame_num, epsilon)
                    if os.path.exists(image_name) and os.path.exists(segt_name):
                        data_list[k] += [[image_name, segt_name, patient]]
                        
                elif not is_acdc:
                    match = re.search(r'frame(\d+)', file)

                    if match:
                        frame_num = match.group(1)
                        
                    # print(frame_num)
                    # exit()
                    image_name = '{0}/{1}_frame{2}_sa.nii.gz'.format(patient_dir, last_dir, frame_num)
                    segt_name = '{0}/{1}_frame{2}_sa_gt.nii.gz'.format(patient_dir, last_dir, frame_num)

                    if os.path.exists(image_name) and os.path.exists(segt_name):
                        data_list[k] += [[image_name, segt_name, patient]]
                        

    return data_list


def select_test_data(testdataset_dir):
   
    data_list = {}
    for k in ['testing']:


        data_list[k] = []

        # for patient in sorted(os.listdir(testdataset_dir)):

        #     patient_dir = os.path.join(testdataset_dir, patient)  # F:/ACDC/training/train/patient001

        #     last_dir = os.path.basename(patient_dir)

        #     data_dir = os.path.join(testdataset_dir, patient) # test_data/patient001

        # data_list[k] = []

        for data in sorted(os.listdir(testdataset_dir)):

            data_dir = os.path.join(testdataset_dir, data) # test_data/patient001

            for fr in sorted(os.listdir(data_dir)):
                if fr[-8] != 'd' and fr[-8] != 'I' and fr[0] != 'M':
                    match = re.search(r'frame(\d+)', fr)
                    if match:
                        frame_num = match.group(1)
                    image_name = '{0}/{1}_frame{2}_sa.nii.gz'.format(data_dir, data, frame_num) # test_data/patient001/patient_frame01.nii.gz
                    seg_name = '{0}/{1}_frame{2}_sa_gt.nii.gz'.format(data_dir, data, frame_num) # test_data/patient001/patient_frame01.nii.gz
                    
                    
                    # image_name = '{0}/{1}_frame{2}.nii.gz'.format(patient_dir, last_dir, frame_num)
                    # seg_name = '{0}/{1}_frame{2}_gt.nii.gz'.format(patient_dir, last_dir, frame_num)
                    # # print(image_name)
                    # exit()
                    if os.path.exists(image_name):
                        data_list[k] += [[image_name, seg_name]]

    return data_list


def select_csn_pred_test(testdataset_dir):
    data_list = {}
    for k in ['testing']:

        data_list[k] = []

        for data in sorted(os.listdir(testdataset_dir)):

            image_name = '{0}/{1}'.format(testdataset_dir, data)

            if os.path.exists(image_name):
                data_list[k] += [image_name]

    return data_list


def normalization(image, range=(1.0, 99.0)):

    low, high = np.percentile(image, range)
    image2 = image
    image2[image < low] = low
    image2[image > high] = high
    # print((high - low), "high - low")
    # print(high, "high")
    # print(low, "low")
    if high != low:
        image2 = (image2 - low) / (high - low)
    # else:
        # image2 = image2
    return image2


def pad_image_3D(image, cx, cy, cz, image_size, n_slice):
    # Pad or crop to the specified size
    X, Y, Z = image.shape[:3]

    rxy = image_size / 2
    r_z = n_slice / 2

    x1, x2 = math.ceil(cx - rxy), math.ceil(cx + rxy)
    y1, y2 = math.ceil(cy - rxy), math.ceil(cy + rxy)
    z1, z2 = math.ceil(cz - r_z), math.ceil(cz + r_z)

    x1_, x2_ = max(x1, 0), min(x2, X)
    y1_, y2_ = max(y1, 0), min(y2, Y)
    z1_, z2_ = max(z1, 0), min(z2, Z)

    pad = image[x1_: x2_, y1_: y2_, z1_: z2_]

    pad = np.pad(pad, ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (z1_ - z1, z2 - z2_)), 'constant')

    return pad


def pad_image_for_csn(image, cx, cy, cz, image_size, n_slice):
    # Pad or crop to the specified size
    X, Y, Z = image.shape[:3]

    rxy = image_size / 2
    r_z = n_slice / 2

    x1, x2 = math.ceil(cx - rxy), math.ceil(cx + rxy)
    y1, y2 = math.ceil(cy - rxy), math.ceil(cy + rxy)
    z1, z2 = math.ceil(cz - r_z), math.ceil(cz + r_z)

    x1_, x2_ = max(x1, 0), min(x2, X)
    y1_, y2_ = max(y1, 0), min(y2, Y)
    z1_, z2_ = max(z1, 0), min(z2, Z)

    pad = image[x1_: x2_, y1_: y2_, z1_: z2_]

    pad = np.pad(pad, ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (z1_ - z1, z2 - z2_)), 'constant')

    first_row = x1_ -x1
    first_column = y1_-y1
    first_slice = z1_-z1
    last_row=image_size-(x2 - x2_)
    last_column = image_size-(y2 - y2_)
    last_slice = n_slice-(z2 - z2_)

    return pad,first_row,first_column,first_slice,last_row,last_column,last_slice


def write_data(dataset, images, gts, first, last, flag):

    print('%d : %d' % (first, last))
    images_tmp = np.asarray(images, dtype=np.float32)
    gts_tmp = np.asarray(gts, dtype=np.uint8)

    dataset['images_%s' % flag][first:last, ...] = images_tmp
    dataset['gts_%s' % flag][first:last, ...] = gts_tmp

    images.clear()
    gts.clear()
    gc.collect()


def read_data(raw_train_data_dir,process_data_dir,image_size,n_slice,is_csn=False, is_acdc=False, is_adver=False):
    if not os.path.exists(process_data_dir):
        os.makedirs(process_data_dir)
    train_data_list = select_training_data(raw_train_data_dir, is_acdc, is_adver)
    train_data_len = len(train_data_list['train'])
    validation_data_len = len(train_data_list['validation'])

    data_save_name = 'size_%s_slices_%s_trainnum_%s_valnum_%s.hdf5' % (image_size, n_slice, train_data_len, validation_data_len)
    data_save_path = os.path.join(process_data_dir, data_save_name)

    # if os.path.exists(data_save_path):
    #     return h5py.File(data_save_path, 'r')

    data_file = h5py.File(data_save_path, "w")

    dataset = {}
    dataset['images_train'] = data_file.create_dataset("images_train", [train_data_len, image_size, image_size, n_slice],
                                                       dtype=np.float32)
    dataset['gts_train'] = data_file.create_dataset("gts_train", [train_data_len, image_size, image_size, n_slice],
                                                    dtype=np.uint8)
    dataset['images_validation'] = data_file.create_dataset("images_validation",
                                                            [validation_data_len, image_size, image_size, n_slice],
                                                            dtype=np.float32)
    dataset['gts_validation'] = data_file.create_dataset("gts_validation",
                                                         [validation_data_len, image_size, image_size, n_slice],
                                                         dtype=np.uint8)

    images = []
    gts = []

    for k in ['train', 'validation']:
        data_len = len(train_data_list[k])
        # print('Reading data')
        # exit()
        queue_num = 0
        first = 0
        # print(data_len)
        # exit()
        for i in range(data_len):
            # print(i)
            image_name = train_data_list[k][i][0]
            gt_name = train_data_list[k][i][1]
            # print(image_name)
            # print(gt_name)
            # exit()
            nim1 = nib.load(image_name)
            image_volume = nim1.get_fdata()
            # print(image_volume.shape)
            nim2 = nib.load(gt_name)
            gt_volume = nim2.get_fdata()

            if is_csn:
                raw_crop_image_volume = image_volume
                raw_crop_gt_volume = gt_volume
            if not is_csn:
                raw_crop_image_volume, firstrow, lastrow, firstcolumn, lastcolumn, raw_crop_gt_volume = cardiac_center_point_positioning(
                    image_volume, gt_volume, image_size)

            raw_shape = raw_crop_image_volume.shape
            # print(raw_shape)

            crop_image_volume = raw_crop_image_volume
            crop_gt_volume = raw_crop_gt_volume

            crop_image_volume = normalization(crop_image_volume, (1.0, 99.0))

            X, Y, Z = crop_image_volume.shape
            cx, cy, cz = X / 2, Y / 2, Z / 2
            # print(crop_image_volume.shape)

            crop_image_volume_pad_3D = pad_image_3D(crop_image_volume, cx, cy, cz, image_size, n_slice)
            crop_gt_volume_pad_3D = pad_image_3D(crop_gt_volume, cx, cy, cz,image_size, n_slice)

            crop_image_volume_pad_3D = crop_image_volume_pad_3D.astype(np.float32)
            crop_gt_volume_pad_3D = crop_gt_volume_pad_3D.astype(np.uint8)

            images.append(crop_image_volume_pad_3D)
            gts.append(crop_gt_volume_pad_3D)
            queue_num += 1
            # print(crop_image_volume_pad_3D.shape)

            if queue_num >= 5:
                last = first + queue_num
                write_data(dataset, images, gts, first, last, k)
                first = last
                queue_num = 0

        last = first + queue_num
        write_data(dataset, images, gts, first, last, k)

    # exit()
    data_file.close()

    return h5py.File(data_save_path, 'r')


def find_the_last_model(model_path):

    epoch_nums = []
    for file in sorted(os.listdir(model_path)):
        if file[-6]!='t' and file[-1]=='a':
            tmp=file.split('-')[-1]
            tmp=int(tmp.split('.')[0])
            epoch_nums.append(tmp)
    last = np.max(epoch_nums)
    return os.path.join(model_path, 'model.ckpt-'+str(last)),last+1


def adjust_data(images_train,masks_train,images_validation,masks_validation,image_size):
  val_slice_num=0
  train_slice_num=0
  #slice_image = np.zeros((image_size, image_size))
  #slice_gt = np.zeros((image_size, image_size))
  train_len = images_train.shape[0]
  val_len = images_validation.shape[0]

  images_train_2D = np.zeros((1680,image_size, image_size))
  masks_train_2D = np.zeros((1680,image_size, image_size))
  images_validation_2D = np.zeros((210,image_size, image_size))
  masks_validation_2D = np.zeros((210,image_size, image_size))
  for i in range(2):
    print(i)
    for j in range(21):
      slice_image = images_train[i, :, :, j]
      if np.max(slice_image) != 0:
        slice_gt=masks_train[i, :, :, j]
        images_validation_2D[val_slice_num,:,:]=slice_image
        masks_validation_2D[val_slice_num,:,:] = slice_gt
        val_slice_num = val_slice_num + 1

  for i in range(20):
    print(i)
    for j in range(21):
      slice_image = images_validation[i, :, :, j]
      if np.max(slice_image) != 0:
        slice_gt=masks_validation[i, :, :, j]
        images_validation_2D[val_slice_num,:,:]=slice_image
        masks_validation_2D[val_slice_num,:,:] = slice_gt
        val_slice_num = val_slice_num + 1

  for i in range(2,63):
    print(i)
    for j in range(21):
      slice_image = images_train[i, :, :, j]
      if np.max(slice_image) != 0:
        slice_gt=masks_train[i, :, :, j]
        images_train_2D[train_slice_num,:,:]=slice_image
        masks_train_2D[train_slice_num,:,:] = slice_gt
        train_slice_num = train_slice_num + 1

  for i in range(2,4):
    print(i)
    for j in range(21):
      slice_image = images_train[i, :, :, j]
      if np.max(slice_image) != 0:
        slice_gt=masks_train[i, :, :, j]
        images_train_2D[train_slice_num,:,:]=slice_image
        masks_train_2D[train_slice_num,:,:] = slice_gt
        train_slice_num = train_slice_num + 1

  print(images_train, masks_train, images_validation, masks_validation)
  print(train_slice_num,val_slice_num)
  return images_train_2D.astype(np.float32),masks_train_2D.astype(np.uint8),images_validation_2D.astype(np.float32),masks_validation_2D.astype(np.uint8)


def adjust_data_3D(images_train,masks_train,images_validation,masks_validation,image_size,n_slice):

  N, _, _, _ = images_train.shape
  N_b, _, _, _ = images_validation.shape
  
  images_train_3D = np.zeros((N,image_size, image_size,n_slice))
  masks_train_3D = np.zeros((N,image_size, image_size,n_slice))
  images_validation_3D = np.zeros((N_b,image_size, image_size,n_slice))
  masks_validation_3D = np.zeros((N_b,image_size, image_size,n_slice))

  for i in range(N):
    print(i)
    for j in range(21):
      images_train_3D[i,:,:,j]=images_train[i, :, :, j]
      masks_train_3D[i,:,:,j] =masks_train[i, :, :, j]

  for i in range(N_b):
    print(i)
    for j in range(21):
      images_validation_3D[i,:,:,j]=images_validation[i, :, :, j]
      masks_validation_3D[i,:,:,j] =masks_validation[i, :, :, j]

  print(images_train, masks_train, images_validation, masks_validation)
  return images_train_3D.astype(np.float32),masks_train_3D.astype(np.uint8),images_validation_3D.astype(np.float32),masks_validation_3D.astype(np.uint8)


def data_augmentation(images, gts, flip=False,gamma=False):

    num_images = images.shape[0]
    aug_images = np.zeros((num_images, 112, 112))
    aug_gts = np.zeros((num_images, 112, 112))

    for i in range(num_images):

        image = images[i, :, :]
        gt = gts[i, :, :]

        if flip:
            random_num = np.random.randint(4)

            if random_num == 0:
                image = np.fliplr(image)
                gt = np.fliplr(gt)
            if random_num == 1:
                image = np.flipud(image)
                gt = np.flipud(gt)
            if random_num == 2:
                image = np.fliplr(np.flipud(image))
                gt = np.fliplr(np.flipud(gt))

        if gamma:

            random_num2 = np.random.uniform(0.5, 1.5)
            random_num3 = 1 - np.abs(random_num2 - 1)
            if random_num2 > 1:
                random_num3 = 1 / random_num3

            image = np.power(image, random_num3)


        aug_images[i, :, :] = image
        aug_gts[i, :, :] = gt

    return aug_images, aug_gts


def data_augmentation_3D(images, gts, flip=False,gamma=False):

    num_images = images.shape[0]
    aug_images = np.zeros((num_images, 288, 288,21))
    aug_gts = np.zeros((num_images, 288,288,21))

    for i in range(num_images):

        image = images[i, :, :,:]
        gt = gts[i, :, :,:]

        if flip:
            random_num = np.random.randint(4)

            for j in range(21):
                if random_num == 0:
                    image[:,:,j] = np.fliplr(image[:,:,j])
                    gt[:,:,j] = np.fliplr(gt[:,:,j])
                if random_num == 1:
                    image[:,:,j]= np.flipud(image[:,:,j])
                    gt[:,:,j] = np.flipud(gt[:,:,j])
                if random_num == 2:
                    image[:,:,j] = np.fliplr(np.flipud(image[:,:,j]))
                    gt[:,:,j] = np.fliplr(np.flipud(gt[:,:,j]))

        if gamma:

            random_num2 = np.random.uniform(0.5, 1.5)
            random_num3 = 1 - np.abs(random_num2 - 1)
            if random_num2 > 1:
                random_num3 = 1 / random_num3
            for j in range(21):
                image[:,:,j] = np.power(image[:,:,j], random_num3)

        aug_images[i, :, :,:] = image
        aug_gts[i, :, :,:] = gt

    return aug_images, aug_gts



def cardiac_center_point_positioning(image_volume,pred_volume,image_size):


    minrow = image_volume.shape[0]-1
    maxrow = 0
    mincolumn = image_volume.shape[1]-1
    maxcolumn = 0

    minrow2 = image_volume.shape[0] - 1
    maxrow2 = 0
    mincolumn2 = image_volume.shape[1] - 1
    maxcolumn2 = 0

    stablenum=20
    # crop_image_volume=np.zeros((size,size,image_volume.shape[-1]))
    pred_volume_copy = pred_volume
    pred_volume = pred_volume==2

    for j in range(pred_volume.shape[-1]):

        firstrow=pred_volume.shape[0]/2
        esc1=0
        num1=0
        for row in range(pred_volume.shape[0]):
            for column in range(pred_volume.shape[1]):
                if(pred_volume[row,column,j]!=0):
                    num1=num1+1
                if num1>=stablenum:
                    firstrow=row
                    esc1=1
                    break
            if esc1==1:
                break

        firstcolumn=pred_volume.shape[1]/2
        esc2=0
        num2=0
        for column in range(pred_volume.shape[1]):
            for row in range(pred_volume.shape[0]):
                if (pred_volume[row, column, j] != 0):
                    num2 = num2 + 1
                if num2 >= stablenum:
                    firstcolumn = column
                    esc2=1
                    break
            if esc2==1:
                break

        lastrow=pred_volume.shape[0]/2
        esc3=0
        num3=0
        for row in range(pred_volume.shape[0]-1,0,-1):
            for column in range(pred_volume.shape[1]-1,0,-1):
                if(pred_volume[row,column,j]!=0):
                    num3 = num3 + 1
                if num3 >= stablenum:
                    lastrow=row
                    esc3=1
                    break
            if esc3==1:
                break

        lastcolumn = pred_volume.shape[1]/2
        esc4 = 0
        num4=0
        for column in range(pred_volume.shape[1]-1,0,-1):
            for row in range(pred_volume.shape[0]-1,0,-1):
                if (pred_volume[row, column, j] != 0):
                    num4 = num4 + 1
                if num4 >= stablenum:
                    lastcolumn = column
                    esc4=1
                    break
            if esc4==1:
                break

        if minrow>firstrow:
            minrow=firstrow
        if maxrow<lastrow:
            maxrow=lastrow
        if firstcolumn<mincolumn:
            mincolumn=firstcolumn
        if lastcolumn>maxcolumn:
            maxcolumn=lastcolumn

    centrerow=int((maxrow+minrow)//2)
    centrecolumn=int((maxcolumn+mincolumn)//2)

    for j in range(pred_volume_copy.shape[-1]):

        firstrow2=pred_volume_copy.shape[0]/2
        for row in range(centrerow,0,-1):
            isthisrow =1
            for column in range(0,pred_volume_copy.shape[1]-1):
                if pred_volume_copy[row,column,j]!=0:
                    isthisrow=0
                    break
            if isthisrow==1:
                firstrow2=row
                break

        lastrow2 = pred_volume_copy.shape[0] / 2
        for row in range(centrerow, pred_volume_copy.shape[0] - 1):
            isthisrow = 1
            for column in range(0, pred_volume_copy.shape[1] - 1):
                if pred_volume_copy[row, column, j] != 0:
                    isthisrow = 0
                    break
            if isthisrow == 1:
                lastrow2 = row
                break

        firstcolumn2 = pred_volume_copy.shape[1] / 2
        for column in range(centrecolumn, 0, -1):
            isthiscolumn = 1
            for row in range(0, pred_volume_copy.shape[0] - 1):
                if pred_volume_copy[row, column, j] != 0:
                    isthiscolumn = 0
                    break
            if isthiscolumn == 1:
                firstcolumn2 = column
                break

        lastcolumn2 = pred_volume_copy.shape[1] / 2
        for column in range(centrecolumn, pred_volume_copy.shape[1] -1):
            isthiscolumn = 1
            for row in range(0, pred_volume_copy.shape[0] - 1):
                if pred_volume_copy[row, column, j] != 0:
                    isthiscolumn = 0
                    break
            if isthiscolumn == 1:
                lastcolumn2 = column
                break

        if minrow2>firstrow2:
            minrow2=firstrow2
        if maxrow2<lastrow2:
            maxrow2=lastrow2
        if firstcolumn2<mincolumn2:
            mincolumn2=firstcolumn2
        if lastcolumn2>maxcolumn2:
            maxcolumn2=lastcolumn2

    centrerow2 = int((maxrow2 + minrow2) // 2)
    centrecolumn2 = int((maxcolumn2 + mincolumn2) // 2)
    row_length=maxrow2 - minrow2
    columen_length=maxcolumn2-mincolumn2
    if columen_length>row_length:
        length=columen_length
    else:
        length=row_length

    raw_crop_size=int(image_size/2) #image_size/2
    #raw_crop_size=56
    crop_index_firstrow = int(centrerow2 - raw_crop_size)
    crop_index_lastrow = int(centrerow2 + raw_crop_size)
    crop_index_firstcolumn = int(centrecolumn2 - raw_crop_size)
    crop_index_lastcolumn = int(centrecolumn2 + raw_crop_size)

    print(crop_index_firstrow,crop_index_lastrow,crop_index_firstcolumn,crop_index_lastcolumn)

    if crop_index_firstrow<=0:
        crop_index_lastrow += (-crop_index_firstrow)
        crop_index_firstrow=0

    if crop_index_firstcolumn <= 0:
        crop_index_lastcolumn += (-crop_index_firstcolumn)
        crop_index_firstcolumn = 0

    if crop_index_lastrow>=image_volume.shape[0]:
        crop_index_firstrow -= (crop_index_lastrow-image_volume.shape[0])
        crop_index_lastrow = image_volume.shape[0]

    if crop_index_lastcolumn>=image_volume.shape[1]:
        crop_index_firstcolumn -= (crop_index_lastcolumn-image_volume.shape[1])
        crop_index_lastcolumn = image_volume.shape[1]


    crop_image_volume=image_volume[crop_index_firstrow:crop_index_lastrow,crop_index_firstcolumn:crop_index_lastcolumn,:]
    crop_pred_volume=pred_volume_copy[crop_index_firstrow:crop_index_lastrow,crop_index_firstcolumn:crop_index_lastcolumn,:]

    return crop_image_volume,crop_index_firstrow,crop_index_lastrow,crop_index_firstcolumn,crop_index_lastcolumn,crop_pred_volume


