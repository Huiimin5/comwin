import numpy as np
from glob import glob
from tqdm import tqdm
import h5py
import nrrd
import os
from pathlib import Path
import pydicom
import scipy
import nibabel
import scipy.ndimage


output_size =[96, 96, 96]
desired_spacing = [1, 1, 1]
X,Y,Z = 0,1,2
N = 82
W = 512
H = 512


def write_id_to_disk(volume_ids, to_file_path):
    to_file = open(to_file_path, 'w')
    for n in volume_ids:
        volumeID = 'PANCREAS_{:0>4}\n'.format(n + 1)
        to_file.write(volumeID)
    to_file.close()
def resample3d(image, scan, new_spacing, slice_thickness_manual=None, order=1):
    # Determine current pixel spacing
    if scan[0].SliceThickness is not None:
        spacing = map(float, [scan[0].PixelSpacing[X], scan[0].PixelSpacing[Y], scan[0].SliceThickness])
    else:
        spacing = map(float, [scan[0].PixelSpacing[X], scan[0].PixelSpacing[Y], slice_thickness_manual])
    spacing = np.array(list(spacing))

    resize_x = spacing[X] / new_spacing[X]
    new_shape_x = np.round(image.shape[X] * resize_x)
    resize_x = float(new_shape_x) / float(image.shape[X])
    sx = spacing[X] / resize_x

    resize_y = spacing[Y] / new_spacing[Y]
    new_shape_y = np.round(image.shape[Y] * resize_y)
    resize_y = new_shape_y / image.shape[Y]
    sy = spacing[Y] / resize_y

    resize_z = spacing[Z] / new_spacing[Z]
    new_shape_z = np.round(image.shape[Z] * resize_z)
    resize_z = float(new_shape_z) / float(image.shape[Z])
    sz = spacing[Z] / resize_z

    image = scipy.ndimage.interpolation.zoom(image, (resize_x, resize_y, resize_z), order=order)

    return (image, (sx, sy, sz))


def load_scans(path):
    """Here we assume equidistant in Z slices,
       sort output by InstanceNumber"""
    p = Path(path)
    slices = [pydicom.dcmread(str(Path(path) / s)) for s in p.iterdir() if s.is_file()]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.fabs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.fabs(slices[0].SliceLocation - slices[1].SliceLocation)

    # for s in slices: # assign all slices the same thickness
    #    s.SliceThickness = slice_thickness

    return slices, slice_thickness



# borrowed from https://github.com/Kri-Ol/DICOM-Resampler/blob/master/Resampling3D.ipynb
def covert_h5(raw_data_dir, raw_label_dir, volume_ids, to_dir, with_norm=True, low_range=-125, high_range=275,
              over_crop_size=25, with_resampling=True,crop_first_norm_last=False, label_resamp_order=0):
    # todo: 1. load data from disk

    for n in range(N):
        if n not in volume_ids:
            continue

        # if n != 49:
        #     continue
        volumeID = '{:0>4}'.format(n + 1)
        print('Processing File ' + volumeID)
        filename1 = 'PANCREAS_' + volumeID
        directory1 = os.path.join(raw_data_dir, filename1)
        label_filename1 = 'label' + volumeID + '.nii.gz'
        file1_label = os.path.join(raw_label_dir, label_filename1)
        labels_to_process = nibabel.load(file1_label).get_data().transpose(1, 0, 2)
        for path_, _, file_ in os.walk(directory1):
            L = len(file_)
            if L > 1:
                l = [os.path.join(path_, file) for file in file_]
                print("Total of {0} DICOM images.\nFirst 5 filenames:".format(len(l)))
                patient, slice_thickness_manual = load_scans(path_)
        print('slice_thickness_manual {}'.format(slice_thickness_manual))
        imgs_to_process = np.stack([s.pixel_array for s in patient], axis=Z)  # so final set of images would be (512,512,168)
        print(f"All images together: {imgs_to_process.shape}")
        # todo: 2. window range [−125, 275] HU
        np.minimum(np.maximum(imgs_to_process, low_range, imgs_to_process), high_range, imgs_to_process)

        assert imgs_to_process.shape == labels_to_process.shape
        # todo: 3. resample
        if with_resampling:
            print(f"Shape before resampling\t{imgs_to_process.shape}")
            image, spacing = resample3d(imgs_to_process, patient, desired_spacing, slice_thickness_manual)
            print(f"Shape after resampling:\t{image.shape}")
            print(f"New spacing: {spacing}")
            label, spacing = resample3d(labels_to_process, patient, desired_spacing, slice_thickness_manual, order=label_resamp_order)
            assert ((label==0) + (label == 1)).all()
        else:
            image = imgs_to_process
            label = labels_to_process

        # todo: post_processing labels: > 0 set as fg; check the maximum and find out how to set the fg and bg labels
        label = label.astype(np.uint8)
        # todo: 4. center crop
        w, h, d = label.shape
        tempL = np.nonzero(label)
        minx, maxx = np.min(tempL[0]), np.max(tempL[0])
        miny, maxy = np.min(tempL[1]), np.max(tempL[1])
        minz, maxz = np.min(tempL[2]), np.max(tempL[2])
        minx = max(minx - over_crop_size, 0)
        maxx = min(maxx + over_crop_size, w)
        miny = max(miny - over_crop_size, 0)
        maxy = min(maxy + over_crop_size, h)
        minz = max(minz - over_crop_size, 0)
        maxz = min(maxz + over_crop_size, d)
        # todo: 5. normalize
        if with_norm:
            if not crop_first_norm_last:
                image = (image - np.mean(image)) / np.std(image)
        image = image.astype(np.float32)
        image = image[minx:maxx, miny:maxy, minz:maxz]  # todo: check with or without minz:maxz
        label = label[minx:maxx, miny:maxy, minz:maxz]

        if with_norm:
            if crop_first_norm_last:
                image = (image - np.mean(image)) / np.std(image)
        print(label.shape)
        # todo: 6. save
        new_path = os.path.join(os.path.join(to_dir, filename1), 'mri_norm2.h5')
        print(new_path)
        new_dir = os.path.dirname(new_path)
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        f = h5py.File(new_path, 'w')
        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_dir', type=str, default='../raw_dataset/Pancreas_SSL/Pancreas-CT', help='Name of Experiment')
    parser.add_argument('--raw_label_dir', type=str, default='../raw_dataset/Pancreas-CT/TCIA_pancreas_labels-02-05-2017', help='image_list_path')
    args = parser.parse_args()
    # randomly shuffle volume ids and put them into all_volume_id
    all_volume_id = [30, 68,  2, 44, 46,  1, 64, 14, 12, 61, 42, 36, 67, 53, 74, 38, 31,
        9,  7, 56, 80, 13, 34, 55,  4, 19, 20,  5, 51, 48, 39, 29, 33, 10,
        6,  8, 28, 17, 75, 21, 50, 60, 32, 18, 66, 47, 58, 81, 37, 70, 62,
       77, 27, 16, 59, 76, 79, 23, 22, 49, 3, 15, 57, 35, 40, 45,
        78, 11, 41, 54, 43, 72, 65, 73, 52, 63, 25, 0, 26, 71]
    training_volume_ids = all_volume_id[:60]
    test_volume_ids = all_volume_id[60:]

    write_id_to_disk(training_volume_ids, os.path.join('../../data/', 'pancreas_train.list'))
    write_id_to_disk(test_volume_ids, os.path.join('../../data/', 'pancreas_test.list'))

    to_dir_train = '../../data/Pancreas-CT-all'
    if not os.path.exists(to_dir_train):
        os.makedirs(to_dir_train)
    covert_h5(args.raw_data_dir, raw_label_dir, training_volume_ids, to_dir_train, True,  crop_first_norm_last=True)
    # to build test set
    to_dir_test = '../../data/Pancreas-CT-all'
    if not os.path.exists(to_dir_test):
        os.makedirs(to_dir_test)
    covert_h5(args.raw_data_dir, raw_label_dir, test_volume_ids, to_dir_test, True, crop_first_norm_last=True)



