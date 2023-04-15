import math
from glob import glob

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from tqdm import tqdm


def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return np.array([dice, hd95])
    else:
        return np.zeros(2)

def test_batch(prediction, label, num_classes=2):
    total_metric = np.zeros((num_classes-1, 2))
    for i in range(1, num_classes):
        total_metric[i-1, :] += cal_metric(label == i, prediction == i)
    return total_metric

