import os
import argparse
import torch
from networks.co_training_net import TriDSBAVNet_after8
from test_util import test_all_case_dsba as test_all_case
def str2bool(v):
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/Pancreas-CT-all/', help='Name of Experiment')
parser.add_argument('--image_list_path', type=str, default='pancreas_test.list', help='image_list_path')
parser.add_argument('--dataset_name', type=str, default='pancreas', help='dataset_name')
parser.add_argument('--model', type=str,  default='pancreas_v2_000', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--iter', type=int,  default=6000, help='model iteration')
parser.add_argument('--ds_starting_layer', type=int,  default=8, help='ds_starting_layer')
parser.add_argument('--head_type', type=int,  default=1, help='head_type')
parser.add_argument('--window_size', type=int,  default=2, help='window_size')
parser.add_argument('--self_atten_head_num', type=int,  default=1, help='self_atten_head_num')
parser.add_argument('--sparse_attn', type=str2bool,  default=False, help='sparse_attn')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "../model/"+FLAGS.model+"/"
test_save_path = "../model/prediction/"+FLAGS.model+"_post/"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 2
with open(FLAGS.root_path + '/../'+FLAGS.image_list_path, 'r') as f:
    image_list = f.readlines()
image_list = [os.path.join(FLAGS.root_path ,item.replace('\n', '')+"/mri_norm2.h5" ) for item in image_list]

patch_size = (96, 96, 96)
stride_xy = 16
stride_z = 16
def test_calculate_metric(epoch_num):
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '.pth')
    checkpoint = torch.load(save_mode_path)
    if FLAGS.ds_starting_layer == 8:
        net = TriDSBAVNet_after8(input_channels=1, num_classes=num_classes, head_type = FLAGS.head_type, window_size = FLAGS.window_size, self_atten_head_num = FLAGS.self_atten_head_num, sparse_attn = FLAGS.sparse_attn, has_dropout=True).cuda()
    else:
        raise NotImplementedError
    net.load_state_dict(checkpoint['model'])

    print("init weight from {}".format(save_mode_path))
    net.eval()

    avg_metric = test_all_case(net, image_list, num_classes=num_classes,
                               patch_size=patch_size, stride_xy=stride_xy, stride_z=stride_z,
                               save_result=True, test_save_path=test_save_path)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric(FLAGS.iter)
    print(metric)

