import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.co_training_net import TriDSBAVNet_after8
from dataloaders import utils
from utils import ramps, losses
import torchio
from dataloaders.pancreas import Pancreas
from dataloaders.colon import Colon

from dataloaders.la_heart import RandomCrop, CenterCrop,ToTensor, LabeledBatchSampler, UnlabeledBatchSampler
from val_3D import test_batch
import matplotlib.pyplot as plt

def str2bool(v):
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/Pancreas-CT-all', help='Name of Experiment')
parser.add_argument('--image_list_path', type=str, default='pancreas_train.list', help='image_list_path')
parser.add_argument('--dataset_name', type=str, default='pancreas', help='dataset_name')

parser.add_argument('--exp', type=str,  default='pancreas_v2_000', help='model_name')
parser.add_argument('--labeled_num', type=int,  default=3, help='labeled_num')
parser.add_argument('--total_num', type=int,  default=60, help='total_num')
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')

parser.add_argument('--ds_starting_layer', type=int,  default=8, help='ds_starting_layer')
parser.add_argument('--head_type', type=int,  default=1, help='head_type')
parser.add_argument('--window_size', type=int,  default=2, help='window_size')
parser.add_argument('--self_atten_head_num', type=int,  default=1, help='self_atten_head_num')
parser.add_argument('--sparse_attn', type=str2bool,  default=False, help='sparse_attn')


### costs
parser.add_argument('--with_dice', type=str2bool,  default=True, help='with_dice loss')
parser.add_argument('--verbose', type=str2bool,  default=False, help='verbose')

parser.add_argument('--cps_la_weight_final', type=float,  default=0.1, help='cps_la_weight_final')
parser.add_argument('--cps_la_rampup_scheme', type=str,  default='None', help='cps_la_rampup_scheme')
parser.add_argument('--cps_la_rampup', type=float,  default=40.0, help='cps_la_rampup')
parser.add_argument('--cps_la_with_dice', type=str2bool,  default=True, help='cps_la_with_dice')

parser.add_argument('--cps_un_weight_final', type=float,  default=0.1, help='consistency')
parser.add_argument('--cps_un_rampup_scheme', type=str,  default='None', help='cps_rampup_scheme')
parser.add_argument('--cps_un_rampup', type=float,  default=40.0, help='cps_rampup')
parser.add_argument('--cps_un_with_dice', type=str2bool,  default=True, help='cps_un_with_dice')


# resume
parser.add_argument('--resume', type=str2bool, default=False, help='resume')
parser.add_argument('--load_epoch_num', type=int, default=3000, help='load_epoch_num')
parser.add_argument('--load_model_name', type=str, default='exp_pancreas_000', help='load_model_name')
parser.add_argument('--fix_bn_after_resume', type=str2bool, default=True, help='fix_bn_after_resume')

args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs
with_dice = args.with_dice


if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (96, 96, 96)


def get_unsup_cont_weight(epoch, weight, scheme, ramp_up_or_down ):
    if  scheme == 'sigmoid_rampup':
        return weight * ramps.sigmoid_rampup(epoch, ramp_up_or_down)
    elif scheme == 'linear_rampup':
        return weight * ramps.linear_rampup(epoch, ramp_up_or_down)
    elif scheme == 'log_rampup':
        return weight * ramps.log_rampup(epoch, ramp_up_or_down)
    elif scheme == 'exp_rampup':
        return weight * ramps.exp_rampup(epoch, ramp_up_or_down)
    elif scheme == 'quadratic_rampdown':
        return weight * ramps.quadratic_rampdown(epoch, ramp_up_or_down)
    elif scheme == 'cosine_rampdown':
        return weight * ramps.cosine_rampdown(epoch, ramp_up_or_down)
    else:
        return weight

def get_supervised_loss(outputs, label_batch,  with_dice=True):
    loss_seg = F.cross_entropy(outputs, label_batch)
    outputs_soft = F.softmax(outputs, dim=1)
    if with_dice:
        loss_seg_dice = losses.dice_loss(outputs_soft[:, 1, :, :, :], label_batch == 1)
        supervised_loss = 0.5 * (loss_seg + loss_seg_dice)
    else:
        loss_seg_dice = torch.zeros([1]).cuda()
        supervised_loss = loss_seg + loss_seg_dice
    return supervised_loss, loss_seg, loss_seg_dice
    

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    def init_parameters(model):
        load_model_path = "../model/" + args.load_model_name + "/"
        save_mode_path = os.path.join(load_model_path, 'iter_' + str(args.load_epoch_num) + '.pth')
        checkpoint = torch.load(save_mode_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])



    def create_model():
        # Network definition
        if args.ds_starting_layer == 8:
            model = TriDSBAVNet_after8(input_channels=1, num_classes=num_classes, head_type = args.head_type, window_size = args.window_size, self_atten_head_num = args.self_atten_head_num,sparse_attn = args.sparse_attn, has_dropout=True)
        else:
            raise NotImplementedError

        model.cuda()
        return model


    scale_num = 2 # hard coded
    model = create_model()
    # criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)

    """data loader and sampler for labeled training"""
    transforms_train = transforms.Compose([
        RandomCrop(patch_size),
        ToTensor(),
    ])

    if args.dataset_name == 'pancreas':
        db_train_labeled = Pancreas(base_dir=train_data_path,
                                    split='train',
                                    transform=transforms_train,
                                    image_list_path=args.image_list_path)
    elif args.dataset_name == 'colon':
        db_train_labeled = Colon(base_dir=train_data_path,
                                 split='train',
                                 transform=transforms_train,
                                 image_list_path=args.image_list_path)
    else:
        assert False

    """data loader and sampler for unlabeled training"""
    transforms_train_unlabeled = transforms.Compose([
        RandomCrop(patch_size),
        ToTensor(),
    ])

    if args.dataset_name == 'pancreas':
        db_train_unlabeled = Pancreas(base_dir=train_data_path,
                                      split='train',
                                      transform=transforms_train_unlabeled,
                                      image_list_path=args.image_list_path)
    elif args.dataset_name == 'colon':
        db_train_unlabeled = Colon(base_dir=train_data_path,
                                   split='train',
                                   transform=transforms_train_unlabeled,
                                   image_list_path=args.image_list_path)
    else:
        assert False


    labeled_idxs = list(range(args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, args.total_num))
    labeled_batch_sampler = LabeledBatchSampler(labeled_idxs, labeled_bs)
    unlabeled_batch_sampler = UnlabeledBatchSampler(unlabeled_idxs, batch_size - labeled_bs)

    db_test = Pancreas(base_dir=train_data_path,
                      split='test',
                      transform=transforms.Compose([
                          CenterCrop(patch_size),
                          ToTensor()
                      ]),)


    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    labeledtrainloader = DataLoader(db_train_labeled, batch_sampler=labeled_batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    unlabeledtrainloader = DataLoader(db_train_unlabeled, batch_sampler=unlabeled_batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    if args.resume:
        init_parameters(model)

    optimizer_1 = optim.SGD(model.branch1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_2 = optim.SGD(model.branch2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_3 = optim.SGD(model.branch3.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(labeledtrainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(labeledtrainloader)+1
    lr_ = base_lr
    model.train()


    def pseudo_labeling_from_most_confident_prediction(pred_1, pred_2, max_1, max_2):
        prob_all_ex_3 = torch.stack([pred_1, pred_2], dim=2)  # bs, n_c, n_branch - 1, h, w, d
        max_all_ex_3 = torch.stack([max_1, max_2], dim=1)  # bs, n_branch - 1, h, w, d
        max_conf_each_branch_ex_3, _ = torch.max(prob_all_ex_3, dim=1)  # bs, n_branch - 1, h, w, d
        max_conf_ex_3, branch_id_max_conf_ex_3 = torch.max(max_conf_each_branch_ex_3, dim=1,
                                                              keepdim=True)  # bs, h, w, d
        pseudo_12 = torch.gather(max_all_ex_3, dim=1, index=branch_id_max_conf_ex_3)[:, 0]

        max_conf_fg_ex_3 = max_conf_ex_3[:, 0][pseudo_12 == 1]
        try:
            mean_max_conf_fg_ex_3, var_max_conf_fg_ex_3, min_max_conf_fg_ex_3, max_max_conf_fg_ex_3 = torch.mean(
                max_conf_fg_ex_3).detach().cpu(), torch.var(max_conf_fg_ex_3).detach().cpu(), torch.min(
                max_conf_fg_ex_3).detach().cpu(), torch.max(max_conf_fg_ex_3).detach().cpu()
        except:
            mean_max_conf_fg_ex_3, var_max_conf_fg_ex_3, min_max_conf_fg_ex_3, max_max_conf_fg_ex_3 = 0, 0, 0, 0
        return pseudo_12, branch_id_max_conf_ex_3, [mean_max_conf_fg_ex_3, var_max_conf_fg_ex_3, min_max_conf_fg_ex_3, max_max_conf_fg_ex_3]


    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, (labeled_sampled_batch, unlabeled_sampled_batch) in enumerate(zip(labeledtrainloader, unlabeledtrainloader)):
            time2 = time.time()

            unlabeled_volume_batch, unlabel_label_batch = unlabeled_sampled_batch['image'], unlabeled_sampled_batch['label']
            labeled_volume_batch, label_label_batch = labeled_sampled_batch['image'], labeled_sampled_batch['label']

            # push to gpu
            unlabeled_volume_batch, unlabel_label_batch = unlabeled_volume_batch.cuda(), unlabel_label_batch.cuda()
            labeled_volume_batch, label_label_batch = labeled_volume_batch.cuda(), label_label_batch.cuda()

            forward_step_num = scale_num * 2 - 1
            # todo: organize labels and input data at diff scales
            label_label_batch_list, unlabel_label_batch_list = [label_label_batch], [unlabel_label_batch]
            labeled_volume_batch_list, unlabeled_volume_batch_list = [labeled_volume_batch], [unlabeled_volume_batch]
            for scale_id in range(scale_num - 2, -1, -1):
                out_size = patch_size[0] // (2 ** (scale_num - scale_id - 1)), patch_size[1] // (2 ** (scale_num - scale_id - 1)), patch_size[2] // (2 ** (scale_num - scale_id - 1)),
                x = torch.linspace(-1, 1, out_size[0]).view(-1, 1, 1).repeat(1, out_size[1], out_size[2])
                y = torch.linspace(-1, 1, out_size[1]).view(1, -1, 1).repeat(out_size[0], 1, out_size[2])
                z = torch.linspace(-1, 1, out_size[2]).view(1, 1, -1).repeat(out_size[0], out_size[1], 1)
                grid = torch.cat((z.unsqueeze(3), y.unsqueeze(3), x.unsqueeze(3)), 3) #
                grid = grid.unsqueeze(0).repeat(label_label_batch.size()[0], 1, 1, 1, 1)
                label_label_batch_cur_res = F.grid_sample(label_label_batch_list[0].unsqueeze(1).float(), grid.cuda(), mode='nearest')[:,0].long()
                unlabel_label_batch_cur_res = F.grid_sample(unlabel_label_batch_list[0].unsqueeze(1).float(), grid.cuda(), mode='nearest')[:,0].long()
                label_label_batch_list.insert(0, label_label_batch_cur_res)
                unlabel_label_batch_list.insert(0, unlabel_label_batch_cur_res)

                labeled_volume_batch_cur_res = F.grid_sample(labeled_volume_batch_list[0], grid.cuda(), mode='nearest')
                unlabeled_volume_batch_cur_res = F.grid_sample(unlabeled_volume_batch_list[0], grid.cuda(), mode='nearest')
                labeled_volume_batch_list.insert(0, labeled_volume_batch_cur_res)
                unlabeled_volume_batch_list.insert(0, unlabeled_volume_batch_cur_res)

            logits_sup_1_list, logits_sup_2_list, logits_sup_3_list = [], [], []
            logits_unsup_1_list, logits_unsup_2_list, logits_unsup_3_list = [], [], []
            max_la_1_list, max_la_2_list, max_la_3_list = [], [], []
            max_un_1_list, max_un_2_list, max_un_3_list = [], [], []
            loss_sup_1, loss_sup_2, loss_sup_3 = 0, 0, 0
            cps_la_loss, cps_un_loss = 0, 0

            # todo: hard code iteration through sub-networks
            # todo: step1 forward propagation
            scale_id = 0
            foward_step = 1
            x1_sup_1, x8_sup_1, out_at8_sup_1 = model(labeled_volume_batch, step=1, foward_step = foward_step)
            x1_unsup_1, x8_unsup_1, out_at8_unsup_1 = model(unlabeled_volume_batch, step=1, foward_step=foward_step)
            x1_sup_2, x8_sup_2, out_at8_sup_2 = model(labeled_volume_batch, step=2, foward_step=foward_step)
            x1_unsup_2, x8_unsup_2, out_at8_unsup_2 = model(unlabeled_volume_batch, step=2, foward_step=foward_step)
            x1_sup_3, x8_sup_3, out_at8_sup_3 = model(labeled_volume_batch, step=3, foward_step=foward_step)
            x1_unsup_3, x8_unsup_3, out_at8_unsup_3 = model(unlabeled_volume_batch, step=3, foward_step=foward_step)
            # todo: gather all logits at current resolution scale
            logits_sup_1_list.append(out_at8_sup_1)
            logits_sup_2_list.append(out_at8_sup_2)
            logits_sup_3_list.append(out_at8_sup_3)
            logits_unsup_1_list.append(out_at8_unsup_1)
            logits_unsup_2_list.append(out_at8_unsup_2)
            logits_unsup_3_list.append(out_at8_unsup_3)
            # todo: gather supervised loss at current resolution scale
            loss_sup_1 += get_supervised_loss(logits_sup_1_list[scale_id], label_label_batch_list[scale_id],args.with_dice)[0]
            loss_sup_2 += get_supervised_loss(logits_sup_2_list[scale_id], label_label_batch_list[scale_id],args.with_dice)[0]
            loss_sup_3 += get_supervised_loss(logits_sup_3_list[scale_id], label_label_batch_list[scale_id],args.with_dice)[0]

            # todo: generate pseudo labels for labeled data at current resolution scale
            _, max_la_1 = torch.max(logits_sup_1_list[scale_id], dim=1)
            _, max_la_2 = torch.max(logits_sup_2_list[scale_id], dim=1)
            _, max_la_3 = torch.max(logits_sup_3_list[scale_id], dim=1)
            max_la_1_list.append(max_la_1.long());
            max_la_2_list.append(max_la_2.long());
            max_la_3_list.append(max_la_3.long())
            pred_sup_1 = F.softmax(logits_sup_1_list[scale_id], dim=1)
            pred_sup_2 = F.softmax(logits_sup_2_list[scale_id], dim=1)
            pred_sup_3 = F.softmax(logits_sup_3_list[scale_id], dim=1)
            pseudo_la_12, branch_id_la_max_conf_ex_3, _ = pseudo_labeling_from_most_confident_prediction(pred_sup_1,pred_sup_2,max_la_1,max_la_2)
            pseudo_la_13, branch_id_la_max_conf_ex_2, _ = pseudo_labeling_from_most_confident_prediction(pred_sup_1, pred_sup_3,max_la_1, max_la_3)
            pseudo_la_23, branch_id_la_max_conf_ex_1, _ = pseudo_labeling_from_most_confident_prediction(pred_sup_2, pred_sup_3, max_la_2, max_la_3)
            if args.cps_la_weight_final==0:
                cps_la_loss = torch.zeros([1]).cuda()
            else:

                # todo: gather cps loss for labeled data at current resolution scale
                cps_la_loss += \
                get_supervised_loss(logits_sup_1_list[scale_id], pseudo_la_23, args.cps_la_with_dice)[0] + \
                get_supervised_loss(logits_sup_2_list[scale_id], pseudo_la_13, args.cps_la_with_dice)[0] + \
                get_supervised_loss(logits_sup_3_list[scale_id], pseudo_la_12, args.cps_la_with_dice)[0]
            # todo: generate pseudo labels for unlabeled data at current resolution scale
            _, max_un_1 = torch.max(logits_unsup_1_list[scale_id], dim=1)
            _, max_un_2 = torch.max(logits_unsup_2_list[scale_id], dim=1)
            _, max_un_3 = torch.max(logits_unsup_3_list[scale_id], dim=1)
            max_un_1_list.append(max_un_1.long());
            max_un_2_list.append(max_un_2.long());
            max_un_3_list.append(max_un_3.long())
            pred_unsup_1 = F.softmax(logits_unsup_1_list[scale_id], dim=1)
            pred_unsup_2 = F.softmax(logits_unsup_2_list[scale_id], dim=1)
            pred_unsup_3 = F.softmax(logits_unsup_3_list[scale_id], dim=1)
            pseudo_un_12, branch_id_un_max_conf_ex_3, _ = pseudo_labeling_from_most_confident_prediction(pred_unsup_1, pred_unsup_2, max_un_1, max_un_2)
            pseudo_un_13, branch_id_un_max_conf_ex_2, _ = pseudo_labeling_from_most_confident_prediction(pred_unsup_1, pred_unsup_3, max_un_1, max_un_3)
            pseudo_un_23, branch_id_un_max_conf_ex_1, _ = pseudo_labeling_from_most_confident_prediction(pred_unsup_2, pred_unsup_3, max_un_2, max_un_3)
            # todo: step2 forward propagation
            foward_step = 2
            x8_after_sup_1 = model(x8_sup_1, pseudo_labels=pseudo_la_23, step=1, foward_step=foward_step)
            x8_after_unsup_1 = model(x8_unsup_1, pseudo_labels=pseudo_un_23, step=1, foward_step=foward_step)
            x8_after_sup_2 = model(x8_sup_2, pseudo_labels=pseudo_la_13, step=2, foward_step=foward_step)
            x8_after_unsup_2 = model(x8_unsup_2, pseudo_labels=pseudo_un_13, step=2, foward_step=foward_step)
            x8_after_sup_3 = model(x8_sup_3, pseudo_labels=pseudo_la_12, step=3, foward_step=foward_step)
            x8_after_unsup_3 = model(x8_unsup_3, pseudo_labels=pseudo_un_12, step=3, foward_step=foward_step)
            # todo: gather cps loss for unlabeled data at current resolution scale
            cps_un_loss += \
            get_supervised_loss(logits_unsup_1_list[scale_id], pseudo_un_23, args.cps_un_with_dice)[0] + \
            get_supervised_loss(logits_unsup_2_list[scale_id], pseudo_un_13, args.cps_un_with_dice)[0] + \
            get_supervised_loss(logits_unsup_3_list[scale_id], pseudo_un_12, args.cps_un_with_dice)[0]
            # todo: if not last output: continue
            # todo: step3 forward propagation at full scale
            foward_step = 3
            scale_id = 1
            logits_sup_1 = model([x1_sup_1, x8_after_sup_1], step=1, foward_step=foward_step)
            logits_unsup_1 = model([x1_unsup_1, x8_after_unsup_1], step=1, foward_step=foward_step)
            logits_sup_2 = model([x1_sup_2, x8_after_sup_2], step=2, foward_step=foward_step)
            logits_unsup_2 = model([x1_unsup_2, x8_after_unsup_2], step=2, foward_step=foward_step)
            logits_sup_3 = model([x1_sup_3, x8_after_sup_3], step=3, foward_step=foward_step)
            logits_unsup_3 = model([x1_unsup_3, x8_after_unsup_3], step=3, foward_step=foward_step)
            # todo: step3 gather all logits at full scale
            logits_sup_1_list.append(logits_sup_1)
            logits_sup_2_list.append(logits_sup_2)
            logits_sup_3_list.append(logits_sup_3)
            logits_unsup_1_list.append(logits_unsup_1)
            logits_unsup_2_list.append(logits_unsup_2)
            logits_unsup_3_list.append(logits_unsup_3)
            # todo: step3 gather supervised loss at full scale
            loss_sup_1 += get_supervised_loss(logits_sup_1_list[scale_id], label_label_batch_list[scale_id],args.with_dice)[0]
            loss_sup_2 += get_supervised_loss(logits_sup_2_list[scale_id], label_label_batch_list[scale_id],args.with_dice)[0]
            loss_sup_3 += get_supervised_loss(logits_sup_3_list[scale_id], label_label_batch_list[scale_id],args.with_dice)[0]

            # todo: step3 generate pseudo labels for unlabeled data at full scale
            _, max_un_1 = torch.max(logits_unsup_1_list[scale_id], dim=1)
            _, max_un_2 = torch.max(logits_unsup_2_list[scale_id], dim=1)
            _, max_un_3 = torch.max(logits_unsup_3_list[scale_id], dim=1)
            max_un_1_list.append(max_un_1.long());
            max_un_2_list.append(max_un_2.long());
            max_un_3_list.append(max_un_3.long())
            pred_unsup_1 = F.softmax(logits_unsup_1_list[scale_id], dim=1)
            pred_unsup_2 = F.softmax(logits_unsup_2_list[scale_id], dim=1)
            pred_unsup_3 = F.softmax(logits_unsup_3_list[scale_id], dim=1)
            pseudo_un_12, branch_id_un_max_conf_ex_3, statistics_un_3 = pseudo_labeling_from_most_confident_prediction(pred_unsup_1, pred_unsup_2, max_un_1, max_un_2)
            pseudo_un_13, branch_id_un_max_conf_ex_2, statistics_un_2 = pseudo_labeling_from_most_confident_prediction(pred_unsup_1, pred_unsup_3, max_un_1, max_un_3)
            pseudo_un_23, branch_id_un_max_conf_ex_1, statistics_un_1 = pseudo_labeling_from_most_confident_prediction(pred_unsup_2, pred_unsup_3, max_un_2, max_un_3)
            # todo: step3 evaluate statistics of pseudo labels for unlabeled data at full scale
            mean_un_max_conf_fg_ex_1, var_un_max_conf_fg_ex_1, min_un_max_conf_fg_ex_1, max_un_max_conf_fg_ex_1 = statistics_un_1
            mean_un_max_conf_fg_ex_2, var_un_max_conf_fg_ex_2, min_un_max_conf_fg_ex_2, max_un_max_conf_fg_ex_2 = statistics_un_2
            mean_un_max_conf_fg_ex_3, var_un_max_conf_fg_ex_3, min_un_max_conf_fg_ex_3, max_un_max_conf_fg_ex_3 = statistics_un_3
            # todo: gather cps loss for unlabeled data at current resolution scale
            cps_un_loss += \
            get_supervised_loss(logits_unsup_1_list[scale_id], pseudo_un_23, args.cps_un_with_dice)[0] + \
            get_supervised_loss(logits_unsup_2_list[scale_id], pseudo_un_13, args.cps_un_with_dice)[0] + \
            get_supervised_loss(logits_unsup_3_list[scale_id], pseudo_un_12, args.cps_un_with_dice)[0]

            """gather all losses"""
            cps_la_weight = get_unsup_cont_weight(iter_num // 150, args.cps_la_weight_final,
                                                  scheme=args.cps_la_rampup_scheme, ramp_up_or_down=args.cps_la_rampup)
            cps_un_weight = get_unsup_cont_weight(iter_num // 150, args.cps_un_weight_final,
                                             scheme=args.cps_un_rampup_scheme, ramp_up_or_down=args.cps_un_rampup)

            loss = loss_sup_1 + loss_sup_2 + loss_sup_3 + cps_la_loss * cps_la_weight + cps_un_loss * cps_un_weight

            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            optimizer_3.zero_grad()
            loss.backward()
            optimizer_1.step()
            optimizer_2.step()
            optimizer_3.step()


            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_sup_1', loss_sup_1, iter_num)
            writer.add_scalar('loss/loss_sup_2', loss_sup_2, iter_num)
            writer.add_scalar('loss/loss_sup_3', loss_sup_3, iter_num)
            writer.add_scalar('loss/loss_cps_la', cps_la_loss, iter_num)
            writer.add_scalar('loss/cps_weight_la', cps_la_weight, iter_num)
            writer.add_scalar('loss/loss_cps_un', cps_un_loss, iter_num)
            writer.add_scalar('loss/cps_weight_un', cps_un_weight, iter_num)




            basic_info = 'iteration %d : loss : %f loss_sup_1: %f, loss_sup_2: %f, loss_sup_3: %f, cps_la_loss: %f,  cps_la_weight: %f, cps_un_loss: %f,  cps_un_weight: %f '% \
                         (iter_num, loss.item(), loss_sup_1.item(), loss_sup_2.item(), loss_sup_3.item(), cps_la_loss.item(), cps_la_weight, cps_un_loss.item(),  cps_un_weight)



            iter_num = iter_num + 1

            if iter_num % 50 == 0:
                slice_start_list, slice_end_list, slice_interval_list = [20], [61], [10]
                for scale_id in range(scale_num - 1):
                    slice_start = slice_start_list[0] // 2
                    slice_interval = slice_interval_list[0] // 2
                    slice_end = slice_start + slice_interval * 4 + 1
                    slice_start_list.insert(0, slice_start)
                    slice_end_list.insert(0, slice_end)
                    slice_interval_list.insert(0, slice_interval)

                for scale_id in range(scale_num):
                    pred_sup_1 = torch.argmax(logits_sup_1_list[scale_id], dim=1)
                    pred_sup_2 = torch.argmax(logits_sup_2_list[scale_id], dim=1)
                    pred_sup_3 = torch.argmax(logits_sup_3_list[scale_id], dim=1)
                    pred_unsup_1 = torch.argmax(logits_unsup_1_list[scale_id], dim=1)
                    pred_unsup_2 = torch.argmax(logits_unsup_2_list[scale_id], dim=1)
                    pred_unsup_3 = torch.argmax(logits_unsup_3_list[scale_id], dim=1)

                    metric_labeled_1 = test_batch(pred_sup_1.cpu().data.numpy(),
                                                label_label_batch_list[scale_id].cpu().data.numpy(), num_classes=num_classes)
                    metric_labeled_2 = test_batch(pred_sup_2.cpu().data.numpy(),
                                                  label_label_batch_list[scale_id].cpu().data.numpy(), num_classes=num_classes)
                    metric_labeled_3 = test_batch(pred_sup_3.cpu().data.numpy(),
                                                  label_label_batch_list[scale_id].cpu().data.numpy(), num_classes=num_classes)

                    metric_unlabeled_1 = test_batch(pred_unsup_1.cpu().data.numpy(),
                                                  unlabel_label_batch_list[scale_id].cpu().data.numpy(), num_classes=num_classes)
                    metric_unlabeled_2 = test_batch(pred_unsup_2.cpu().data.numpy(),
                                                    unlabel_label_batch_list[scale_id].cpu().data.numpy(), num_classes=num_classes)
                    metric_unlabeled_3 = test_batch(pred_unsup_3.cpu().data.numpy(),
                                                    unlabel_label_batch_list[scale_id].cpu().data.numpy(), num_classes=num_classes)


                    print(metric_unlabeled_3)

                    for i in range(num_classes - 1):
                        writer.add_scalars('train_evaluator/dice_class{}_scale_{}'.format(i+1, scale_id),
                                           {'labeled_1': metric_labeled_1[i][0], 'labeled_2': metric_labeled_2[i][0], 'labeled_3': metric_labeled_3[i][0],
                                            'unlabeled_1': metric_unlabeled_1[i][0], 'unlabeled_2': metric_unlabeled_2[i][0], 'unlabeled_3': metric_unlabeled_3[i][0]},
                                           iter_num)
                        writer.add_scalars('train_evaluator/hd95_class{}_scale_{}'.format(i+1, scale_id),
                                           {'labeled_1': metric_labeled_1[i][1], 'labeled_2': metric_labeled_2[i][1], 'labeled_3': metric_labeled_3[i][1],
                                            'unlabeled_1': metric_unlabeled_1[i][1],'unlabeled_2': metric_unlabeled_2[i][1], 'unlabeled_3': metric_unlabeled_3[i][1],
                                            }, iter_num)
                        basic_info += 'scale%d: dice_labeled_1: %f,  dice_labeled_2: %f, dice_labeled_3: %f, ' \
                                      'dice_unlabeled_1: %f, dice_unlabeled_2: %f, dice_unlabeled_3: %f ' % \
                                      (scale_id, metric_labeled_1[i][0], metric_labeled_2[i][0], metric_labeled_3[i][0],
                                       metric_unlabeled_1[i][0], metric_unlabeled_2[i][0], metric_unlabeled_3[i][0])

                    image = labeled_volume_batch_list[scale_id][0, 0:1, :, :, slice_start_list[scale_id]:slice_end_list[scale_id]:slice_interval_list[scale_id]].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=True)
                    writer.add_image('train/Image_scale{}'.format(scale_id), grid_image, iter_num)

                    # image = outputs_soft[0, 3:4, :, :, slice_start_list[scale_id]:slice_end_list[scale_id]:slice_interval_list[scale_id]].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                    image = pred_sup_1[0, :, :, slice_start_list[scale_id]:slice_end_list[scale_id]:slice_interval_list[scale_id]].permute(2, 0, 1).data.cpu().numpy()
                    image = utils.decode_seg_map_sequence(image)
                    grid_image = make_grid(image, 5, normalize=False)
                    writer.add_image('train/Predicted_label_scale{}_1'.format(scale_id), grid_image, iter_num)

                    image = pred_sup_2[0, :, :, slice_start_list[scale_id]:slice_end_list[scale_id]:slice_interval_list[scale_id]].permute(2, 0, 1).data.cpu().numpy()
                    image = utils.decode_seg_map_sequence(image)
                    grid_image = make_grid(image, 5, normalize=False)
                    writer.add_image('train/Predicted_label_scale{}_2'.format(scale_id), grid_image, iter_num)

                    image = pred_sup_3[0, :, :, slice_start_list[scale_id]:slice_end_list[scale_id]:slice_interval_list[scale_id]].permute(2, 0, 1).data.cpu().numpy()
                    image = utils.decode_seg_map_sequence(image)
                    grid_image = make_grid(image, 5, normalize=False)
                    writer.add_image('train/Predicted_label_scale{}_3'.format(scale_id), grid_image, iter_num)

                    image = label_label_batch_list[scale_id][0, :, :, slice_start_list[scale_id]:slice_end_list[scale_id]:slice_interval_list[scale_id]].permute(2, 0, 1)
                    grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                    writer.add_image('train/Groundtruth_label_scale{}'.format(scale_id), grid_image, iter_num)

                    image = unlabeled_volume_batch_list[scale_id][-1, 0:1, :, :, slice_start_list[scale_id]:slice_end_list[scale_id]:slice_interval_list[scale_id]].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=True)
                    writer.add_image('unlabel/Image_scale{}'.format(scale_id), grid_image, iter_num)

                    image = pred_unsup_1[-1, :, :, slice_start_list[scale_id]:slice_end_list[scale_id]:slice_interval_list[scale_id]].permute(2, 0, 1).data.cpu().numpy()
                    image = utils.decode_seg_map_sequence(image)
                    grid_image = make_grid(image, 5, normalize=False)
                    writer.add_image('unlabel/Predicted_label_scale{}_1'.format(scale_id), grid_image, iter_num)

                    image = pred_unsup_2[-1, :, :, slice_start_list[scale_id]:slice_end_list[scale_id]:slice_interval_list[scale_id]].permute(2, 0, 1).data.cpu().numpy()
                    image = utils.decode_seg_map_sequence(image)
                    grid_image = make_grid(image, 5, normalize=False)
                    writer.add_image('unlabel/Predicted_label_scale{}_2'.format(scale_id), grid_image, iter_num)

                    image = pred_unsup_3[-1, :, :, slice_start_list[scale_id]:slice_end_list[scale_id]:slice_interval_list[scale_id]].permute(2, 0, 1).data.cpu().numpy()
                    image = utils.decode_seg_map_sequence(image)
                    grid_image = make_grid(image, 5, normalize=False)
                    writer.add_image('unlabel/Predicted_label_scale{}_3'.format(scale_id), grid_image, iter_num)

                    image = unlabel_label_batch_list[scale_id][-1, :, :, slice_start_list[scale_id]:slice_end_list[scale_id]:slice_interval_list[scale_id]].permute(2, 0, 1)
                    grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                    writer.add_image('unlabel/Groundtruth_label_scale{}'.format(scale_id), grid_image, iter_num)

                image = unlabeled_volume_batch[-1, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('unlabel_source/Image', grid_image, iter_num)

                image = unlabel_label_batch[-1, :, :, 20:61:10].permute(2, 0, 1)
                grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                writer.add_image('unlabel_source/Groundtruth_label', grid_image, iter_num)




            logging.info(basic_info)
            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer_1.param_groups:
                    param_group['lr'] = lr_
                for param_group in optimizer_2.param_groups:
                    param_group['lr'] = lr_
                for param_group in optimizer_3.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save({'model': model.state_dict(),
                            'max_iterations': max_iterations}, save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    torch.save({'model': model.state_dict(),
                'max_iterations': max_iterations}, save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
