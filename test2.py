#!/usr/bin/env python

# Created on 2018/12
# Author: Kaituo XU

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from data import AudioDataLoader, AudioDataset, RandomDataLoader, RandomAudioDataset
from solver import Solver
from conv_tasnet import ConvTasNet
from encoder3decoder import Encoder3Decoder

parser = argparse.ArgumentParser(
    "Fully-Convolutional Time-domain Audio Separation Network (Conv-TasNet) "
    "with Permutation Invariant Training")
# General config
# Task related
parser.add_argument('--train_dir', type=str, default=None,
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--valid_dir', type=str, default=None,
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample rate')
parser.add_argument('--segment', default=4, type=float,
                    help='Segment length (seconds)')
parser.add_argument('--cv_maxlen', default=8, type=float,
                    help='max audio length (seconds) in cv, to avoid OOM issue.')
# Network architecture
parser.add_argument('--N', default=64, type=int,
                    help='Number of filters in autoencoder')
parser.add_argument('--L', default=6, type=int,
                    help='Length of the filters in samples (40=5ms at 8kHZ)')
parser.add_argument('--B', default=64, type=int,
                    help='Number of channels in bottleneck 1 × 1-conv block')
parser.add_argument('--H', default=6, type=int,
                    help='Number of channels in convolutional blocks')
parser.add_argument('--P', default=3, type=int,
                    help='Kernel size in convolutional blocks')
parser.add_argument('--X', default=6, type=int,
                    help='Number of convolutional blocks in each repeat')
parser.add_argument('--R', default=4, type=int,
                    help='Number of repeats')
parser.add_argument('--C', default=3, type=int,
                    help='Number of speakers')
parser.add_argument('--norm_type', default='gLN', type=str,
                    choices=['gLN', 'cLN', 'BN'], help='Layer norm type')
parser.add_argument('--causal', type=int, default=0,
                    help='Causal (1) or noncausal(0) training')
parser.add_argument('--mask_nonlinear', default='softmax', type=str,
                    choices=['relu', 'softmax'], help='non-linear to generate mask')
# Training config
parser.add_argument('--use_cuda', type=int, default=1,
                    help='Whether use GPU')
parser.add_argument('--epochs', default=30, type=int,
                    help='Number of maximum epochs')
parser.add_argument('--half_lr', dest='half_lr', default=1, type=int,
                    help='Halving learning rate when get small improvement')
parser.add_argument('--early_stop', dest='early_stop', default=1, type=int,
                    help='Early stop training when no improvement for 10 epochs')
parser.add_argument('--max_norm', default=5, type=float,
                    help='Gradient norm threshold to clip')
# minibatch
parser.add_argument('--shuffle', default=0, type=int,
                    help='reshuffle the data at every epoch')
parser.add_argument('--batch_size', default=64, type=int,
                    help='Batch size')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers to generate minibatch')
# optimizer
parser.add_argument('--optimizer', default='sgd', type=str,
                    choices=['sgd', 'adam'],
                    help='Optimizer (support sgd and adam now)')
parser.add_argument('--lr', default=0.3, type=float,
                    help='Init learning rate')
parser.add_argument('--momentum', default=0.4, type=float,
                    help='Momentum for optimizer')
parser.add_argument('--l2', default=0.00, type=float,
                    help='weight decay (L2 penalty)')
# save and load model
parser.add_argument('--save_folder', default='exp/temp',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=0, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue_from', default='',
                    help='Continue from checkpoint model')
parser.add_argument('--model_path', default='test2.pth.tar',
                    help='Location to save best validation model')
# logging
parser.add_argument('--print_freq', default=10, type=int,
                    help='Frequency of printing training infomation')
parser.add_argument('--visdom', dest='visdom', type=int, default=0,
                    help='Turn on visdom graphing')
parser.add_argument('--visdom_epoch', dest='visdom_epoch', type=int, default=0,
                    help='Turn on visdom graphing each epoch')
parser.add_argument('--visdom_id', default='TasNet training',
                    help='Identifier for visdom run')


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mixitem = np.load('/mnt/denoisedata/eegnoise_3mix_3.npy')
    eeglabels = np.load('/mnt/denoisedata/eeglabels_3mix_3.npy')

    X_test = mixitem[432:648]
    y_test = eeglabels[432:648]
    X_train = mixitem[648:]
    y_train = eeglabels[648:]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    print(X_train.shape)

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    batch_size = 200
    # 创建数据集和数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,drop_last=True)

    # Construct Solver
    # data
    # tr_dataset = AudioDataset(args.train_dir, args.batch_size,
    #                           sample_rate=args.sample_rate, segment=args.segment)
    # cv_dataset = AudioDataset(args.valid_dir, batch_size=1,  # 1 -> use less GPU memory to do cv
    #                           sample_rate=args.sample_rate,
    #                           segment=-1, cv_maxlen=args.cv_maxlen)  # -1 -> use full audio
    # tr_loader = AudioDataLoader(tr_dataset, batch_size=1,
    #                             shuffle=args.shuffle,
    #                             num_workers=args.num_workers)
    # cv_loader = AudioDataLoader(cv_dataset, batch_size=1,
    #                             num_workers=0)
    data = {'tr_loader': train_loader, 'cv_loader': test_loader}
    # model
    model = ConvTasNet(args.N, args.L, args.B, args.H, args.P, args.X, args.R,
                       args.C, norm_type=args.norm_type, causal=args.causal,
                       mask_nonlinear=args.mask_nonlinear)
    # model = Encoder3Decoder()
    print(model)
    if args.use_cuda:
        model = torch.nn.DataParallel(model)
        model.cuda()
    # optimizer
    if args.optimizer == 'sgd':
        optimizier = torch.optim.SGD(model.parameters(),
                                     lr=args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.l2)
    elif args.optimizer == 'adam':
        optimizier = torch.optim.Adam(model.parameters(),
                                      lr=args.lr,
                                      weight_decay=args.l2)
    else:
        print("Not support optimizer")
        return

    # solver
    solver = Solver(data, model, optimizier, args)
    solver.train()


if __name__ == '__main__':
    args = parser.parse_args()
    # args = parser.parse_args(['--N','64',
    #                           '--L','8',
    #                           '--B','4',
    #                           '--H','6',
    #                           '--P','3',
    #                           '--X','6',
    #                           '--R','4',
    #                           '--C','3',
    #                           '--norm_type','gLN',
    #                           '--causal','0',
    #                           '--mask_nonlinear','softmax',
    #                           '--use_cuda','1',
    #                           '--epochs','75',
    #                           '--half_lr','0',
    #                           '--early_stop','0',
    #                           '--max_norm','5',
    #                           '--shuffle','0',
    #                           '--batch_size','64',
    #                           '--num_workers','4',
    #                           '--optimizer','sgd',
    #                           '--lr','1e-3',
    #                           '--momentum','0.4',
    #                           '--l2','0.0',
    #                           '--save_folder','exp/temp',
    #                           '--checkpoint','0',
    #                           '--continue_from','',
    #                           '--model_path','test1.pth.tar',
    #                           '--print_freq','10',
    #                           '--visdom','0',
    #                           '--visdom_epoch','0',
    #                           '--visdom_id','TasNet training'
    #                           ])

    print(args)
    main(args)

