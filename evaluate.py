#!/usr/bin/env python

# Created on 2018/12
# Author: Kaituo XU

import argparse
import itertools
import os

import librosa
import matplotlib.pyplot as plt
from mir_eval.separation import bss_eval_sources
import numpy as np
import torch

from data import AudioDataLoader, AudioDataset
from pit_criterion import cal_loss
from conv_tasnet import ConvTasNet
from utils import remove_pad
from torch.utils.data import DataLoader, TensorDataset, random_split
from encoder3decoder import Encoder3Decoder


parser = argparse.ArgumentParser('Evaluate separation performance using Conv-TasNet')
parser.add_argument('--model_path', type=str, required=True,default='-',
                    help='Path to model file created by training')
parser.add_argument('--data_dir', type=str, required=True,default='-',
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--cal_sdr', type=int, default=0,
                    help='Whether calculate SDR, add this option because calculation of SDR is very slow')
parser.add_argument('--use_cuda', type=int, default=0,
                    help='Whether use GPU')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size')

def correlation_coef(x, y):
    vx = x - x.mean(dim=-1, keepdim=True)
    vy = y - y.mean(dim=-1, keepdim=True)
    vx_var = (vx * vx).sum(dim=-1)
    vy_var = (vy * vy).sum(dim=-1)

    # Add a small constant to avoid division by zero
    vx_var = vx_var + 1e-8
    vy_var = vy_var + 1e-8

    corr = (vx * vy).sum(dim=-1) / (torch.sqrt(vx_var) * torch.sqrt(vy_var))
    return corr

# def correlation_coef(x, y):
#     vx = x - np.mean
#     vy = y - y.mean(dim=-1, keepdim=True)
#     vx_var = (vx * vx).sum(dim=-1)
#     vy_var = (vy * vy).sum(dim=-1)
#
#     # Add a small constant to avoid division by zero
#     vx_var = vx_var + 1e-8
#     vy_var = vy_var + 1e-8
#
#     corr = (vx * vy).sum(dim=-1) / (torch.sqrt(vx_var) * torch.sqrt(vy_var))
#     return corr

def best_channel_permutation(mixture, source):
    B, C, H = mixture.size()

    # Function to compute the correlation coefficient

    best_corr_sum = -float('inf')
    best_corr = None

    # Iterate over all permutations of the channels
    for perm in itertools.permutations(range(C)):
        # Compute the correlation for this permutation
        corr_sum = 0
        current_corr = []
        for i, j in enumerate(perm):
            corr = correlation_coef(mixture[:, i, :], source[:, j, :]).abs()
            corr_sum += corr
            current_corr.append(corr)

        corr_sum = torch.abs(corr_sum)
        # Check if this is the best permutation so far
        if corr_sum.sum() > best_corr_sum:
            best_corr_sum = corr_sum.sum()
            best_corr = torch.stack(current_corr, dim=1)

    # The result should be the sum of the best correlation coefficients for each batch
    best_corr_sum_per_batch = best_corr.sum(dim=1, keepdim=True)
    # print(best_corr)
    return best_corr_sum_per_batch / 3


def evaluate(args,data_loader):
    total_SISNRi = 0
    total_SDRi = 0
    total_cnt = 0

    # Load model
    model = ConvTasNet.load_model(args.model_path)
    # model = Encoder3Decoder()
    print(model)
    model.eval()
    if args.use_cuda:
        model.cuda()
    cont = 0
    # Load data
    # dataset = AudioDataset(args.data_dir, args.batch_size,
    #                        sample_rate=args.sample_rate, segment=-1)
    # data_loader = AudioDataLoader(dataset, batch_size=1, num_workers=2)
    cc = []
    with (((torch.no_grad()))):
        for i, (data) in enumerate(data_loader):
            # Get batch data
            padded_mixture, padded_source = data
            mixture_lengths = torch.full((args.batch_size,), 2000)
            if args.use_cuda:
                padded_mixture = padded_mixture.cuda()
                mixture_lengths = mixture_lengths.cuda()
                padded_source = padded_source.cuda()
            # Forward
            estimate_source = model(padded_mixture)  # [B, C, T]
            # plt.plot(estimate_source.cpu().numpy()[0][1],linewidth=0.4)
            # plt.show()
            # plt.plot(estimate_source.cpu().numpy()[0][2],linewidth=0.4)
            # plt.show()

            # cc.append(torch.mean(best_channel_permutation(padded_source, estimate_source)).cpu().numpy())
            # print('cc,',np.corrcoef(estimate_source[0][0].cpu().numpy(),padded_source[0][0].cpu().numpy())[0,1])
            # print('cc,',torch.mean(best_channel_permutation(padded_source, estimate_source)).cpu().numpy())

            loss, max_snr, estimate_source, reorder_estimate_source = \
                cal_loss(padded_source, estimate_source, mixture_lengths)
            # Remove padding and flat
            mixture = remove_pad(padded_mixture, mixture_lengths)
            source = remove_pad(padded_source, mixture_lengths)
            # NOTE: use reorder estimate source
            estimate_source = remove_pad(reorder_estimate_source,
                                         mixture_lengths)
            est = np.array(estimate_source)
            mix = est[0][0]+est[0][1]+est[0][2]
            mean0 = np.mean(mix)
            maxval = np.max(np.abs(mix))
            std0 = np.std(mix)

            mix1 = -est[0][0] - np.mean(-est[0][0])
            mix1 = mix1 / np.std(-est[0][0])
            mix2 = -est[0][1] - np.mean(-est[0][1])
            mix2 = mix2 / np.max(np.abs(-est[0][1]))
            mix3 = -est[0][2] - np.mean(-est[0][2])
            mix3 = mix3 / np.max(np.abs(-est[0][2]))

            # mix1 = est[0][0] - mean0
            # mix1 = mix1 / maxval
            # mix2 = est[0][1] - mean0
            # mix2 = mix2 / maxval
            # mix3 = est[0][2] - mean0
            # mix3 = mix3 / maxval
            print('cc,',np.corrcoef(mix1,padded_source[0][0].cpu().numpy())[0,1])
            # print(np.mean(mix1))
            # print(np.mean(padded_source[0][0].cpu().detach().numpy()))
            nCC = np.corrcoef(mix1,padded_source[0][0].cpu().numpy())[0,1]
            # if nCC < 0:
            #     nCC = - nCC
            cc.append(nCC)

            # print(correlation_coef(torch.tensor(mix1).to(device),padded_source[0][0]))
            # print(np.corrcoef(mix1,padded_source[0][1].cpu().numpy())[0,1])
            # print(np.corrcoef(mix1,padded_source[0][2].cpu().numpy())[0,1])
            # print(np.corrcoef(mix2,padded_source[0][0].cpu().numpy())[0,1])
            # print(correlation_coef(torch.tensor(mix2).to(device),padded_source[0][1]))
            # print(np.corrcoef(mix2,padded_source[0][2].cpu().numpy())[0,1])
            # print(np.corrcoef(mix3,padded_source[0][0].cpu().numpy())[0,1])
            # print(np.corrcoef(mix3,padded_source[0][1].cpu().numpy())[0,1])
            # print(correlation_coef(torch.tensor(mix3).to(device),padded_source[0][2]))

            # if nCC<0:
            plt.plot(mix1, linewidth=0.4)
            plt.title('<0')
            plt.show()
            plt.plot(mix2,linewidth=0.4)
            plt.show()
            plt.plot(mix3,linewidth=0.4)
            plt.show()

            # plt.plot(mix3,linewidth=0.4)
            # plt.show()

            ori = padded_mixture[0].cpu().detach().numpy()
            ori = ori - np.mean(ori)
            ori = ori / np.max(np.abs(ori))

            estimate_source_tensor = torch.tensor(estimate_source).to(device)
            if nCC<0:
                cont += 1
                # plt.plot(padded_source[0][0].cpu().detach().numpy(),linewidth=0.4,color='red')
                # plt.show()
                # plt.plot(padded_source[0][1].cpu().detach().numpy(),linewidth=0.4,color='red')
                # plt.show()
                # plt.plot(padded_source[0][2].cpu().detach().numpy(),linewidth=0.4,color='red')
                # plt.show()
            plt.plot(padded_mixture[0].cpu().detach().numpy(),linewidth=0.4,color='orange')
            plt.show()

            # plt.plot(padded_source[0][2].cpu().detach().numpy(),linewidth=0.4,color='red')
            # plt.show()

            # for each utterance
            for mix, src_ref, src_est in zip(mixture, source, estimate_source):
                print("Utt", total_cnt + 1)
                # Compute SDRi
                if args.cal_sdr:
                    avg_SDRi = cal_SDRi(src_ref, src_est, mix)
                    total_SDRi += avg_SDRi
                    print("\tSDRi={0:.2f}".format(avg_SDRi))
                # Compute SI-SNRi
                avg_SISNRi = cal_SISNRi(src_ref, src_est, mix)
                print("\tSI-SNRi={0:.2f}".format(avg_SISNRi))
                total_SISNRi += avg_SISNRi
                total_cnt += 1
            print('count',cont)
    print("mean cc:",np.mean(np.array(cc)))
    if args.cal_sdr:
        print("Average SDR improvement: {0:.2f}".format(total_SDRi / total_cnt))
    print("Average SISNR improvement: {0:.2f}".format(total_SISNRi / total_cnt))


def cal_SDRi(src_ref, src_est, mix):
    """Calculate Source-to-Distortion Ratio improvement (SDRi).
    NOTE: bss_eval_sources is very very slow.
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SDRi
    """
    src_anchor = np.stack([mix, mix], axis=0)
    sdr, sir, sar, popt = bss_eval_sources(src_ref, src_est)
    sdr0, sir0, sar0, popt0 = bss_eval_sources(src_ref, src_anchor)
    avg_SDRi = ((sdr[0]-sdr0[0]) + (sdr[1]-sdr0[1])) / 2
    # print("SDRi1: {0:.2f}, SDRi2: {1:.2f}".format(sdr[0]-sdr0[0], sdr[1]-sdr0[1]))
    return avg_SDRi


def cal_SISNRi(src_ref, src_est, mix):
    """Calculate Scale-Invariant Source-to-Noise Ratio improvement (SI-SNRi)
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SISNRi
    """
    sisnr1 = cal_SISNR(src_ref[0], src_est[0])
    sisnr2 = cal_SISNR(src_ref[1], src_est[1])
    sisnr1b = cal_SISNR(src_ref[0], mix)
    sisnr2b = cal_SISNR(src_ref[1], mix)
    # print("SISNR base1 {0:.2f} SISNR base2 {1:.2f}, avg {2:.2f}".format(
    #     sisnr1b, sisnr2b, (sisnr1b+sisnr2b)/2))
    # print("SISNRi1: {0:.2f}, SISNRi2: {1:.2f}".format(sisnr1, sisnr2))
    avg_SISNRi = ((sisnr1 - sisnr1b) + (sisnr2 - sisnr2b)) / 2
    return avg_SISNRi


def cal_SISNR(ref_sig, out_sig, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        ref_sig: numpy.ndarray, [T]
        out_sig: numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(ref_sig) == len(out_sig)
    ref_sig = ref_sig - np.mean(ref_sig)
    out_sig = out_sig - np.mean(out_sig)
    ref_energy = np.sum(ref_sig ** 2) + eps
    proj = np.sum(ref_sig * out_sig) * ref_sig / ref_energy
    noise = out_sig - proj
    ratio = np.sum(proj ** 2) / (np.sum(noise ** 2) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    return sisnr


if __name__ == '__main__':
    args = parser.parse_args(['--model_path','exp/temp/test2.pth.tar',
                              '--data_dir','',
                              '--cal_sdr','0',
                              '--use_cuda','1',
                              '--batch_size','1'])
    mixitem = np.load('/mnt/denoisedata/eegnoise_3mix_3.npy')
    eeglabels = np.load('/mnt/denoisedata/eeglabels_3mix_3.npy')
    # eeglabels = eeglabels[:35,:,:]
    # eeglabels = np.zeros([35,3,2000])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    X_test = mixitem[2800:2825]
    y_test = eeglabels[2800:2825]

    # X_test = mixitem[:]
    # y_test = eeglabels[:]

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(args)
    evaluate(args,test_loader)
