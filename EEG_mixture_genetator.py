import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
import math
import torch
import pywt
from torchcrf import CRF
from scipy.io import loadmat
from scipy.signal import butter, lfilter, filtfilt,resample
from sklearn.preprocessing import MaxAbsScaler
import matplotlib.pyplot as plt
from scipy.fft import fft
import random
from torch.utils.data import DataLoader, TensorDataset, random_split

# 数据生成

fs = 200

# FFT函数
def my_fft(x, fs):
    fft_x = fft(x)  # fft计算
    amp_x = 2 * abs(fft_x) / len(x)  # 纵坐标变换
    label_x = np.linspace(0, int(len(x) / 2) - 1, int(len(x) / 2))  # 生成频率坐标
    amp = amp_x[0:int(len(x) / 2)]  # 选取前半段计算结果即可
    # amp[0] = 0
    fre = label_x / len(x) * fs  # 频率坐标变换
    pha = np.unwrap(np.angle(fft_x))  # 计算相位角并去除2pi跃变
    return amp, fre, pha  # 返回幅度和频率
    # plt.plot(fre,amp)


def generate_unique_random_numbers(N, M):
    # 检查输入是否有效
    if N > M:
        raise ValueError("N 不能大于 M")

    # 生成N个不重复的正整数，范围在1到M之间
    random_numbers = random.sample(range(1, M + 1), N)

    return random_numbers


filename = os.path.join(r'E:\dataset\眼电_脑电去噪数据集 semi-simulated\semi-simulated','Pure_Data.mat')
EEG = loadmat(filename)
filename = os.path.join(r'E:\dataset\眼电_脑电去噪数据集 semi-simulated\semi-simulated','VEOG.mat')
VEOG = loadmat(filename)


def waveget(sig):
    # 进行连续小波变换
    scales = np.arange(1, 64)
    waveletname = 'cmor'
    [cwtmatr, frequencies] = pywt.cwt(sig, scales, waveletname, 1.0)
    power = (abs(cwtmatr)) ** 2
    # plt.imshow(abs(cwtmatr), extent=[0, 1, 1, 128], aspect='auto',
    #            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())  # 绘制时频图
    # plt.show()
    #
    # wavepw = []
    # for i in range(len(sig)):
    #     sumer = sum(power[:, i])
    #     wavepw.append(float(sumer))
    # print(cwtmatr.shape)
    return power


class UNet3(nn.Module):
    def __init__(self):
        super(UNet3, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv7 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.conv8 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.up1 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.conv9 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.conv10 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.up2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.conv11 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.conv12 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.up3 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.conv13 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.conv14 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.conv15 = nn.Conv1d(64, 1, kernel_size=1)

    def forward(self, x):
        conv1 = nn.functional.relu(self.conv1(x))
        conv1 = nn.functional.relu(self.conv2(conv1))
        pool1 = self.pool1(conv1)
        conv2 = nn.functional.relu(self.conv3(pool1))
        conv2 = nn.functional.relu(self.conv4(conv2))

        # pool2 = self.pool2(conv2)
        # conv3 = nn.functional.relu(self.conv5(pool2))
        # conv3 = nn.functional.relu(self.conv6(conv3))
        # pool3 = self.pool3(conv3)
        # conv4 = nn.functional.relu(self.conv7(pool3))
        # conv4 = nn.functional.relu(self.conv8(conv4))
        # up1 = self.up1(conv4)
        # up1 = torch.cat([up1, conv3], dim=1)
        # conv5 = nn.functional.relu(self.conv9(up1))
        # conv5 = nn.functional.relu(self.conv10(conv5))
        # up2 = self.up2(conv5)
        # up2 = torch.cat([up2, conv2], dim=1)
        # conv6 = nn.functional.relu(self.conv11(up2))
        # conv6 = nn.functional.relu(self.conv12(conv6))

        # up3 = self.up3(conv6)
        up3 = self.up3(conv2)

        up3 = torch.cat([up3, conv1], dim=1)
        conv7 = nn.functional.relu(self.conv13(up3))
        conv7 = nn.functional.relu(self.conv14(conv7))
        out = self.conv15(conv7)
        return out


class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, encoded_space_dim, output_dim):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            # nn.Linear(512, 256),
            # nn.ReLU(True),
            nn.Linear(256, encoded_space_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoded_space_dim, 256),
            nn.ReLU(True),
            # nn.Linear(256, 512),
            # nn.ReLU(True),
            nn.Linear(256, output_dim),
            # nn.Sigmoid()  # Assuming the input is scaled between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded




def sigseg(sig, second):
    res = []
    start = 0
    while start + (second * fs) <= len(sig):
        res.append(sig[start:start + second * fs])
        start += second * fs
    return np.array(res)


import random

# 创建一个包含1到100的列表
numbers = list(range(1, 5832 + 1))

# 使用shuffle函数打乱列表顺序
random.shuffle(numbers)
# 没有EOG的序号
zeroindex = numbers[:200]
# 有1段
oneindex = numbers[200:4200]
# 有2段
twoindex = numbers[4200:]

def inter_to_mask(count, inters, length):
    mask = np.zeros(length)
    if count in zeroindex:
        return mask
    elif count in twoindex:
        for inter in inters:
            mask[inter[0]:inter[1]] = 1
        return mask

    elif count in oneindex:
        mask[inters[0]:inters[1]] = 1
        return mask
    else:
        print('err')

def resample_sig(x, fs, fs_target):
    t = np.arange(x.shape[0]).astype("float64")

    if fs == fs_target:
        return x, t

    new_length = int(x.shape[0] * fs_target / fs)
    # Resample the array if NaN values are present
    if np.isnan(x).any():
        x = pd.Series(x.reshape((-1,))).interpolate().values
    resampled_x, resampled_t = resample(x, num=new_length, t=t)
    assert (
            resampled_x.shape == resampled_t.shape
            and resampled_x.shape[0] == new_length
    )
    assert np.all(np.diff(resampled_t) > 0)

    return resampled_x, resampled_t

def get_rms(records):
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))


np.random.seed(365)

second = 20
snr = -5
count = 0
mixitem = []
labels = []
eeglabels = []

EMG = np.load('E:\\dataset\\wavelet-quantile-normalization-master\\data\\eeg-denoise-net\\EMG_all_epochs.npy')

for ii in range(1, 55):
    for kk in range(1, 55):
        for jj in [0, 1]:
            EEGsig = EEG[f'sim{ii}_resampled'][jj]
            # eegseg = sigseg(EEGsig,second)
            eegseg = EEGsig[1000:(second * fs + 1000)]
            eegseg = eegseg - np.mean(eegseg)
            eegnoise = eegseg.copy()

            # 创建一个1x4000的全零数组,然后区间是1或2，代表不同噪声
            mask1 = np.zeros(fs*second)
            # 随机生成2到4个区间
            num_intervals = np.random.randint(2, 5)
            # 记录已经使用的索引
            used_indices = []
            # 区间
            intervals = []
            for vas in range(num_intervals):
                # 生成随机的起始索引和长度
                start_index = np.random.randint(0, fs*(second-2))
                length = np.random.randint(0.5*fs, 1.5*fs)
                # 确保区间不重叠
                while any(start_index <= idx <= start_index + length for idx in used_indices):
                    start_index = np.random.randint(0, fs*(second-2))
                # 记录已使用的索引
                used_indices.extend(range(start_index, start_index + length))
                # 生成随机的区间内的值
                value = np.random.randint(1, 3)
                # 将生成的值赋给数组的相应位置
                mask1[start_index:start_index + length] = value
                intervals.append([start_index,start_index+length,value])
            for inters in intervals:
                if inters[2] == 1:
                    k = random.randint(1, 54)
                    EOGsig = VEOG[f'veog_{k}'][0]
                    eogseg = EOGsig[1000:(second * fs + 1000)]
                    SNR_dB = random.randint(-10, -1)
                    SNR = 10 ** (0.1 * (SNR_dB))
                    noise = np.zeros_like(eegseg)
                    noise[inters[0]:inters[1]] = eogseg[inters[0]:inters[1]] - np.mean(eogseg[inters[0]:inters[1]])
                    coe = get_rms(eegseg[inters[0]:inters[1]]) / (get_rms(noise[inters[0]:inters[1]]) * np.sqrt(SNR))
                    noise[inters[0]:inters[1]] = noise[inters[0]:inters[1]] * coe
                    eegnoise = noise + eegnoise

                elif inters[2] == 2:
                    k = random.randint(0, 5597)
                    emgseg = EMG[k]
                    emgseg, _ = resample_sig(emgseg, 256, 200)
                    start = random.randint(0, 100)
                    end = start+inters[1]-inters[0]
                    SNR_dB = random.randint(-10, -1)
                    SNR = 10 ** (0.1 * (SNR_dB))
                    noise = np.zeros_like(eegseg)
                    noise[inters[0]:inters[1]] = emgseg[start:end] - np.mean(emgseg[start:end])
                    coe = get_rms(eegseg[inters[0]:inters[1]]) / (get_rms(noise[inters[0]:inters[1]]) * np.sqrt(SNR))
                    noise[inters[0]:inters[1]] = noise[inters[0]:inters[1]] * coe
                    eegnoise = noise + eegnoise
            std1 = np.std(eegnoise)
            eegnoise = eegnoise / np.std(eegnoise)
            eegseg = eegseg / std1
            # plt.plot(eegseg,linewidth=0.4)
            # plt.plot(eegnoise,linewidth=0.4)
            # plt.plot(mask1*np.max(eegnoise)/2,linewidth=0.4)
            # plt.show()
            labels.append(mask1)
            mixitem.append(eegnoise)
            eeglabels.append(eegseg)


# np.save('E:\\denoisedata\\labels.npy',labels)
# np.save('E:\\denoisedata\\eegnoise.npy',mixitem)
# np.save('E:\\denoisedata\\eeglabels.npy',eeglabels)
#
plt.plot()
plt.show()
