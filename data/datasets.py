# from .config import HOME
import os
import sys
import torch
import torch.utils.data as data
import scipy.io as sio
import numpy as np
# from data_augmentation import *
from utils.data_augmentation import *


def concat_data(path_):
    path = path_ + '_'
    # data_ = np.load(path +'1.npy')
    print(path_)
    path = os.path.abspath(path)
    data_ = np.load(path+'1.npz',allow_pickle = True)['datas_10_fft']
    data_RC20 = np.load(path + '1.npz', allow_pickle=True)['datas_RC20']
    data_RC40 = np.load(path + '1.npz', allow_pickle=True)['datas_RC40']
    data_80 = np.load(path + '1.npz', allow_pickle=True)['datas_80_fft']
    i = 2
    while os.path.exists(path + str(i) + '.npz'):
        print('loading data '+path + str(i) + '.npz....')
        new_data = np.load(path + str(i) + '.npz', allow_pickle=True)['datas_10_fft']
        new_data_RC20 = np.load(path + str(i) + '.npz', allow_pickle=True)['datas_RC20']
        new_data_RC40 = np.load(path + str(i) + '.npz', allow_pickle=True)['datas_RC40']
        new_data_80 = np.load(path + str(i) + '.npz', allow_pickle=True)['datas_80_fft']
        data_ = np.concatenate((data_, new_data), axis=0)
        data_RC20 = np.concatenate((data_RC20, new_data_RC20), axis=0)
        data_RC40 = np.concatenate((data_RC40, new_data_RC40), axis=0)
        data_80 = np.concatenate((data_80, new_data_80), axis=0)
        i += 1
    return data_,data_80,data_RC20,data_RC40



def concat_label(path_):
    path = path_ + '_'
    # data_ = np.load(path +'1.npy')
    print(path_)
    path=os.path.abspath(path)
    data_ = np.load(path+'1.npz',allow_pickle = True)['labels']
    i = 2
    while os.path.exists(path + str(i) + '.npz'):
        print('loading data '+path + str(i) + '.npz....')
        new_data = np.load(path + str(i) + '.npz', allow_pickle=True)['labels']
        data_ = np.concatenate((data_, new_data), axis=0)
        i += 1
    return data_

class SignalDetectionv2(data.Dataset):
    def __init__(self, data_root, label_root, data_aug=False, dataset_name='SignalDetedtion'):
        self.data_root = data_root
        self.label_root = label_root
        self.data,self.data_80,self.dataRC20,self.dataRC40, self.labels = self.load_json()
        self.data_aug = data_aug
        self.dataset_name = dataset_name

    def __getitem__(self, idx):
        seq = np.array(self.data[idx])
        seq_80 = np.array(self.data_80[idx])
        seq_RC20 = np.array(self.dataRC20[idx])
        seq_RC40 = np.array(self.dataRC40[idx])
        seq_label = np.array(self.labels[idx])
        if self.data_aug:
            # max_value = np.max(seq, axis=(0, 1))
            roll = np.random.rand(1)
            if roll < 0.5:
                seq,seq_80,seq_RC20,seq_RC40, seq_label = sample_filplr(seq,seq_80,seq_RC20,seq_RC40, seq_label)
            seq,seq_80,seq_RC20,seq_RC40, seq_label = sample_jitter(seq,seq_80,seq_RC20,seq_RC40, seq_label)
            seq,seq_80,seq_RC20,seq_RC40, seq_label = sample_shift(seq,seq_80,seq_RC20,seq_RC40, seq_label)

        import scipy.io as scio
        # scio.savemat('1.mat',
        #              {'new_data': seq, 'new_dataRC20': seq_RC20, 'new_dataRC40': seq_RC40, 'label': seq_label})

        #转换为torch
        seq = torch.from_numpy(seq).type(torch.FloatTensor)
        seq_80 = torch.from_numpy(seq_80).type(torch.FloatTensor)
        seq_RC20 = torch.from_numpy(seq_RC20).type(torch.FloatTensor)
        seq_RC40 = torch.from_numpy(seq_RC40).type(torch.FloatTensor)

        # n x 3, n is the number of objects for each sequence
        labels = torch.from_numpy(seq_label).type(torch.FloatTensor).view(-1, 3)

        return seq ,seq_80,seq_RC20, seq_RC40, labels

    def __len__(self):
        return len(self.data)

    def load_json(self):
        dirs = os.listdir(self.label_root)
        for label_name in dirs:
            if label_name.endswith('.npz'):
                label = np.load(os.path.join(self.label_root, label_name), allow_pickle=True)['labels']
        dirs = os.listdir(self.data_root)
        for data_name in dirs:
            if data_name.endswith('.npz'):
                data = np.load(os.path.join(self.data_root, data_name), allow_pickle=True)['datas_10_fft']
                data_RC20 = np.load(os.path.join(self.data_root, data_name), allow_pickle=True)['datas_RC20']
                data_RC40 = np.load(os.path.join(self.data_root, data_name), allow_pickle=True)['datas_RC40']
                data_80 = np.load(os.path.join(self.data_root, data_name), allow_pickle=True)['datas_80_fft']
        return data,data_80,data_RC20,data_RC40, label

    def pull_seq(self, idx):
        """
        return m x 8192 np.array
        """
        return self.data[idx],self.data_80[idx],self.dataRC20[idx],self.dataRC40[idx]

    def pull_anno(self, idx):
        """
        return  n x 3 np.array
         """
        labels = np.reshape(self.labels[idx], [-1, 3])
        return str(idx), labels


def detection_collate(batch):

    targets = []
    imgs = []
    imgs_RC20 = []
    imgs_RC40 = []
    imgs_80 = []
    for sample in batch:
        imgs.append(sample[0])
        imgs_80.append(sample[1])
        imgs_RC20.append(sample[2])
        imgs_RC40.append(sample[3])
        targets.append(torch.FloatTensor(sample[4]))
    return torch.stack(imgs, 0), torch.stack(imgs_80, 0), torch.stack(imgs_RC20, 0), torch.stack(imgs_RC40, 0),targets


if __name__ == '__main__':
    data_set = SignalDetectionv2(r'C:\Users\华为\Desktop\Modulation_Detection\data\train\train_data',
                                 r'C:\Users\华为\Desktop\Modulation_Detection\data\train\train_label', True)

    data_loader = data.DataLoader(data_set, 10,
                                  num_workers=1, shuffle=True,
                                  collate_fn=detection_collate, pin_memory=True)
    batch_iterator = iter(data_loader)
    sample1, sample80,sample2 ,sample3 ,target1= next(batch_iterator)
    # sample1_, sample2_,target2 = next(batch_iterator)
    sample1= sample1.numpy()
    sample80 = sample80.numpy()
    sample2 = sample2.numpy()
    sample3 = sample3.numpy()


    # target1 = target1.numpy()

    import scipy.io as scio
    scio.savemat('fft_RC20_RC40.mat', {'sample1': sample1, 'sample2': sample2,'sample3': sample3,'sample80': sample80})
    print("wait")

