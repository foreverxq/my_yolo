# from .config import HOME
import os
import sys
import torch
import torch.utils.data as data
import scipy.io as sio
import numpy as np
# from data_augmentation import *
from utils.data_augmentation import *




class SignalDetection(data.Dataset):
    def __init__(self, data_root, label_root, data_aug=False, dataset_name='SignalDetedtion'):
        self.data_root = data_root
        self.label_root = label_root
        ##这里是否可以一次加载多个npz文件 之后每次读取一个batchsize大小的文件 否则每次只能对一个npz进行训练  todo 写出使用yield读取多个npz文件的函数

        self.data,self.data_80,self.dataRC20,self.dataRC40, self.labels = self.load_data_label()
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
        #转换为tensor
        seq = torch.from_numpy(seq).type(torch.FloatTensor)
        seq_80 = torch.from_numpy(seq_80).type(torch.FloatTensor)
        seq_RC20 = torch.from_numpy(seq_RC20).type(torch.FloatTensor)
        seq_RC40 = torch.from_numpy(seq_RC40).type(torch.FloatTensor)
        # n x 3, n is the number of objects for each sequence
        labels = torch.from_numpy(seq_label).type(torch.FloatTensor).view(-1, 3)
        return seq ,seq_80,seq_RC20, seq_RC40, labels

    def __len__(self):
        return len(self.data)

    def load_data_label(self):
        dirs = os.listdir(self.label_root)
        for label_name in dirs:
            if label_name.endswith('.npz'):
                label = np.load(os.path.join(self.label_root, label_name), allow_pickle=True)['labels']
        dirs = os.listdir(self.data_root)
        for data_name in dirs:
            if data_name.endswith('.npz'):
                temp_data = np.load(os.path.join(self.data_root, data_name), allow_pickle=True)
                data = temp_data['datas_10_fft']
                data_RC20 = temp_data['datas_RC20']
                data_RC40 = temp_data['datas_RC40']
                data_80 = temp_data['datas_80_fft']
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
    targets, imgs, imgs_RC20, imgs_RC40, imgs_80 = [],[],[],[],[]
    for sample in batch:
        imgs.append(sample[0])
        imgs_80.append(sample[1])
        imgs_RC20.append(sample[2])
        imgs_RC40.append(sample[3])
        targets.append(torch.FloatTensor(sample[4]))
    return torch.stack(imgs, 0), torch.stack(imgs_80, 0), torch.stack(imgs_RC20, 0), torch.stack(imgs_RC40, 0),targets


if __name__ == '__main__':
    data_set = SignalDetection('../train_data',
                                 '../train_label', True)

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

