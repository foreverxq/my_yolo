import numpy as np
import torch
# from data_plot import draw_plot


def sample_filplr(data,data_80,data_RC20, data_RC40, label):
    # data = np.array(data)
    # label = np.array(label)
    data = data[:, ::-1]
    data_80 = data_80[:, ::-1]
    data_RC20 = data_RC20[:, ::-1]
    data_RC40 = data_RC40[:, ::-1]
    label = label.reshape(-1, 3)
    len = label.shape[0]
    for i in range(len):
        label[:, i * 3: i * 3 + 2] = 1.0 - label[:, i * 3: i * 3 + 2]
        label[:, i * 3: i * 3 + 2] = label[:, i * 3: i * 3 + 2][:, ::-1]

    return data.copy(),data_80.copy(),data_RC20.copy(),data_RC40.copy(), label


def sample_shift(data,data_80,data_RC20, data_RC40,label):
    # data = np.array(data)
    # label = np.array(label)
    label = label.reshape(-1, 3)
    len = label.shape[0]
    sample_len = data.shape[1]
    min_value = 1
    for i in range(len):
       min_value = min(min_value, label[i][0])
    shift_ = np.random.uniform(0, 1, 1) * max(min_value, 0)

    shift = int(shift_ * sample_len)
    new_data = data.copy()
    new_data[:, :(sample_len - shift)] = data[:, shift:]
    new_data[:, sample_len - shift:] = data[:, :shift]

    sample_lenRC20 = data_RC20.shape[1]
    shiftRC20 = int(shift_ * sample_lenRC20)
    new_dataRC20 = data_RC20.copy()
    new_dataRC20[:, :(sample_lenRC20 - shiftRC20)] = data_RC20[:, shiftRC20:]
    new_dataRC20[:, sample_lenRC20 - shiftRC20:] = data_RC20[:, :shiftRC20]

    sample_lenRC40 = data_RC40.shape[1]
    shiftRC40 = int(shift_ * sample_lenRC40)
    new_dataRC40 = data_RC40.copy()
    new_dataRC40[:, :(sample_lenRC40 - shiftRC40)] = data_RC40[:, shiftRC40:]
    new_dataRC40[:, sample_lenRC40 - shiftRC40:] = data_RC40[:, :shiftRC40]

    sample_len80 = data_80.shape[1]
    shift80 = int(shift_ * sample_len80)
    new_data80 = data_80.copy()
    new_data80[:, :(sample_len80 - shift80)] = data_80[:, shift80:]
    new_data80[:, sample_len80 - shift80:] = data_80[:, :shift80]

    for i in range(len):
        label[:, i * 3: i * 3 + 2] = label[:, i * 3: i * 3 + 2] - shift_

    # import scipy.io as scio
    # scio.savemat('1.mat', {'new_data': new_data, 'new_dataRC20': new_dataRC20,'new_dataRC40': new_dataRC40,'label':label})
    return new_data,new_data80,new_dataRC20, new_dataRC40,label


def sample_jitter(data,data_80,data_RC20,data_RC40, sample):
    factor = np.random.uniform(0, 1, 1) * 0.2 + 0.9
    data = data * factor

    factorC20 = np.random.uniform(0, 1, 1) * 0.2 + 0.9
    data_RC20 = data_RC20 * factorC20

    factorall = np.random.uniform(0, 1, 1) * 0.2 + 0.9
    data_RC40 = data_RC40 * factorall

    factor80 = np.random.uniform(0, 1, 1) * 0.2 + 0.9
    data_80 = data_80 * factor80

    return data,data_80,data_RC20,data_RC40, sample
