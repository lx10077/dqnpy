import torch
import os
import numpy as np


USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
DOUBLE = torch.cuda.DoubleTensor if USE_CUDA else torch.DoubleTensor
LONG = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor

TYPE_LIST = {"FLOAT": (np.float32, FLOAT),
             "DOUBLE": (np.float64, DOUBLE),
             "LONG": (np.int64, LONG)}


def to_numpy(ten):
    return ten.cpu().data.numpy() if USE_CUDA else ten.data.numpy()


def to_tensor(ndarray, requires_grad=False, dtype="FLOAT"):
    ndarray = ndarray.astype(TYPE_LIST[dtype][0])
    ten = torch.from_numpy(ndarray)
    ten.requires_grad = requires_grad
    ten.type(TYPE_LIST[dtype][1])
    return ten.cuda() if USE_CUDA else ten


def train_dir(exp_name: str):
    def make_soft_link(base_path, soft_path):

        if not os.path.exists(soft_path):
            os.system('ln -s {} {}'.format(base_path, soft_path))
        else:
            os.system('rm {}'.format(soft_path))
            os.system('ln -s {} {}'.format(base_path, soft_path))

    def make_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    abs_path = soft_path = os.path.abspath(__file__)
    try:
        count = 0
        project_name = current = abs_path.split(os.path.sep)[-1]
        while count < 50 and current != "config":
            project_name = current
            abs_path = os.path.dirname(abs_path)
            current = abs_path.split(os.path.sep)[-1]
            count += 1
            if count >= 50:
                raise ValueError("Make sure your codes lie in a CONFIG file!")
    except Exception as e:
        raise Exception(e)

    abs_path = os.path.abspath(os.path.join(abs_path, '../train_log/{}/{}'.format(project_name, exp_name)))
    soft_path = os.path.abspath(os.path.join(soft_path, '../train_log/'))
    make_dir(abs_path)
    make_soft_link(abs_path, soft_path)

    return abs_path
