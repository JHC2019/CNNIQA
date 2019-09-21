import os
import h5py
import torch
import numpy as np

from PIL import Image
from scipy.signal import convolve2d
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

def default_loader(path, flag):
    if flag == 1:
        return Image.open(path).convert('L')
    elif flag == 3:
        return Image.open(path).convert('RGB')

def get_data_loaders(config, train_batch_size, exp_id=0):
    train_dataset = IQADataset(config, exp_id, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=train_batch_size,
                                              shuffle=True,
                                              num_workers=0)

    val_dataset = IQADataset(config, exp_id, 'val')
    val_loader = torch.utils.data.DataLoader(val_dataset)

    if config['test_ratio']:
        test_dataset = IQADataset(config, exp_id, 'test')
        test_loader = torch.utils.data.DataLoader(test_dataset)

        return train_loader, val_loader, test_loader

    return train_loader, val_loader

class IQADataset(Dataset):
    def __init__(self, config, exp_id=0, status='train', loader=default_loader):
        self.loader = default_loader
        im_dir = config['im_dir']
        self.patch_size = config['patch_size']
        self.stride = config['stride']
        datainfo = config['datainfo']

        Info = h5py.File(datainfo, 'r')
        index = Info['index'][:, int(exp_id) % 1000]
        ref_ids = Info['ref_ids'][0, :]
        test_ratio = config['test_ratio']
        train_ratio = config['train_ratio']
        trainindex = index[:int(train_ratio * len(index))]
        testindex = index[int((1-test_ratio) * len(index)):]
        train_index, val_index, test_index = [],[],[]
        for i in range(len(ref_ids)):
            train_index.append(i) if (ref_ids[i] in trainindex) else \
                test_index.append(i) if (ref_ids[i] in testindex) else \
                    val_index.append(i)
        if status == 'train':
            self.index = train_index
            print("# Train Images: {}".format(len(self.index)))
            print('Ref Index:')
            print(trainindex)
        if status == 'test':
            self.index = test_index
            print("# Test Images: {}".format(len(self.index)))
            print('Ref Index:')
            print(testindex)
        if status == 'val':
            self.index = val_index
            print("# Val Images: {}".format(len(self.index)))

        self.mos = Info['subjective_scores'][0, self.index]
        self.mos_std = Info['subjective_scoresSTD'][0, self.index]
        im_names = [Info[Info['im_names'][0, :][i]][()].tobytes()\
                        [::2].decode() for i in self.index]

        self.patches = ()
        self.label = []
        self.label_std = []
        for idx in range(len(self.index)):
            # print("Preprocessing Image: {}".format(im_names[idx]))
            im = self.loader(os.path.join(im_dir, im_names[idx]), config['flag'])

            patches = NonOverlappingCropPatches(config, im, self.patch_size, self.stride)
            if status == 'train':
                self.patches = self.patches + patches
                for i in range(len(patches)):
                    self.label.append(self.mos[idx])
                    self.label_std.append(self.mos_std[idx])
            else:
                self.patches = self.patches + (torch.stack(patches), )
                self.label.append(self.mos[idx])
                self.label_std.append(self.mos_std[idx])

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return (self.patches[idx], (torch.Tensor([self.label[idx],]),
                torch.Tensor([self.label_std[idx],])))

def NonOverlappingCropPatches(config, im, patch_size=32, stride=32):
    w, h = im.size
    patches = ()
    for i in range(0, h - stride, stride):
        for j in range(0, w - stride, stride):
            patch = to_tensor(im.crop((j, i, j + patch_size, i + patch_size)))
            if config['is_norm']:
                patch = LocalNormalization(config, patch)
            patches = patches + (patch,)
    return patches

def LocalNormalization(config, patch, P=3, Q=3, C=1):
    kernel = np.ones((P, Q)) / (P * Q)
    if config['flag'] == 1:
        patch = patch[0].numpy()
        patch_mean = convolve2d(patch, kernel, boundary='symm', mode='same')
        patch_sm = convolve2d(np.square(patch), kernel, boundary='symm', mode='same')
        patch_std = np.sqrt(np.maximum(patch_sm - np.square(patch_mean), 0)) + C
        patch_ln = torch.from_numpy((patch - patch_mean) / patch_std).float().unsqueeze(0)
    elif config['flag'] == 3:
        patch = patch.numpy()
        tmp = np.zeros_like(patch)
        for i in range(3):
            patch_mean = convolve2d(patch[i], kernel, boundary='symm', mode='same')
            patch_sm = convolve2d(np.square(patch[i]), kernel, boundary='symm', mode='same')
            patch_std = np.sqrt(np.maximum(patch_sm - np.square(patch_mean), 0)) + C
            tmp[i] = (patch[i] - patch_mean) / patch_std
        patch_ln = torch.from_numpy(tmp).float().unsqueeze(0)
    return patch_ln
