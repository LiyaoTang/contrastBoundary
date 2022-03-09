import numpy as np
import random
import SharedArray as SA

import torch

from util.voxelize import voxelize
from model.basic_operators import get_overlap


def sa_create(name, var):
    x = SA.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x


def collate_fn(batch):
    """
    Args:
        batch - [(xyz, feat, label, ...), ...] - each tuple from a sampler
    Returns: [
        xyz     : [BxN, 3]
        feat    : [BxN, d]
        label   : [BxN]
        ...
        offset  : int
    ]
    """
    batch_list = []
    for sample in batch:
        if isinstance(sample, list):
            batch_list += sample
        else:
            batch_list.append(sample)
    batch_list = list(zip(*batch_list))  # [[xyz, ...], [feat, ...], ...]

    offset, count = [], 0
    for item in batch_list[0]:
        count += item.shape[0]
        offset.append(count)
    offset = torch.IntTensor(offset)
    batch_list = [torch.cat(v) for v in batch_list]
    return [*batch_list, offset]


def data_prepare(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, origin='min'):
    """ coord, feat, label - an entire cloud
    """
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        # voxelize the entire cloud
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]

    if 'train' in split and voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0])
    else:
        # NOTE: not random during test
        init_idx = label.shape[0] // 2
    coord_init = coord[init_idx]

    if voxel_max and label.shape[0] > voxel_max:
        # radius crop with a random center point
        crop_idx = np.argsort(np.sum(np.square(coord - coord_init), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]
    xyz = coord

    if origin == 'min':
        coord_min = np.min(coord, 0)
        coord -= coord_min
    elif origin == 'mean':
        coord[..., :-1] -= coord[..., :-1].mean(0)
        coord[..., -1] -= coord[..., -1].min()
    elif origin == 'center':
        coord[..., :-1] -= coord_init[..., :-1]
        coord[..., -1] -= coord[..., -1].min()
    else:
        raise ValueError(f'not support origin={origin}')

    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    xyz = torch.FloatTensor(coord)
    return coord, feat, label, xyz


def data_prepare_block(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, sample_max=None, uniq_idx=None):
    """ coord, feat, label - an entire cloud
    """
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        # voxelize the entire cloud
        coord_min = np.min(coord, 0)
        coord -= coord_min
        if uniq_idx is None:
            uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]

    if voxel_max and label.shape[0] > voxel_max:
        # radius crop with a center point - till full coverage
        coord_list, feat_list, label_list, xyz_list = [], [], [], []
        potentials = np.random.rand(label.shape[0]) * 1e-3
        crop_i = 0

        # while crop_i < sample_max:
        while crop_i < sample_max and ((potentials > 1).sum() < voxel_max * 3 / 4 or crop_i < 2):
            # choose from not-covered
            init_idx = np.argmin(potentials)
            dist = np.sum(np.square(coord - coord[init_idx]), 1)
            crop_idx = np.argsort(dist)[:voxel_max]

            if shuffle_index:  # shuffle
                np.random.shuffle(crop_idx)

            # convert to input sample
            xyz, f, l = coord[crop_idx], feat[crop_idx], label[crop_idx]

            # store
            xyz_list += [torch.FloatTensor(xyz)]  # xyz in cloud
            xyz = xyz - np.min(xyz, 0)  # xyz in sample
            xyz = torch.FloatTensor(xyz)
            coord_list.append(xyz)
            f = torch.FloatTensor(f) / 255.  # rgb
            feat_list.append(f)
            l = torch.LongTensor(l)  # label
            label_list.append(l)

            # update & next
            # dist = dist[crop_idx]
            # weights = np.square(1 - dist / dist.max())
            weights = 1
            potentials[crop_idx] += weights
            crop_i += 1

        # if split == 'train':
        #     print(label.shape, label.shape[0] - voxel_max, '\t=>', len(label_list), [l.shape for l in label_list], 'sample_max', sample_max, flush=True)
        coord, feat, label, xyz = coord_list, feat_list, label_list, xyz_list

    else:
        # full cloud as input
        if shuffle_index:
            shuf_idx = np.arange(coord.shape[0])
            np.random.shuffle(shuf_idx)
            coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

        coord_min = np.min(coord, 0)
        coord -= coord_min
        coord = [torch.FloatTensor(coord)]
        feat = [torch.FloatTensor(feat) / 255.]
        label = [torch.LongTensor(label)]
        xyz = [torch.FloatTensor(np.zeros((0, 3)))]

    # to be compound in 'data_gen'
    return coord, feat, label, xyz

