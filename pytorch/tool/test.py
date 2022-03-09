import os, sys
import time
import random
import numpy as np
import logging
import pickle
import argparse
import collections

DIR_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.dirname(DIR_PATH)
sys.path.insert(0, DIR_PATH)
sys.path.insert(0, ROOT_PATH)

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

# from sklearn.metrics import confusion_matrix
# conf = confusion_matrix(label[mask], pred[mask], labels=np.arange(config.num_classes))

from util import config
from util.common_util import AverageMeter, intersectionAndUnion, check_makedirs
from util.voxelize import voxelize
from util.logger import *
from model.basic_operators import _eps, _inf
from model.utils import *
from lib.pointops.functions import pointops

random.seed(123)
np.random.seed(123)

def _try_eval(s):
    try:
        s = eval(s)
    except:
        pass
    return s

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/s3dis/s3dis_pointtransformer_repro.yaml', help='config file')
    parser.add_argument('--set', type=str, default=None, help='command line setting k-v tuples')
    parser.add_argument('opts', help='see config/s3dis/s3dis_pointtransformer_repro.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    if args.set:
        for kv in args.set.split(';'):
            if kv.startswith('{') and kv.endswith('}'):
                kv = eval(kv)
            else:
                kv = dict([[i.strip() for i in t.split(':')] for t in kv.split(',')])
                kv = {k: _try_eval(v) for k, v in kv.items()}
            for k, v in kv.items():
                setattr(cfg, k, v)
    if not cfg.save_path:
        cfg_dir, cfg_yaml = args.config.split(os.sep)[-2:]
        cfg.save_path = os.path.join('results', cfg_dir, '.'.join(cfg_yaml.split('.')[:-1]))  # results / s3dis / origin
    if not cfg.save_folder:
        cfg.save_folder = os.path.join(cfg.save_path, 'result')

    print_dict(cfg, head='>>> ======== config ======== >>>')
    cfg = config.CfgNode(cfg, default='')
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main():
    global args, logger
    args = get_parser()
    logger = get_logger()
    logger.info(args)
    assert args.classes > 1
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    if args.arch == 'pointtransformer_seg_repro':
        from model.pointtransformer_seg import pointtransformer_seg_repro as Model
    else:
        raise Exception('architecture not supported yet'.format(args.arch))
    model = Model(c=args.fea_dim, k=args.classes, config=args).cuda()
    logger.info(model)
    from model.pointtransformer_seg import Loss
    # criterion = Loss(config=args).cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    names = [line.rstrip('\n') for line in open(args.names_path)]
    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        state_dict = checkpoint['state_dict']
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
        args.epoch = checkpoint['epoch']
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
    test(model, criterion, names)


def data_prepare():
    if args.data_name == 's3dis':
        data_list = sorted(os.listdir(args.data_root))
        data_list = [item[:-4] for item in data_list if 'Area_{}'.format(args.test_area) in item]
    else:
        raise Exception('dataset not supported yet'.format(args.data_name))
    print("Totally {} samples in val set.".format(len(data_list)))
    return data_list


def data_load(data_name):
    # load the whole cloud - coord, feat, label
    # idx_data - enmerate over pts selection in voxel
    data_path = os.path.join(args.data_root, data_name + '.npy')
    data = np.load(data_path)  # xyzrgbl, N*7
    coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]

    idx_data = []
    if args.voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        idx_sort, count = voxelize(coord, args.voxel_size, mode=1)  # sorted idx & cnts of entire cloud
        for i in range(count.max()):
            # enumerate through the max num of pts in the same voxel - [N_vx]
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
            idx_part = idx_sort[idx_select]
            idx_data.append(idx_part)
    else:
        idx_data.append(np.arange(label.shape[0]))
    # [N, 3], [N, 3], [N], [[N_vx], ...]
    return coord, feat, label, idx_data


def input_normalize(coord, feat):
    coord_min = np.min(coord, 0)
    coord -= coord_min
    feat = feat / 255.
    return coord, feat


def test(model, criterion, names):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    intersection_meter = AverageMeter()  # accumulating over clouds
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    args.batch_size_test = 10
    model.eval()
    config = model.config

    check_makedirs(args.save_folder)
    pred_save, label_save = [], []
    cum_dict_list = TorchList([])  # list of cum_dict - cloud idx -> cumulated result
    data_list = data_prepare()
    for idx, item in enumerate(data_list):
        end = time.time()
        pred_save_path = os.path.join(args.save_folder, '{}_{}_pred.npy'.format(item, args.epoch))
        label_save_path = os.path.join(args.save_folder, '{}_{}_label.npy'.format(item, args.epoch))
        if os.path.isfile(pred_save_path) and os.path.isfile(label_save_path):
            logger.info('{}/{}: {}, loaded pred and label.'.format(idx + 1, len(data_list), item))
            pred, label = np.load(pred_save_path), np.load(label_save_path)
            coord = None
            cum_dict = init_cumulate_dict(label.size, model.config).numpy()
            cum_dict['probs'][np.arange(label.size), pred] = 1
        else:
            coord, feat, label, idx_data = data_load(item)
            cum_dict = init_cumulate_dict(label.size, model.config, device='cuda')
            pred = cum_dict['probs']
            idx_size = len(idx_data)

            # prepare into multiple input sample(s)
            idx_list, coord_list, feat_list, offset_list  = [], [], [], []
            for i in range(idx_size):  # for each enmeration of the full (potentially voxelized) cloud
                logger.info('{}/{}: {}/{}/{}, {}'.format(idx + 1, len(data_list), i + 1, idx_size, idx_data[0].shape[0], item))
                idx_part = idx_data[i]

                # select points from full cloud into voxels
                coord_part, feat_part = coord[idx_part], feat[idx_part]

                if args.voxel_max and coord_part.shape[0] > args.voxel_max:
                    # if exceeding test/val voxel num - spatially regular gen
                    coord_p, idx_uni, cnt = np.random.rand(coord_part.shape[0]) * 1e-3, np.array([]), 0
                    # until all voxels of current cloud covered
                    while idx_uni.size != idx_part.shape[0]:
                        # select center (min potentials)
                        init_idx = np.argmin(coord_p)
                        # crop by distance
                        dist = np.sum(np.power(coord_part - coord_part[init_idx], 2), 1)
                        idx_crop = np.argsort(dist)[:args.voxel_max]
                        coord_sub, feat_sub, idx_sub = coord_part[idx_crop], feat_part[idx_crop], idx_part[idx_crop]
                        # update potentials
                        dist = dist[idx_crop]
                        delta = np.square(1 - dist / np.max(dist))
                        coord_p[idx_crop] += delta
                        # prepare cropped input into baches - centralized, 0-1 rgb
                        coord_sub, feat_sub = input_normalize(coord_sub, feat_sub)
                        # finish current input sample
                        idx_list.append(idx_sub), coord_list.append(coord_sub), feat_list.append(feat_sub), offset_list.append(idx_sub.size)
                        # check newly covered voxels
                        idx_uni = np.unique(np.concatenate((idx_uni, idx_sub)))

                else:
                    # direct processing the full voxels in one sample
                    coord_part, feat_part = input_normalize(coord_part, feat_part)
                    idx_list.append(idx_part), coord_list.append(coord_part), feat_list.append(feat_part), offset_list.append(idx_part.size)

            batch_num = int(np.ceil(len(idx_list) / args.batch_size_test))
            for i in range(batch_num):
                # consrtuct batch from multiple input samples
                s_i, e_i = i * args.batch_size_test, min((i + 1) * args.batch_size_test, len(idx_list))
                idx_part, coord_part, feat_part, offset_part = idx_list[s_i:e_i], coord_list[s_i:e_i], feat_list[s_i:e_i], offset_list[s_i:e_i]
                idx_part = np.concatenate(idx_part)
                coord_part = torch.FloatTensor(np.concatenate(coord_part)).cuda(non_blocking=True)  # [BxN, 3]
                feat_part = torch.FloatTensor(np.concatenate(feat_part)).cuda(non_blocking=True)    # [BxN, d=3]
                offset_part = torch.IntTensor(np.cumsum(offset_part)).cuda(non_blocking=True)       # [BxN]
                with torch.no_grad():
                    inputs = {'points': coord_part, 'features': feat_part, 'offset': offset_part}
                    pred_part, _ = model(inputs)  # (n, k)
                torch.cuda.empty_cache()
                # accumulate logits
                cumulate_probs(cum_dict, pred_part, inds=idx_part)
                logger.info('Test: {}/{}, {}/{}, {}/{}'.format(idx + 1, len(data_list), e_i, len(idx_list), args.voxel_max, idx_part.shape[0]))

            loss = criterion(pred, torch.LongTensor(label).cuda(non_blocking=True))  # for reference
            pred = pred.max(1)[1].data.cpu().numpy()  # pred = argmx of logits (dim=1)
            cum_dict = cum_dict.numpy()  # to cpu

        # accumulate per-cloud results
        if any(k in ['neighbors'] for k in cum_dict) and coord is None:
            data = np.load(os.path.join(args.data_root, f'{item}.npy'))  # xyzrgbl, N*7
            coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]

        if 'neighbors' in cum_dict:
            for kr in cum_dict['neighbors']:
                xyz = torch.from_numpy(coord.astype(np.float32)).contiguous().cuda()
                offset = torch.IntTensor([len(coord)]).cuda()
                neighbor_idx, _ = pointops.knnquery(kr, xyz, xyz, offset, offset)  # (m, nsample)
                cum_dict['neighbors'][kr] = neighbor_idx.data.cpu().numpy()  # cleanup cuda
                del neighbor_idx
                torch.cuda.empty_cache()

        # store cur results
        cum_dict_list += [cum_dict]

        # calculation 1: add per room predictions
        intersection, union, target = intersectionAndUnion(pred, label, args.classes, args.ignore_label)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)

        accuracy = sum(intersection) / (sum(target) + 1e-10)
        batch_time.update(time.time() - end)
        logger.info('Test: [{}/{}]-{} '
                    'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Accuracy {accuracy:.4f}.'.format(idx + 1, len(data_list), label.size, batch_time=batch_time, accuracy=accuracy))
        pred_save.append(pred); label_save.append(label)
        np.save(pred_save_path, pred); np.save(label_save_path, label)

    with open(os.path.join(args.save_folder, "pred.pickle"), 'wb') as handle:
        pickle.dump({'pred': pred_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.save_folder, "label.pickle"), 'wb') as handle:
        pickle.dump({'label': label_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # calculation 1 - using intersection-union accumulated over clouds
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU1 = np.mean(iou_class)
    mAcc1 = np.mean(accuracy_class)
    allAcc1 = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    # calculation 2 - using prediction of all clouds
    intersection, union, target = intersectionAndUnion(np.concatenate(pred_save), np.concatenate(label_save), args.classes, args.ignore_label)
    iou_class = intersection / (union + 1e-10)
    accuracy_class = intersection / (target + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection) / (sum(target) + 1e-10)
    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    logger.info('Val1 result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU1, mAcc1, allAcc1))

    for i in range(args.classes):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], names[i]))

    if config.extra_ops:
        logger.info('<<<<<<<<<<<<<<<<< Extra Ops <<<<<<<<<<<<<<<<<')
        solve_extra_ops(cum_dict_list, label_save, config.extra_ops, config)

    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    if config.save:
        logger.info(f'Saving {config.save}')
        save_to_h5py(cum_dict_list, data_list, config)
    print(flush=True)
    return cum_dict_list

def init_cumulate_dict(N, config, device=None):
    cum_dict = {
        'probs': torch.zeros((N, config.num_classes))
    }
    save_keys = config.save
    extra_ops = config.extra_ops
    if 'novote' in extra_ops:
        cum_dict['probs_last'] = torch.zeros((N, config.num_classes))
    if 'boundary' in extra_ops or 'neighbor' in save_keys:
        cum_dict['neighbors'] = {16: [], 32: [], 64: []}  # should be config.nsample

    cum_dict = TorchDict(cum_dict)
    if device:
        cum_dict = cum_dict.to(device)

    return cum_dict

def cumulate_probs(cum_dict, pred, inds, smooth=None, pred_type='logits'):
    assert pred_type in ['logits'], f'not support pred_type = {pred_type}'
    if smooth is None:
        cum_dict['probs'][inds, ...] += pred
    else:
        cum_dict['probs'][inds, ...] = smooth * cum_dict['probs'][inds, ...] + (1-smooth) * pred

    if 'probs_1st' in cum_dict:
        # update only non-covered
        inds_exists = np.where(cum_dict['probs_1st'].sum(-1) > 0)[0]  # idx into N
        raise
        inds, pred_inds = np.intersect1d(inds, inds_exists, return_indices=True)
        pred = pred[pred_inds]
        if smooth is None:
            cum_dict['probs'][inds, ...] += pred
        else:
            cum_dict['probs'][inds, ...] = smooth * cum_dict['probs'][inds, ...] + (1-smooth) * pred
    
    if 'probs_last' in cum_dict:
        # directly overwrite
        cum_dict['probs_last'][inds, ...] = pred

    return cum_dict

def solve_extra_ops(cum_dict_list, label_list, extra_ops, config):
    """ result of each cloud should be in a separate cum_dict, indexed by cloud_idx
    """
    if not extra_ops:
        return
    print(flush=True)

    if 'novote' in extra_ops:
        print('-' * 10, 'probs_last', '-'*10)
        intersection_meter = AverageMeter()  # accumulating over clouds
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        for cum_dict, label in zip(cum_dict_list, label_list):
            pred = cum_dict['probs_last'].argmax(-1)
            intersection, union, target = intersectionAndUnion(pred, label, config.num_classes, config.ignore_label)
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)

        # calculation 1 - using intersection-union accumulated over clouds
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU1 = np.mean(iou_class)
        mAcc1 = np.mean(accuracy_class)
        allAcc1 = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

        # calculation 2 - using prediction of all clouds
        pred_list = [d['probs_last'].argmax(-1) for d in cum_dict_list]
        intersection, union, target = intersectionAndUnion(np.concatenate(pred_list), np.concatenate(label_list), config.num_classes, config.ignore_label)
        iou_class = intersection / (union + 1e-10)
        accuracy_class = intersection / (target + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection) / (sum(target) + 1e-10)
        print('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        print('Val1 result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU1, mAcc1, allAcc1))
        print(flush=True)

    if 'boundary' in extra_ops:
        print('-' * 10, 'boundary', '-'*10)
        from model.basic_operators import get_boundary_mask
        rst_dict = {}
        for cum_dict, label in zip(cum_dict_list, label_list):
            pred = cum_dict['probs'].argmax(-1)
            bound_dict = {}
            for kr in cum_dict['neighbors']:
                if kr not in rst_dict:
                    rst_dict[kr] = collections.defaultdict(lambda: 0)
                bound_dict[kr] = {}
                neighbor_idx = cum_dict['neighbors'][kr]  # [[BxN, kr], ...]
                bound, plain = get_boundary_mask(torch.from_numpy(label), neighbor_idx=torch.from_numpy(neighbor_idx), get_plain=True)
                bound, plain = bound.cpu().numpy(), plain.cpu().numpy()
                for mask_n, mask in zip(['bound', 'plain'], [bound, plain]):
                    i, u, t = intersectionAndUnion(pred[mask], label[mask], config.num_classes, config.ignore_label)
                    bound_dict[kr][f'{mask_n}-mask'] = mask
                    for v_n, v in zip('iut', [i, u, t]):
                        key = f'{mask_n}-{v_n}'  # e.g. bound-i
                        rst_dict[kr][key] += v
                        bound_dict[kr][key] = v
            bound_dict = dict(bound_dict)
            cum_dict['boundary'] = bound_dict

        # printing rst
        for kr in rst_dict:
            print(f'\t -- kr = {kr}')
            for mask_n in ['bound', 'plain']:
                i = rst_dict[kr][f'{mask_n}-i']
                u = rst_dict[kr][f'{mask_n}-u']
                t = rst_dict[kr][f'{mask_n}-t']
                iou = i / (u + _eps)
                acc = i / (t + _eps)
                oa = i.sum() / (t.sum() + _eps)
                print(f'\t\t {mask_n}:\tmIoU/mAcc/allAcc {iou.mean():.4f}/{acc.mean():.4f}/{oa:.4f}.')
        del rst_dict
        print(flush=True)


def save_to_h5py(cum_dict_list, data_list, config):
    save_keys = config.save
    import h5py
    save_name = config.model_path.split(os.sep)[-1]
    save_name = '_'.join(save_name.split('_')[1:])
    save_name = '.'.join(save_name.split('.')[:-1])
    h5f = h5py.File(f'{config.save_path}/{save_name}.h5f', 'w')  # fresh save directly in 'saving_path'

    for cloud_idx, item in enumerate(data_list):
        h5g = h5f.create_group(f'{cloud_idx}')
        if cum_dict_list:
            cum_dict = cum_dict_list[cloud_idx]
        else:
            pred_save_path = os.path.join(config.save_folder, '{}_{}_pred.npy'.format(item, config.epoch))
            label_save_path = os.path.join(config.save_folder, '{}_{}_label.npy'.format(item, config.epoch))
            assert os.path.isfile(pred_save_path) and os.path.isfile(label_save_path)
            pred, label = np.load(pred_save_path), np.load(label_save_path)
            cum_dict = {'probs': pred, 'label': label}  # NOTE: probs is indeed logits here
        cum_dict = cum_dict.numpy()

        if any(k in save_keys for k in ['label', 'point']):
            data = np.load(os.path.join(config.data_root, f'{item}.npy'))  # xyzrgbl, N*7
            coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]

        if 'prob' in h5g: del h5g['prob']
        h5g.create_dataset('prob', data=cum_dict['probs'])

        if 'point' in save_keys:
            h5g.create_dataset('point', data=coord)

        if 'label' in save_keys:
            h5g.create_dataset('label', data=label)

        if 'boundary' in save_keys and 'boundary' in cum_dict:
            if 'boundary' in h5g: del h5g['boundary']
            for kr in cum_dict['boundary']:
                for n in cum_dict['boundary']:  # may have pred-{n}-{i}
                    raise NotImplementedError(f'check the key---')
                    h5g.create_dataset(f'boundary/{kr}/{n}', data=cum_dict['boundary'][kr][f'mask_{n}-mask'])

        if 'neighbor' in save_keys:
            if 'neighbor' in h5g: del h5g['neighbor']
            for kr in cum_dict['neighbors']:
                h5g.create_dataset(f'neighbor/{kr}', data=cum_dict['neighbors'][kr])

if __name__ == '__main__':
    main()
