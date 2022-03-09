import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from lib.pointops.functions import pointops
from .utils import *
from .blocks import *
from .heads import *
from .basic_operators import *
from .basic_operators import _eps, _inf



class Loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.contrast_head = ContrastHead(config.contrast, config) if 'contrast' in config else None
        self.xen = nn.CrossEntropyLoss(ignore_index=config.ignore_label)
    def forward(self, output, target, stage_list):
        loss_list = [self.xen(output, target)]
        if self.contrast_head is not None:
            loss_list += self.contrast_head(output, target, stage_list)
        return torch.stack(loss_list)

class PointTransformerSeg(nn.Module):
    def __init__(self, block, blocks, c=6, k=13, config=None):
        super().__init__()
        self.c = c
        self.in_planes = c

        # fdims
        if 'planes' not in config:
            config.planes = [32, 64, 128, 256, 512]
        planes = config.planes

        # shared head in att
        if 'share_planes' not in config:
            config.share_planes = 8
        share_planes = config.share_planes

        fpn_planes, fpnhead_planes = 128, 64
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        if 'stride' not in config:
            config.stride = stride
        if 'nsample' not in config:
            config.nsample = nsample
        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])  # N/1   - planes(fdims)=32,  blocks=2, nsample=8
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])  # N/4   - planes(fdims)=64,  blocks=3, nsample=16
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])  # N/16  - planes(fdims)=128, blocks=4, nsample=16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])  # N/64  - planes(fdims)=256, blocks=6, nsample=16
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])  # N/256 - planes(fdims)=512, blocks=3, nsample=16
        self.dec5 = self._make_dec(block, planes[4], 2, share_planes, nsample=nsample[4], is_head=True)  # transform p5
        self.dec4 = self._make_dec(block, planes[3], 2, share_planes, nsample=nsample[3])  # fusion p5 and p4
        self.dec3 = self._make_dec(block, planes[2], 2, share_planes, nsample=nsample[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(block, planes[1], 2, share_planes, nsample=nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(block, planes[0], 2, share_planes, nsample=nsample[0])  # fusion p2 and p1
        self.head = self.cls = None

        self.config = config
        config.num_layers = 5
        config.num_classes = k

        if 'multi' in config:
            self.head = MultiHead(planes, config.multi, config)
        else:
            self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], k))

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        """
        stride = 1 => TransitionDown = mlp, [block, ...]
        stride > 1 => 
        """
        layers = []
        layers.append(TransitionDown(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion  # expansion default to 1
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(TransitionUp(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, inputs):
        pxo = inputs['points'], inputs['features'], inputs['offset']
        p0, x0, o0 = pxo[:3]  # (n, 3), (n, c), (b) - c = in_feature_dims
        if self.c == 3:
            x0 = p0
        elif self.c == 6:
            x0 = torch.cat((p0, x0), 1)
        elif self.c == 7:
            x0 = torch.cat((torch.ones_like(p0[..., :1]), p0, x0), 1)
        else:
            raise

        stage_list = {'inputs': inputs}

        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        down_list = [
            # [p0, x0, o0],  # (n, 3), (n, in_feature_dims), (b)
            {'p_out': p1, 'f_out': x1, 'offset': o1},  # (n, 3), (n, base_fdims), (b) - default base_fdims = 32
            {'p_out': p2, 'f_out': x2, 'offset': o2},  # n_1
            {'p_out': p3, 'f_out': x3, 'offset': o3},  # n_2
            {'p_out': p4, 'f_out': x4, 'offset': o4},  # n_3
            {'p_out': p5, 'f_out': x5, 'offset': o5},  # n_4 - fdims = 512
        ]
        # for i, s in enumerate(down_list):
        #     print('\n\t'.join([str(i)] + [str(ss.shape) for ss in s]))
        stage_list['down'] = down_list

        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]  # no upsample - concat with per-cloud mean: mlp[ x, mlp[mean(x)] ]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        up_list = [
            {'p_out': p1, 'f_out': x1, 'offset': o1},  # n_0 = n, fdims = 32
            {'p_out': p2, 'f_out': x2, 'offset': o2},  # n_1
            {'p_out': p3, 'f_out': x3, 'offset': o3},  # n_2
            {'p_out': p4, 'f_out': x4, 'offset': o4},  # n_3
            {'p_out': p5, 'f_out': x5, 'offset': o5},  # n_4 - fdims = 512 (extracted through dec5 = mlps)
        ]
        # for i, s in enumerate(up_list):
        #     print('\n\t'.join([str(i)] + [str(ss.shape) for ss in s]))
        stage_list['up'] = up_list

        if self.head is not None:
            x, stage_list = self.head(stage_list)
        else:
            x = self.cls(x1)

        # stage_list['up'][0]['logits'] = x  # TODO: make stage_list a dict at each stage
        return x, stage_list


def pointtransformer_seg_repro(**kwargs):
    config = kwargs['config'] if 'config' in kwargs else None
    blocks = [2, 3, 4, 6, 3]
    model = PointTransformerSeg(PointTransformerBlock, blocks, **kwargs)
    return model
