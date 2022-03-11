"""
config for head - auxilary network + output (label)
"""

import re, itertools
from .base import Base, Config
from .utils import gen_config, _xor, _is_property, _is_float

class Head(Base):
    _cls = 'Head'
    def __init__(self, cfg=None, parse=False):
        self._weight = ''  # loss weight
        self._ftype = 'out'  # pts & features to use (of each stage)
        self._stage = ''  # down/up stage

    # TODO:
    # 1. adding loss with stage & ftype control (no footprint on stage_list)
    # 2. adding loss with label level control (sub-scene loss)
    # 3. potential control on range of sub-scene?

    @property
    def weight(self): return self._weight
    @property
    def ftype(self): return self._ftype
    @property
    def stage(self): return self._stage
    @property
    def head_n(self): return type(self).__name__
    @property
    def idx_name_pre(self): return self.head_n
    @property
    def task(self): return self.head_n

    def parse(self, attr=''):
        if not attr:  # skip static parse - as using gen_config/load_config
            return
        for a in attr.split('_'):
            k, v = a.split('-')[0], '-'.join(a.split('-')[1:])
            if self.common_setter(k):
                pass
            elif k and v:
                setattr(self, k, v)
            else:
                raise NotImplementedError(f'Head Base - not supported a = {a} in attr = {attr}')
    def common_setter(self, a):
        if a in ['sample', 'out']:
            self._ftype = a
        elif any([a.startswith(i)] for i in ['up', 'down', 'U', 'D']):
            self._stage = a
        elif re.fullmatch('w\d+', a):
            self._weight = float(a[1:])
        else:
            return False
        return True

class mlp(Head):
    # cross-entropy loss, by default, on the last (most upsampled) layer
    _attr_dict = {'_ops': [
        '1-xen',   # xen - per-pixel softmax with cross entropy,
    ]}
    act = 'relu'
    task = 'seg'
    ftype = 'f_out'
    stage = None
    @property
    def mlp(self): return int(self._ops.split('-')[0])
    @property
    def loss(self): return self._ops.split('-')[1]
    @property
    def weight(self): return '-'.join([i for i in self._ops.split('-')[2:] if i in ['class', 'center'] or i.startswith('w')])
    @property
    def drop(self):
        dp = [i for i in self._ops.split('-')[2:] if i.startswith('dp')]
        return float(dp[0][2:]) if dp else None
    @property
    def name(self): return f'{self.head_n}-{self._ops}'
    def parse(self, attr=''):
        for a in attr.split('-'):
            if a.isdigit(): self.mlp = int(a)
            elif a in ['xen', 'sigmoid', 'none']: self.loss = a
            elif a != 'pred' and self.common_setter(a): pass
            else: raise NotImplementedError(f'Head mlp: not supported a = {a} in attr = {attr}')

class multiscale(Head):
    _attr_dict = {'_ops': [
        '||Ua-concat-fout',  # collect all - main only
        '||Ua-concat-latent',
        '||Ua-concat-latent|w0',  # zero-out loss & grad

    ]}
    loss = 'xen'
    mlp = 1  # used for getting fout->latent
    labels = 'U0'
    task = 'seg'
    @property  # per-branch ops: latent -> logits -> pred -> loss
    def branch(self): return self._ops.split('|')[0]
    @property
    def condition(self): return self._ops.split('|')[1]
    @property
    def main(self): return self._ops.split('|')[2]

    @property
    def _extra(self): return self._ops.split('|')[3:]
    @property
    def weight(self):
        w = self._extra[0] if self._extra else ''
        w = [i for i in w.split('-') if i.startswith('w')]
        return  float(w[0][1:]) if w else ''

    @property
    def name(self): return 'multi-' + '_'.join([i for i in self._ops.split('|') if i])

main_dict = {}
gen_config([mlp, multiscale], main_dict)
for k, v in main_dict.items():
    globals()[k] = v


class contrast(Head):
    _attr_dict = {'_ops': [
        'softnn|latent|label|l2||w.1|Ua',

        'softnn|latent|label_nst|l2||w.1|Ua',
        'softnn|latent|label_recur|l2||w.1|Ua',
        'softnn|latent|label_recurhard|l2||w.1|Ua',

        'softnn|latent|labelkl.3|l2||w.1|Ua',
        'softnn|latent|labelkl.5|l2||w.1|Ua',
        'softnn|latent|labelkl1|l2||w.1|Ua',
        'softnn|latent|labelkl2|l2||w.1|Ua',

        'softnn|latent|label|l2|mT.3|w.1|Ua',
        'softnn|latent|label|l2|mT.5|w.1|Ua',
        'softnn|latent|label|l2|mT2|w.1|Ua',
        'softnn|latent|label|l2|mT3|w.1|Ua',
        'softnn|latent|label|l2|mT5|w.1|Ua',
        'softnn|latent|label|l2|mT10|w.1|Ua',

    ]}
    head_n = 'contrast'
    @property
    def contrast(self): return self._ops.split('|')[0]  # how to use the distance - softnn
    @property
    def ftype(self):
        return self._ops.split('|')[1]
    @property
    def project(self):
        proj = self._ops.split('|')[1][len(self.ftype):]
        return proj.strip('-')
    @property
    def sample(self):
        s = self._ops.split('|')[2]
        return s
    @property
    def dist(self): return self._ops.split('|')[3]  # distance
    @property
    def _aug(self): return self._ops.split('|')[4].split('-')  # margin-mask-power
    @property
    def mask(self): return ''.join([i for i in self._aug if i.startswith('mask')])  # if selecting the dist
    @property
    def margin(self): return ''.join([i for i in self._aug if re.search('m(?!ask)', i)])[1:]  # margin/temperature
    @property
    def contrast_aug(self): return '-'.join([i for i in self._aug if i[:1] in ['p']])  # shared aug - power
    @property
    def weight(self): return self._ops.split('|')[5][1:]
    @property
    def stage(self): return self._ops.split('|')[6]
    @property
    def name(self): return '-'.join([i for i in [self.head_n, self.stage] + self._ops.split('|')[:6] if i])

contrast_dict = {}
gen_config([contrast], contrast_dict)
for k, v in contrast_dict.items():
    globals()[k] = v

del k, v

