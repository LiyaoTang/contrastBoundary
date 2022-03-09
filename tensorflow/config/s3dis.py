import re, itertools
from . import archs
from .base import Base, Config
from .utils import gen_config, _xor, _is_property, _is_float
from collections import defaultdict

def add_cfg(d):
    for k, v in d.items():
        assert k not in add_cfg.k, f'idx_name = {k} alread existed'
        assert v.name not in add_cfg.v, f'idx_name = {k}, cfg name = {v.name} - alread existed'
        add_cfg.k.add(k)
        add_cfg.v.add(v.name)
        globals()[k] = v
add_cfg.k = set()
add_cfg.v = set()

class Default(Config):
    """
    dataset default setting
    """
    dataset = 'S3DIS'

    # ---------------------------------------------------------------------------- #
    # Training
    # ---------------------------------------------------------------------------- #
    gpus = 4
    batch_size = 4  # actual running batch (per-gpu) - influence BxN
    batch_size_val = 16
    # epoch setting
    epoch_batch = 8  # batch size per-step - TODO: set to batch_size * gpu_num
    epoch_steps = 500  # desired step per-epoch - #samples (from global point cloud) = epoch_batch (# input point cloud per step) * epoch_steps
    validation_steps = 100  # desired step per validation epoch
    max_epoch = 600    # optimizer
    learning_rate = 0.01  # 0.005 * sqrt(total batch size) / 2
    optimizer = 'sgd'
    momentum = 0.98
    decay_epoch = 1
    decay_rate = 0.9885531
    grad_norm = 100
    grad_raise_none = True
    # saving & io
    num_threads = 12  # thread to provide data
    print_freq = 60
    update_freq = 10
    save_freq = 50
    val_freq = 10
    save_val = ''  # if saving val results (by keys)
    save_sample = False  # if saving sample stat
    save_compact = True  # if saving only vars
    summary = False  # if using tf summary
    runtime_freq = 0  # runtime stat
    # testing
    num_votes = 20

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #
    in_radius = 2.0  # sampling data
    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.7
    augment_scale_max = 1.3
    augment_noise = 0.001
    augment_color = 0.8


    # ---------------------------------------------------------------------------- #
    # Model
    # ---------------------------------------------------------------------------- #
    num_classes = 13  # num of valid classes
    in_features_dim = 5  # input point feature type
    @property
    def in_features(self):
        return {1: '1', 2: '1-Z', 3: 'rgb', 4: '1-rgb', 5: '1-rgb-Z', 6: '1-rgb-xyz', 7: '1-rgb-xyz-Z'}[self.in_features_dim]
    first_features_dim = 72
    num_layers = 5
    # sampling & search
    search = 'radius'  # knn/radius search
    sample = 'grid'  # grid/random search
    density_parameter = 5.0
    first_subsampling_dl = 0.04
    # radius - neighbor/pooling (during sub-sampling)/up-sampling
    kr_search = [dl * dp / 2 * (2 ** i) for i, (dl, dp) in enumerate([(first_subsampling_dl, density_parameter)] * num_layers)]
    kr_sample = kr_search[:-1]
    kr_sample_up = [2 * r for r in kr_search[:-1]]  # up-sampling radius
    r_sample = [dl * 2 ** (i+1) for i, dl in enumerate([first_subsampling_dl]*(num_layers-1))]  # ratio/radius of sub-sampling points
    neighborhood_limits = [26, 31, 38, 41, 39, 29]
    # global setting
    activation = 'relu'  # relu, prelu
    init = 'xavier'
    weight_decay = 0.001
    bn_momentum = 0.99
    bn_eps = 1e-6
    extra_ops = 'boundary-stat'
    idx_name = name = 'default'
default = Default()

class Origin(Default):
    _attr_dict = {'_ops': [
            'adapt|',
            'grid|',
            'pospool|',
        ]}
    _update_dict = {
        'adapt': 'config/s3dis/adapt.yaml',
        'grid': 'config/s3dis/pseudogrid.yaml',
        'pospool': 'config/s3dis/pospool.yaml',
    }
    idx_name_pre = 'origin'
    builder = 'SceneSegModel'
    architecture = ''
    gpus = 2


    def __init__(self, cfg=None, parse=True):
        super(Origin, self).__init__(cfg, parse)
        self._init = default.init

    @property
    def _main(self): return re.match('[a-z]+', self._ops.split('|')[0]).group()
    @property
    def update_path(self): return self._update_dict[self._main]

    @property
    def arch_out(self):  # head
        out = self._ops.split('|')[1:]
        if not out or not any(h.split('-')[0] in ['mlp', 'multi'] for h in out):
            # add main-head if not present
            out = ['mlp-1-xen'] + out
        return out

    @property
    def init(self):
        init = None
        for i in self._ops.split('|')[0].split('-')[1:]:
            if i.startswith('I'): init = i[1:]
        return init if init else self._init
    @init.setter
    def init(self, i): self._init = i

    @property
    def sep_head(self): return any(h.startswith(i) for i in ['multi'] for h in self.arch_out)
    @property
    def name(self):
        return '_'.join([self.idx_name_pre, *[i for i in self._ops.split('|') if i]])

class Conv(Origin):
    # try some high train score & low val score with relu (instead of leaky_relu)
    _attr_dict = {'_ops': [
        '|multi-Ua-concat-latent|contrast-Ua-softnn-latent-label-l2-w.1',  # good
        # temperature
        '|multi-Ua-concat-latent|contrast-Ua-softnn-latent-label-l2-mT.3-w.1',
        '|multi-Ua-concat-latent|contrast-Ua-softnn-latent-label-l2-mT.5-w.1',
        '|multi-Ua-concat-latent|contrast-Ua-softnn-latent-label-l2-mT2-w.1',
        '|multi-Ua-concat-latent|contrast-Ua-softnn-latent-label-l2-mT3-w.1',
        '|multi-Ua-concat-latent|contrast-Ua-softnn-latent-label-l2-mT5-w.1',
        # kl
        '|multi-Ua-concat-latent|contrast-Ua-softnn-latent-labelkl.5-l2-w.1',
        '|multi-Ua-concat-latent|contrast-Ua-softnn-latent-labelkl.5-l2-mT.5-w.1',
    ]}
    activation = 'relu'
    idx_name_pre = 'conv'

    def __init__(self, cfg=None, parse=True):
        super(Conv, self).__init__(cfg, parse)

    @property
    def update_path(self): m = self._main if self._main else 'adapt'; return self._update_dict[m]

    @property
    def name(self):
        name_pre = self._name_pre if self._name_pre else self.idx_name_pre
        m = self._ops.split('|')[0][len(self._main):] if self._main == 'adapt' else '_' + self._ops.split('|')[0]
        return '_'.join([i for i in [(name_pre + m).strip('_')] + self._ops.split('|')[1:] if i])

class pospool(Conv):
    _attr_dict = {'_ops': [
        # pospool
        'pospool|multi-Ua-concat-latent|contrast-Ua-softnn-latent-label-l2-w.1',
    ]}
    idx_name_pre = 'pospool'
    _name_pre = Origin_relu.idx_name_pre

origin_dict = {}
gen_config([Conv, pospool], store_dict=origin_dict)
for k, v in origin_dict.items():
    if v.update_path:
        v.update(v.update_path)
add_cfg(origin_dict)

del k, v

