import re, itertools
from .base import Base, Config
from .utils import gen_config, _xor, _is_property, _is_float
from collections import defaultdict

class Default(Config):
    """
    dataset default setting
    """
    dataset = 'ScanNet'

    # ---------------------------------------------------------------------------- #
    # Training
    # ---------------------------------------------------------------------------- #
    gpus = 4
    batch_size = 4  # actual running batch (per-gpu)
    # epoch setting
    epoch_steps = 500  # desired step per-epoch
    epoch_batch = 8  # batch size per-step
    max_epoch = 1000
    validation_steps = 50  # Number of validation step per epoch - #samples = total_batch * validation_steps
    # loss
    loss_weight = None  # if using weighted loss
    # optimizer
    learning_rate = 0.01
    optimizer = 'sgd'
    momentum = 0.98
    decay_epoch = 1
    decay_rate = 0.9885531
    # decay_rate = 0.1**(1/100)  # 0.9772372209558107
    grad_norm = 100
    # saving & io
    num_threads = 12  # thread to provide data
    print_freq = 60
    update_freq = 10
    save_freq = 10
    val_freq = 10
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
    augment_scale_min = 0.7  # 0.9
    augment_scale_max = 1.3  # 1.1
    augment_noise = 0.001
    augment_color = 0.8  # 1.0


    # ---------------------------------------------------------------------------- #
    # Model
    # ---------------------------------------------------------------------------- #
    num_classes = 20  # num of valid classes
    in_features = '1rgbZ'  # input point feature type
    first_features_dim = 72
    num_layers = 5
    # sampling & search
    search = 'radius'  # knn/radius search
    sample = 'grid'  # grid/random/fps search
    density_parameter = 5.0
    first_subsampling_dl = 0.04
    # radius - neighbor/pooling (during sub-sampling)/up-sampling
    kr_search = [dl * dp / 2 * (2 ** i) for i, (dl, dp) in enumerate([(first_subsampling_dl, density_parameter)] * num_layers)]
    kr_sample = kr_search[:-1]
    kr_sample_up = [2 * r for r in kr_search[:-1]]  # up-sampling radius
    r_sample = [dl * 2 ** (i+1) for i, dl in enumerate([first_subsampling_dl]*(num_layers-1))]  # ratio/radius of sub-sampling points
    neighborhood_limits = [26, 31, 35, 39, 35]  # [26, 31, 38, 41, 39] 
    # global setting
    activation = 'relu'
    init = 'xavier'
    weight_decay = 0.001
    bn_momentum = 0.99
    bn_eps = 1e-6
    idx_name = name = 'default'
default = Default()

class _adapt(Base):
    init = 'xavier'
    activation_fn = 'relu'
    bottleneck_ratio = 2
    depth = 1
    local_aggreagtion = 'adaptive_weight'
    adaptive_weight = Base({
        'local_input_feature': 'dp',
        'reduction': 'mean',
        'shared_channels': 1,
        'fc_num': 1,
        'weight_softmax': False,
        'output_conv': False,
    })
    name = 'default_adaptive'
    builder = 'SceneSegModel'
_adapt = _adapt()

class Conv(Default):
    _attr_dict = {'_ops': [
        'adapt|',

        # contrastive boundary
        '|multi-Ua-concat-latent|contrast-Ua-softnn-latent-label-l2-w.1',

        # contrastive boundary - temperature
        '|multi-Ua-concat-latent|contrast-Ua-softnn-latent-label-l2-mT.1-w.1',
        '|multi-Ua-concat-latent|contrast-Ua-softnn-latent-label-l2-mT.3-w.1',
        '|multi-Ua-concat-latent|contrast-Ua-softnn-latent-label-l2-mT.5-w.1',
        '|multi-Ua-concat-latent|contrast-Ua-softnn-latent-label-l2-mT2-w.1',
        '|multi-Ua-concat-latent|contrast-Ua-softnn-latent-label-l2-mT5-w.1',
        '|multi-Ua-concat-latent|contrast-Ua-softnn-latent-label-l2-mT10-w.1',

        ]}
    _update_dict = {
        'adapt': _adapt,
    }
    idx_name_pre = 'conv'
    builder = 'SceneSegModel'
    architecture = ''
    extra_ops = 'boundary-stat'
    gpus = 2

    @property
    def _main(self): return re.match('[a-zA-Z]+', self._ops.split('|')[0]).group()
    @property
    def update_path(self): m = self._main if self._main else 'adapt'; return m
    @property
    def arch_out(self):  # head
        out = self._ops.split('|')[1:]
        if not out or not any(h.split('-')[0] in ['mlp', 'multi'] for h in out):
            # add main-head if not present
            out = ['mlp-1-xen'] + out
        return out
    @property
    def sep_head(self): return any(h.startswith(i) for i in ['multi'] for h in self.arch_out)
    @property
    def name(self):
        return '_'.join([self.idx_name_pre, *[i for i in self._ops.split('|') if i]])


conv_dict = {}
gen_config([Conv], store_dict=conv_dict)
for k, v in conv_dict.items():
    if v.update_path:
        v.update(v._update_dict[v.update_path])
    globals()[k] = v

del k, v
