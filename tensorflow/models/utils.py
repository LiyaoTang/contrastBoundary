import os, re
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------- #
# decorator
# ---------------------------------------------------------------------------- #

def tf_scope(func):
    """ decorator: automatically wrap a var scope """
    def scopped_func(*args, name=None, reuse=None, **kwargs):
        if name is not None and not reuse:
            with tf.variable_scope(name):
                return func(*args, **kwargs)
        elif name is not None and reuse:  # variable reuse, naming ops as desired
            with tf.variable_scope(reuse, auxiliary_name_scope=False, reuse=True):
                with tf.name_scope(name):
                    return func(*args, **kwargs)
        elif reuse:  # variable reuse + naming ops as is re-enter the scope
            with tf.variable_scope(reuse, reuse=True):
                    return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return scopped_func

def tf_device(func):
    """ decorator: automatically wrap a device scope """
    def scopped_func(*args, device=None, **kwargs):
        if device is not None:
            with tf.device(device):
                return func(*args, **kwargs)
        return func(*args, **kwargs)
    return scopped_func

def tf_Print(*args, summarize=100, **kwargs):
    if 'summarize' in kwargs:
        summarize = kwargs['summarize']
        del kwargs['summarize']
    with tf.device('/cpu:0'): return tf.Print(*args, summarize=summarize, **kwargs)


# ---------------------------------------------------------------------------- #
# helper func
# ---------------------------------------------------------------------------- #

def get_kr(config, stage_n, stage_i):
    assert stage_n in _valid_stage, f'invalid stage_n={stage_n}'
    if stage_n:
        kr = config.kr_sample[stage_i - 1] if stage_n == 'down' else config.kr_sample_up[stage_i]
    else:
        kr = config.kr_search[stage_i]
    return kr

def get_kwargs(block_cfg, config, is_training, act=False):
    # NOTE: may consider provide bias, bn, activation - 1. matching def of dense_layer & mlps, 2. arch level bn control, e.g. ln/gn
    kwargs = {
        'is_training': is_training,
        'initializer': block_cfg.init if block_cfg.init != '' else config.init,
        'weight_decay': block_cfg.wd if block_cfg.wd != '' else config.weight_decay,
        'bn_momentum': config.bn_momentum, 'bn_eps': config.bn_eps,
    }
    if block_cfg.bn != '' or config.bn != '':
        kwargs['bn'] = block_cfg.bn if block_cfg.bn != '' else config.bn
    if act is True:
        kwargs['activation'] = block_cfg.act if block_cfg.act else config.activation
    elif act:
        kwargs['activation'] = act
    return kwargs

def get_kwargs_mlp(block_cfg, config, is_training, act=True, **_kwargs):
    kwargs = get_kwargs(block_cfg, config, is_training, act=act)
    kwargs.update({
        'linearbn': block_cfg.linearbn if block_cfg.linearbn != '' else config.linearbn if config.linearbn != '' else False,
    })
    kwargs.update(_kwargs)
    return kwargs

def get_ftype(ftype, raise_not_found=True):
    if ftype in ['out', 'fout', 'f_out']:
        ptype = 'p_out'
        ftype = 'f_out'
    elif any(re.fullmatch(f'{k}(\d*mlp|mlp\d*|linear|)', ftype) for k in ['latent', 'logits', 'probs']):
        ptype = 'p_out'
        ftype = [k for k in ['latent', 'logits', 'probs'] if ftype.startswith(k)][0]
    elif ftype in ['sample', 'fsample', 'f_sample']:
        ptype = 'p_sample'
        ftype = 'f_sample' if ftype in ['sample', 'fsample'] else ftype
    elif raise_not_found:
        raise KeyError(f'not supported ftype = {ftype}')
    else:
        ftype = ptype = None
    return ftype, ptype


_valid_stage = ['down', 'up', '']
def fetch_supports_flow(inputs, stage_n, stage_i):
    # update based on the flowing direction down/up - building
    assert stage_n in _valid_stage, f'invalid stage_n={stage_n}'
    if stage_n:
        stage_i += -1 if stage_n == 'down' else 1
        idx = inputs['sample_idx'][stage_n][stage_i]
        pts = inputs['points'][stage_i]
    else:
        idx = inputs['neighbors'][stage_i]
        pts = inputs['points'][stage_i]
    return pts, idx

def fetch_supports_stage(inputs, stage_n, stage_i, ftype):
    # indexing the existing stages - all built
    stage_n = to_valid_stage(stage_n)
    stage = inputs['stage_list'][stage_n][stage_i]
    ftype, ptype = get_ftype(ftype)
    pts = stage[ptype]
    f = stage[ftype]
    idx = inputs['neighbors'][stage_i]
    return pts, f, idx

def to_valid_stage(stage_n, short=False):
    if stage_n in ['D', 'down']:
        stage_n = 'D' if short else 'down'
    elif stage_n in ['U', 'up']:
        stage_n = 'U' if short else 'up'
    else:
        raise ValueError(f'invalid stage_n={stage_n}')
    return stage_n

def parse_stage(stage, num_layers):
    stage = stage.replace('a', ''.join(f'{i}' for i in range(num_layers)))
    stage_list = [i.strip('_') for i in re.split('(\d+)', stage) if i and i.strip('_')]  # e.g. D012_U34
    assert len(stage_list) % 2 == 0, f'invalid stage compound: stage_list={stage_list} from stage={stage}'
    stage_n = [s for i, s in enumerate(stage_list) if i % 2 == 0]
    stage_i = [s for i, s in enumerate(stage_list) if i % 2 == 1]
    stage_list = [[(to_valid_stage(n), int(i)) for i in i_str] for n, i_str in zip(stage_n, stage_i)]
    stage_list = sum(stage_list, [])
    return stage_list
