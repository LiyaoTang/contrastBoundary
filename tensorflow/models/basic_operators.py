import os, sys
import numpy as np
import tensorflow as tf
if tf.__version__.split('.')[0] == '2':
    tf = tf.compat.v1
    tf.disable_v2_behavior()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, '..'))

from .utils import tf_scope

def masked_softmax(*args, impl='mask', **kwargs):
    impl = masked_softmax.impl if masked_softmax.impl else impl
    func = {
        'mask': masked_softmax_mask,
        'inf': masked_softmax_inf,  # more memory efficient
    }[impl]
    return func(*args, **kwargs)
masked_softmax.impl = None  # config-dependent analysis use

def masked_softmax_mask(vec, valid_mask, axis=-1):
    if valid_mask.dtype == tf.bool:
        valid_mask = tf.cast(valid_mask, vec.dtype)
    assert valid_mask.dtype == vec.dtype, f'not supported mask {valid_mask.dtype} with vec {vec.dtype}'

    vec = vec - tf.reduce_max(vec, axis=axis, keepdims=True)
    exps = tf.exp(vec)
    masked_exps = exps * valid_mask
    masked_sums = tf.reduce_sum(masked_exps, axis=axis, keepdims=True)
    return masked_exps / masked_sums  # invalid position masked to 0

def masked_softmax_inf(vec, valid_mask, axis=-1):
    if valid_mask.dtype == tf.bool:
        valid_mask = tf.cast(valid_mask, vec.dtype)
    assert valid_mask.dtype == vec.dtype, f'not supported mask {valid_mask.dtype} with vec {vec.dtype}'
    return tf.nn.softmax(vec - (1 - valid_mask) * _inf, axis=axis)

def dense_masked_softmax(logits, mask, T):
    """ Masked softmax over dim 1, mask broadcasts over dim 2, using normal softmax

    Args:
        logits: (N, L, T)
        mask: (N, L)
        T: number of dim 2
    Returns:
        probabilities (N, L, T)
    """

    v = T
    indices = tf.cast(tf.where(tf.logical_not(mask)), tf.int32)
    inf = tf.constant(np.array([[_inf]], dtype=np.float32), dtype=tf.float32)
    infs = tf.tile(inf, [tf.shape(indices)[0], v])
    infmask = tf.scatter_nd(
        indices=indices,
        updates=infs,
        shape=tf.shape(logits))
    _p = tf.nn.softmax(logits - infmask, axis=1)
    return _p


def sparse_masked_softmax(logits, mask):
    """Masked softmax over dim -1 using sparse softmax

    Args:
        logits: (N, L, T)
        mask: (N, L, T)

    Returns:
        probabilities (N, L, T)
    """

    indices = tf.where(mask)
    values = tf.gather_nd(logits, indices)
    denseShape = tf.cast(tf.shape(logits), tf.int64)
    sparseResult = tf.sparse_softmax(tf.SparseTensor(indices, values, denseShape))
    result = tf.scatter_nd(sparseResult.indices, sparseResult.values, sparseResult.dense_shape)
    result.set_shape(logits.shape)
    return result


def _variable_on_cpu(name, shape, initializer, use_fp16=False):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device("/cpu:0"):
        dtype = tf.float16 if use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, wd, stddev=1e-3, init='xavier'):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      wd: add L2Loss weight decay multiplied by this float.
      stddev: standard deviation of a truncated Gaussian
      init: weight initializer type

    Returns:
      Variable Tensor
    """
    if init == 'xavier':
        initializer = tf.glorot_uniform_initializer()
    elif init == 'msra':
        initializer = tf.contrib.layers.variance_scaling_initializer()
    elif init == 'fan_in':
        initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=False)
    elif not isinstance(init, str):
        initializer = init
    else:
        initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = _variable_on_cpu(name, shape, initializer)
    if wd > 0:  # L2 regularization
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
        tf.add_to_collection('weight_losses', weight_decay)
    return var


def batch_norm(inputs, is_training, scope, bn_decay=0.99, epsilon=0.001):
    """ Batch normalization on convolutional maps and beyond...
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow

    Args:
        inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
        is_training:   boolean tf.Varialbe, true indicates training phase
        scope:         string, variable scope
        bn_decay:      float or float tensor variable, controling moving average weight
    Return:
        normed:        batch-normalized maps
    """
    return tf.layers.batch_normalization(inputs, axis=-1,
                                         momentum=bn_decay,
                                         epsilon=epsilon,
                                         training=is_training,
                                         trainable=True,
                                         name=scope,
                                         fused=False)


def ind_max_pool(x, inds, scope):
    """ This tensorflow operation compute a maxpooling according to the list of indices 'inds'.

    Args:
        x: [n1, d] features matrix
        inds: [n2, max_num] each row of this tensor is a list of indices of features to be pooled together
        scope: name scope

    Returns:
        [n2, d] pooled features matrix
    """
    with tf.variable_scope(scope) as sc:
        # Add a last row with minimum features for shadow pools
        x = tf.concat([x, tf.reduce_min(x, axis=0, keepdims=True)], axis=0)
        # Get features for each pooling cell [n2, max_num, d]
        pool_features = tf.gather(x, inds, axis=0)
        # Pool the maximum
        return tf.reduce_max(pool_features, axis=1)


def ind_closest_pool(x, inds, scope):
    """This tensorflow operation compute a pooling according to the list of indices 'inds'.

    Args:
        x: [n1, d] features matrix
        inds: [n2, max_num] We only use the first column of this which should be the closest points too pooled positions
        scope:

    Returns:
        [n2, d] pooled features matrix
    """

    with tf.variable_scope(scope) as sc:
        # Add a last row with minimum features for shadow pools
        x = tf.concat([x, tf.zeros((1, int(x.shape[1])), x.dtype)], axis=0)
        # Get features for each pooling cell [n2, d]
        pool_features = tf.gather(x, inds[:, 0], axis=0)
        return pool_features


def conv1d_1x1(features,
               out_fdim,
               scope,
               is_training,
               with_bias=False,
               init='xavier',
               weight_decay=0,
               activation_fn='relu',
               bn=True,
               bn_momentum=0.98,
               bn_eps=1e-3):
    """A simple 1x1 1D convolution

    Args:
        features: Input features, float32[n_points, in_fdim]
        out_fdim: Output features dim
        scope: name scope
        is_training: True indicates training phase
        with_bias: If True, adds a learnable bias to the output
        init: Weight initializer
        weight_decay: If > 0 , add L2Loss weight decay multiplied by this float.
        activation_fn: Activation function
        bn: If True, add batch norm after convolution


    Returns:
        [n_points, out_fdim]
    """
    with tf.variable_scope(scope) as sc:
        in_fdim = int(features.shape[-1])
        w = _variable_with_weight_decay('weights',
                                        shape=[in_fdim, out_fdim],
                                        init=init,
                                        wd=weight_decay)
        if with_bias:
            biases = _variable_on_cpu('biases', [out_fdim], tf.constant_initializer(0.0))
            x = tf.matmul(features, w) + biases
        else:
            x = tf.matmul(features, w)
        if bn:
            x = batch_norm(x, is_training=is_training, scope='bn', bn_decay=bn_momentum, epsilon=bn_eps)

        if activation_fn == 'relu':
            x = tf.nn.relu(x)
        elif activation_fn == 'leaky_relu':
            x = tf.nn.leaky_relu(x, alpha=0.2)
        return x

def batch_conv1d_1x1(features,
                     out_fdim,
                     scope,
                     is_training,
                     with_bias=False,
                     init='xavier',
                     weight_decay=0,
                     activation_fn='relu',
                     bn=True,
                     bn_momentum=0.98,
                     bn_eps=1e-3):
    """A simple 1x1 1D convolution for batch inputs

        Args:
            features: Input features, float32[b, n_points, in_fdim]
            out_fdim: Output features dim
            scope: name scope
            is_training: True indicates training phase
            with_bias: If True, adds a learnable bias to the output
            init: Weight initializer
            weight_decay: If > 0, add L2Loss weight decay multiplied by this float.
            activation_fn: Activation function
            bn: If True, add batch norm after convolution


        Returns:
            [b, n_points, out_fdim]
        """
    with tf.variable_scope(scope) as sc:
        in_fdim = int(features.shape[-1])
        w = _variable_with_weight_decay('weights',
                                        shape=[in_fdim, out_fdim],
                                        init=init,
                                        wd=weight_decay)
        if with_bias:
            biases = _variable_on_cpu('biases', [out_fdim], tf.constant_initializer(0.0))
            x = tf.tensordot(features, w, 1) + biases
        else:
            x = tf.tensordot(features, w, 1)
        if bn:
            x = batch_norm(x, is_training=is_training, scope='bn', bn_decay=bn_momentum, epsilon=bn_eps)

        if activation_fn == 'relu':
            x = tf.nn.relu(x)
        elif activation_fn == 'leaky_relu':
            x = tf.nn.leaky_relu(x, alpha=0.2)
        return x


def global_average_block(inputs, features, scope):
    """This Block performing a global average pooling over batch pooling

    Args:
        inputs: a dict contains all inputs
        features: [n_points, in_fdim]
        scope: name scope

    Returns:
        [b, in_fdim]

    """
    with tf.variable_scope(scope) as sc:
        # Get the number of features
        N = tf.shape(features)[0]
        # Add a last zero features for shadow batch inds
        features = tf.concat([features, tf.zeros((1, int(features.shape[1])), features.dtype)], axis=0)
        # Collect each batch features
        batch_features = tf.gather(features, inputs['out_batches'], axis=0)
        # Average features in each batch
        batch_features = tf.reduce_sum(batch_features, axis=1)
        # batch_num = tf.reduce_sum(tf.cast(inputs['out_batches'] >= 0, tf.float32), axis=1, keepdims=True)
        batch_num = tf.reduce_sum(tf.cast(inputs['out_batches'] < N, tf.float32), axis=1, keepdims=True)
        features = batch_features / batch_num
    return features


def global_max_block(inputs, features, scope):
    """This Block performing a global max pooling over batch pooling

    Args:
        inputs: a dict contains all inputs
        features: [n_points, in_fdim]
        scope: name scope

    Returns:
        [b, in_fdim]
    """

    # Average pooling to aggregate feature in the end
    with tf.variable_scope(scope) as sc:
        # Get the number of features
        N = tf.shape(features)[0]

        # Add a last zero features for shadow batch inds
        features = tf.concat([features, -256.0 + tf.zeros((1, int(features.shape[1])), features.dtype)], axis=0)

        # Collect each batch features
        batch_features = tf.gather(features, inputs['out_batches'], axis=0)

        # Average features in each batch
        batch_features = tf.reduce_max(batch_features, axis=1)

        features = batch_features

    return features

# global config
_eps = 1e-12
_inf = 1e9
def get_initializer(k):
    if not isinstance(k, str) and k is not None : return k  # assume to be an initialization already
    elif k in [None, 'xavier', 'glorot']: return tf.initializers.glorot_uniform()  # scale = 1.0, fan_avg (uniform is used in paper)
    elif k == 'glorotN': return tf.initializers.glorot_normal()  # scale = 1.0, fan_avg
    elif k == 'fanin': return tf.initializers.variance_scaling(scale=1.0)  # fan_in, tunc normal - default variance_scaling - usually for conv
    elif k in ['msra', 'he']: return tf.initializers.variance_scaling(scale=2.0)  # fan_in, tunc normal - he normal (as used in paper)
    elif k == 'zeros': return tf.initializers.zeros()
    elif k == 'ones': return tf.initializers.ones()
    else:
        raise KeyError(f'not supported initializer = {k}')

def get_type(k):
    if k in [True, None, float, 'float', 'float32']: return tf.float32
    elif k in ['float16']: return tf.float16
    elif k in [int, 'int', 'int32']: return tf.int32
    elif k in [bool, 'bool']: return tf.bool
    elif hasattr(tf, k) and isinstance(k, str): return getattr(tf, k)
    else: return k

def tf_get_variable(name, shape, initializer=None, wd=None, device='/cpu:0'):
    initializer = get_initializer(initializer)
    with tf.device(device):
        var = tf.get_variable(name, shape=shape, initializer=initializer)  # use_resource=True

    if wd and wd > 0:  # L2 regularization
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('weight_losses', weight_decay)
    return var

def tf_gather(supports, neighbor_idx, shadow_fn=tf.zeros, get_mask=True):
    mask = None
    batch_dims = len(neighbor_idx.shape) - 2  # [..., N, k]
    # assert batch_dims >= 0, f'batch_dims = {batch_dims}, consider direct calling of tf.gather on supports = {supports}'
    if batch_dims == 0 and get_mask:  # radius search - [BxN, k]
        T = get_type(get_mask)  # may specify types
        N = tf.shape(supports)[0]
        mask = neighbor_idx < N if T == tf.bool else tf.cast(neighbor_idx < N, T)

    # supports - [..., N] or [..., N, d]
    if batch_dims == 0:
        if isinstance(shadow_fn, (int, float)):
            if shadow_fn == 0:
                shadow_fn = tf.zeros
            else:
                shadow_fn = lambda *args, _const=shadow_fn, **kwargs: tf.zeros(*args, **kwargs) + _const
        elif isinstance(shadow_fn, str):
            shadow_fn = lambda *args, _func=shadow_fn, **kwargs: getattr(tf, f'reduce_{_func}')(supports, axis=0, keepdims=True)
        if shadow_fn is not None:
            if isinstance(shadow_fn, tf.Tensor):
                shadow = shadow_fn
            else:
                shape = tf.shape(supports[:1, ...])  # [1, ...]
                shadow = shadow_fn(shape=shape, dtype=supports.dtype)
            supports = tf.concat([supports, shadow], axis=0)
    # batch_dims = max(batch_dims, 0)  # avoid '-1'
    neighbors = tf.gather(supports, neighbor_idx, batch_dims=batch_dims)

    rtn = (neighbors, mask) if get_mask else neighbors
    return rtn

def get_activation(activation):
    if isinstance(activation, str):
        if activation == 'relu':
            activation = tf.nn.relu
        elif activation.startswith('lrelu') or activation.startswith('leaky_relu'):
            alpha = activation.split('relu')[-1]
            alpha = float(alpha) if alpha else 0.2
            activation = lambda x, a=alpha: tf.nn.leaky_relu(x, a)
        elif activation.startswith('prelu'):
            alpha = activation.split('relu')[-1]
            alpha = float(alpha) if alpha else None
            activation = lambda x, a=alpha: prelu(x, init=a)
        elif activation == 'gelu':
            activation = gelu
        elif activation == 'sine':
            # following "Implicit Neural Representations with Periodic Activation Functions"
            raise
            activation = lambda x: 30 * x
        else:
            raise NotImplementedError(f'not supported activation = {activation}')

    return activation

def dense_layer(features, d_out, name=None, activation=None, bias=True, bn=True, is_training=None, initializer=None,
                weight_decay=0, bn_momentum=0.99, bn_eps=1e-6):
    if dense_layer.config and dense_layer.config.dense_by_conv:
        bias = False if bn else bias
        return conv1d_1x1(features, d_out, name, is_training, bias, initializer, weight_decay, activation, bn, bn_momentum, bn_eps)
    if name:
        with tf.variable_scope(name) as sc:
            return dense_layer(features, d_out, None, activation, bias, bn, is_training, initializer, weight_decay, bn_momentum, bn_eps)

    kernel = tf_get_variable('weights', shape=(int(features.shape[-1]), d_out,), initializer=initializer, wd=weight_decay)
    features = tf.tensordot(features, kernel, 1)
    bn = 'bn' if bn is True else bn  # default to bn
    if bn == 'bn':
        assert is_training is not None
        features = tf.layers.batch_normalization(features, momentum=bn_momentum, epsilon=bn_eps, training=is_training, fused=False)
    elif bn:
        raise NotImplementedError(f'not implemented norm = {bn}')
    elif bias:
        bias = tf_get_variable('bias', shape=(d_out,), initializer=tf.initializers.zeros())
        features = features + bias

    activation = get_activation(activation)
    if activation is not None:
        features = activation(features)
    return features
dense_layer.config = None  # config-dependent analysis use

@tf_scope
def mlps_by_ops(features, d_out, num_classes, ops, kwargs, **_kwargs):
    # warpper of mlps - specifying with str (ops)
    ops = f'{ops}mlp' if isinstance(ops, int) else ops
    if not (ops.startswith('mlp') or ops.endswith('mlp') or ops in ['linear', 'linearmap', 'linearbn']):
        raise ValueError(f'invalid ops = {ops}')  # e.g. 2mlp, mlp, mlp2, linear
    linear = not ops.endswith('mlp')  # 2mlp / mlp -> no linear        
    mlp_num = int(ops.replace('mlp', '')) if ops.replace('mlp', '').isdigit() else 1
    kwargs.update(**_kwargs)
    if ops in ['linearmap', 'linearbn']:
        kwargs.update(linearbn={'linearmap': False, 'linearbn': True}[ops])
    features = mlps(features, d_out, num_classes, mlp_num, linear=linear, **kwargs)
    return features

@tf_scope
def mlps(features, d_out, num_classes, num_layers, is_training, dp=False, linear=True, bn=True, activation=tf.nn.leaky_relu,
         initializer=None, weight_decay=0, bn_momentum=0.99, bn_eps=1e-6, linearbn=False, _cnt=1):
    if dp:
        raise NotImplementedError
    if isinstance(num_layers, str):
        assert num_layers.startswith('mlp') or num_layers == 'linear', f'not supported num_layers = {num_layers}'
        assert num_layers != 'linear' or linear  # num_layers='linear' -> must have linear=True
        num_layers = int(num_layers[len('mlp'):]) if num_layers.startswith('mlp') and num_layers[len('mlp'):] else 1

    for cnt in range(_cnt, num_layers + _cnt - int(linear)):  # if linear - last mlp becomes linear, else add activation to all mlp(s)
        features = dense_layer(features, d_out, f'mlp_{cnt}', activation, True, bn, is_training, initializer, weight_decay=weight_decay, bn_momentum=bn_momentum, bn_eps=bn_eps)

    if linear:
        l_bn = linearbn  # if linearbn is not None else bn
        logits = dense_layer(features, num_classes, f'linear', None, True, l_bn, is_training, initializer, weight_decay=weight_decay, bn_momentum=bn_momentum, bn_eps=bn_eps)
    else:
        logits = features
    return logits

def reduce_neighbors(features, neighbor_idx, reduction):
    if reduction == 'mean':
        features, mask = tf_gather(features, neighbor_idx)  # [B x sample_num, kr, d]
        features = tf.reduce_sum(features, axis=-2) / tf.reduce_sum(mask, axis=-1, keepdims=True) if mask is not None else tf.reduce_mean(features, -2)
    elif reduction == 'max':
        features, mask = tf_gather(features, neighbor_idx, shadow_fn=tf.reduce_min(features, axis=0, keepdims=True))
        features = tf.reduce_max(sampled_f, axis=-2)  # [B x sample_num, d]
    else:
        raise NotImplementedError(f'not support reduction = {reduction}')
    return features


def check_mask(x, mask):
    """ check & try fix the mask to match the shape of x
    """
    if mask is not None and len(mask.shape) < len(x.shape):
        mask = tf.expand_dims(mask, axis=-1)
    if mask is not None:
        assert len(mask.shape) == len(x.shape), f'unmatched shape: mask.shape={mask.shape} v.s. x.shape={x.shape}'
    return mask


def normalize(x, norm_cfg, axis=None, mask=None):
    """
    Args:
        x   : [..., N, k, d]  - assume the last two axes being the spatial-feature dim
        mask: [..., N, k, /d] - valid mask
    """
    assert norm_cfg, f'norm_cfg = {norm_cfg}, should check for empty config outside the func'
    mask = check_mask(x, mask)

    _axis = axis
    for n in norm_cfg.split('-'):
        if _axis is None and n in ['l2', 'l1']: axis = -1
        elif _axis is None and n in ['softmax', 'norm', 'norml1']: axis = -2
        elif _axis is None and n in ['G_softmax'] :axis = -3

        if n == 'l2':  # channel
            unit = x * mask if mask is not None else x
            unit = tf.sqrt(tf.reduce_sum(unit ** 2, axis=axis, keepdims=True) + _eps)
            x = x / unit
        elif n == 'l1':  # channel
            unit = x * mask if mask is not None else x
            unit = tf.reduce_sum(tf.abs(unit), axis=axis, keepdims=True) + _eps
            x = x / unit
        elif n == 'softmax':  # spatial
            x = masked_softmax(x, mask, axis=axis, impl='inf') if mask is not None else tf.nn.softmax(x, axis=axis)
        elif n == 'norm':  # spatial
            unit = x * mask if mask is not None else x
            unit = tf.reduce_sum(unit, axis=axis, keepdims=True) + _eps  # potential sum to 0
            x = x / unit
        elif n == 'norml1':  # spatial - l1 (reduce to 'norm' if all positive)
            unit = x * mask if mask is not None else x
            unit = tf.reduce_sum(tf.abs(unit), axis=axis, keepdims=True) + _eps
            x = x / unit
        else:
            raise NotImplementedError(f'not supported norm: {n} in {norm_cfg}')

    if mask is not None and n not in ['softmax']:  # mitigate the last mask-out
        x *= mask
    return x
normalize.spatial = ['softmax', 'norm', 'norml1']
normalize.channel = ['l2', 'l1']
normalize.ops = normalize.spatial + normalize.channel

def combine(feature_list, ops, mask_list=None, kwargs=None, raise_not_support=True):
    # combine the features (assume has been matched) in the list
    num_f = len(feature_list)
    assert num_f > 1, f'len(feature_list) < 2 for feature_list = {feature_list}'
    rank = len(feature_list[0].shape)
    assert all([rank == len(f.shape) for f in feature_list]), f'unmatched rank for features in list: {feature_list}'
    kwargs = {} if kwargs is None else kwargs
    activation = kwargs.pop('activation') if 'activation' in kwargs else None

    full_shape = None
    if mask_list is not None:
        raise NotImplementedError
        mask_list = mask_list if isinstance(mask_list, list) else [mask_list] * num_f
        full_shape = tf.cast(tf.shape([m for m in mask_list if m is not None][0]), tf.int64)
        full_shape = tf.concat([full_shape, [int(feature_list[0].shape[-1])]], axis=0)  # [BxN (full mask), d (feature)]

    def scatter_feature_list(f_list, m_list, w=None):
        # scatter the feature in f_list back to original shape, according to mask in m_list
        w_list = [w[i] for i in range(len(f_list))] if w is not None else None
        if m_list is not None and w is not None:
            f_list = [f if m is None else tf.scatter_nd(tf.where(m), updates=f*w, shape=full_shape) for w, f, m in zip(w_list, f_list, m_list)]
        elif m_list is not None and w is None:
            f_list = [f if m is None else tf.scatter_nd(tf.where(m), updates=f, shape=full_shape) for f, m in zip(f_list, m_list)]
        elif m_list is None and w is not None:
            f_list = [f*w for w, f in zip(w_list, f_list)]
        return f_list

    if ops == 'concat':
        f_list = scatter_feature_list(feature_list, mask_list)
        features = tf.concat(f_list, axis=-1)

    elif ops == 'sum':
        f_list = scatter_feature_list(feature_list, mask_list)
        features = tf.add_n(f_list)

    elif ops == 'mul':
        f_list = scatter_feature_list(feature_list, mask_list)
        features = tf.reduce_prod(tf.stack(f_list), axis=0)

    elif ops.startswith('max'):
        f_list = scatter_feature_list(feature_list, mask_list)
        features = tf.reduce_max(tf.stack(f_list), axis=0)

    elif raise_not_support:
        raise NotImplementedError(f'not supported ops = {ops}')
    else:
        features = None
    return features

@tf_scope
def dropout(features, rate, is_training, noise_shape=None):
    """ Dropout layer.
    Args:
        noise_shape: None / list of ints
    """
    features = tf.cond(is_training,
        true_fn=lambda: tf.nn.dropout(features, rate=rate, noise_shape=noise_shape),
        false_fn=lambda: features,
    )
    return features



def get_boundary_mask(labels, neighbor_label=None, neighbor_idx=None, valid_mask=None, get_plain=False, get_cnt=False):
    """ assume all label valid indicated by valid_mask """
    labels_shape = labels.shape
    if neighbor_label is None:
        neighbor_label = tf_gather(labels, neighbor_idx, get_mask=False, shadow_fn=-1)  # label -1 at invalid places

    valid_neighbor = tf.greater_equal(neighbor_label, 0)
    labels = tf.expand_dims(labels, axis=-1)

    neq = tf.not_equal(labels, neighbor_label)  # same - F, diff - T, invalid center - T, invalid neighbor - T
    neq = tf.logical_and(neq, valid_neighbor)
    if get_cnt:
        bound = tf.reduce_sum(tf.cast(neq, tf.int32), axis=-1)
        bound = bound * tf.cast(valid_mask, tf.int32) if valid_mask is not None else bound
    else:
        bound = tf.reduce_any(neq, axis=-1)
        bound = tf.logical_and(bound, valid_mask) if valid_mask is not None else bound  # mask out row of invalid center
    assert len(bound.shape) == len(labels_shape), f'invalid shape - bound {bound.shape}, label {labels_shape}, neighbor label {neighbor_label.shape}, with valid_mask = {valid_mask}'

    if get_plain:
        assert not get_cnt, 'no need to get plain if having cnt of boundary (together with valid neighbor)'
        eq = tf.equal(labels, neighbor_label)  # same - T, diff - F, invalid center - F, invalid neighbor - F
        eq = tf.logical_or(eq, tf.logical_not(valid_neighbor))  # valid -> all eq => plain if neighbor all invalid
        plain = tf.reduce_all(eq, axis=-1)
        plain = tf.logical_and(plain, valid_mask) if valid_mask is not None else plain
        return bound, plain
    return bound  # [BxN]
