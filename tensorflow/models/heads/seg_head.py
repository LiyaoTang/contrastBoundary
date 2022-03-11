import tensorflow as tf
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
ROOT_DIR = os.path.join(BASE_DIR, '..')
sys.path.insert(0, ROOT_DIR)

from ..local_aggregation_operators import *


def nearest_upsample_block(layer_ind, inputs, features, scope):
    """
    This Block performing an upsampling by nearest interpolation
    Args:
        layer_ind: Upsampled to which layer
        inputs: a dict contains all inputs
        features: x = [n1, d]
        scope: name scope

    Returns:
        x = [n2, d]
    """

    with tf.variable_scope(scope) as sc:
        upsampled_features = ind_closest_pool(features, inputs['upsamples'][layer_ind], 'nearest_upsample')
        return upsampled_features


def resnet_scene_segmentation_head(config,
                                   inputs,
                                   F,
                                   base_fdim,
                                   is_training,
                                   init='xavier',
                                   weight_decay=0,
                                   activation_fn='relu',
                                   bn=True,
                                   bn_momentum=0.98,
                                   bn_eps=1e-3):
    """A head for scene segmentation with resnet backbone.

    Args:
        config: config file
        inputs: a dict contains all inputs
        F: all stage features
        base_fdim: the base feature dim
        is_training: True indicates training phase
        init: weight initialization method
        weight_decay: If > 0, add L2Loss weight decay multiplied by this float.
        activation_fn: Activation function
        bn: If True, add batch norm after convolution

    Returns:
        prediction logits [num_points, num_classes]
    """
    F_up = []
    with tf.variable_scope('resnet_scene_segmentation_head') as sc:
        fdim = base_fdim
        features = F[-1]

        features = nearest_upsample_block(4, inputs, features, 'nearest_upsample_0')
        features = tf.concat((features, F[3]), axis=1)
        features = conv1d_1x1(features, 8 * fdim, 'up_conv0', is_training=is_training, with_bias=False, init=init,  # 2^3 * fdim
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)
        F_up.append(features)

        features = nearest_upsample_block(3, inputs, features, 'nearest_upsample_1')
        features = tf.concat((features, F[2]), axis=1)
        features = conv1d_1x1(features, 4 * fdim, 'up_conv1', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)
        F_up.append(features)

        features = nearest_upsample_block(2, inputs, features, 'nearest_upsample_2')
        features = tf.concat((features, F[1]), axis=1)
        features = conv1d_1x1(features, 2 * fdim, 'up_conv2', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)
        F_up.append(features)

        features = nearest_upsample_block(1, inputs, features, 'nearest_upsample_3')
        features = tf.concat((features, F[0]), axis=1)
        features = conv1d_1x1(features, fdim, 'up_conv3', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)

        F_up.append(features)
        F_up = list(reversed(F_up))

        if config.sep_head or config.arch_up:
            # build head with config.arch_out
            return F_up, None

        features = conv1d_1x1(features, fdim, 'segmentation_head', is_training=is_training, with_bias=False, init=init,
                              weight_decay=weight_decay, activation_fn=activation_fn, bn=bn, bn_momentum=bn_momentum,
                              bn_eps=bn_eps)
        logits = conv1d_1x1(features, config.num_classes, 'segmentation_pred', is_training=is_training, with_bias=True,
                            init=init, weight_decay=weight_decay, activation_fn=None, bn=False)
    return F_up, (features, logits)
