import os, re, sys, copy, warnings
import tensorflow as tf
if tf.__version__.split('.')[0] == '2':
    tf = tf.compat.v1
    tf.disable_v2_behavior()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

from collections import defaultdict
from config import log_config, load_config
from utils.logger import print_dict
from .heads import resnet_scene_segmentation_head, apply_head_ops
from .backbone import resnet_backbone
from .utils import tf_scope
from .basic_operators import *

class Model(object):

    def get_inputs(self, inputs):
        config = self.config
        if isinstance(inputs, dict):
            pass
        else:
            flat_inputs = inputs
            self.inputs = dict()
            self.inputs['points'] = flat_inputs[:config.num_layers]
            self.inputs['neighbors'] = flat_inputs[config.num_layers:2 * config.num_layers]
            self.inputs['pools'] = flat_inputs[2 * config.num_layers:3 * config.num_layers]
            self.inputs['upsamples'] = flat_inputs[3 * config.num_layers:4 * config.num_layers]
            ind = 4 * config.num_layers
            self.inputs['features'] = flat_inputs[ind]
            ind += 1
            self.inputs['batch_weights'] = flat_inputs[ind]
            ind += 1
            self.inputs['in_batches'] = flat_inputs[ind]
            ind += 1
            self.inputs['out_batches'] = flat_inputs[ind]
            ind += 1
            self.inputs['point_labels'] = flat_inputs[ind]
            ind += 1
            self.inputs['augment_scales'] = flat_inputs[ind]
            ind += 1
            self.inputs['augment_rotations'] = flat_inputs[ind]
            ind += 1
            self.inputs['point_inds'] = flat_inputs[ind]
            ind += 1
            self.inputs['cloud_inds'] = flat_inputs[ind]
            inputs = self.inputs
        inputs['sample_idx'] = {
            'down': inputs['pools'],
            'up': inputs['upsamples']
        }
        if 'batches_len' in inputs:
            inputs['batches_ind'] = [inputs['in_batches']] + [None] * (config.num_layers - 2) + [inputs['out_batches']]
        inputs['_glb'] = {}  # per-model/device global storage
        # inputs['assert_ops'] = []
        return inputs

    def get_result(self):
        # keys=['logits', 'probs', 'labels']
        # head_rst = {h: {k: d[k] for k in keys if k in d} for h, d in self.head_dict['result'].items()}
        head_rst = self.head_dict['result']
        rst = {  # {head/task: {probs, labels}, ..., 'inputs': input related}
            **head_rst,
            'inputs': {
                'point_inds': self.inputs['point_inds'],
                'cloud_inds': self.inputs['cloud_inds'],                
            }
        }
        for k in ['batches_len']:
            if k in self.inputs:
                rst['inputs'][k] = self.inputs[k]
        return rst

    def get_loss(self):
        return self.loss_dict

    """
    TODO: to check - multiple keys indexing the inputs['point_labels'] should be having the same id in rst - ensure only one tensor passed from gpu to cpu <=
    """

    @tf_scope
    def build_head(self, head_list, verbose=True):

        # building ouput heads & losses
        head_dict = self.inputs['head_dict'] if 'head_dict' in self.inputs else {'loss': {}, 'result': {}, 'config': {}}
        head_list = head_list if isinstance(head_list, (tuple, list)) else [head_list]
        head_list = [load_config(dataset_name='head', cfg_name=h) if isinstance(h, str) else h for h in head_list]

        if verbose:
            print('\n\n==== arch output')
        for head_cfg in head_list:
            if verbose:
                log_config(head_cfg)
                # if self.config.debug:
                #     print_dict(self.inputs)
            with tf.variable_scope(f'output/{head_cfg.head_n}'):
                head_rst = apply_head_ops(self.inputs, head_cfg, self.config, self.is_training)
            if verbose:
                print_dict(head_rst)

            # loss
            head_k = head_cfg.task if head_cfg.task else head_cfg.head_n  # head for specified task, or head_n as key by default
            loss_keys = ['loss',]
            for k in loss_keys:
                head_rst_d = head_rst[k] if isinstance(head_rst[k], dict) else {head_k: head_rst[k]}  # use returned dict if provided
                joint = head_dict[k].keys() & head_rst_d.keys()
                assert len(joint) == 0, f'head rst {k} has overlapping keys {joint}'
                head_dict[k].update(head_rst_d)
            # result
            rst_keys = ['logits', 'probs', 'labels',]
            head_rst_d = {k: head_rst[k] for k in head_rst if k not in loss_keys}
            assert head_cfg.head_n not in head_dict['result'], f'duplicate head {head_cfg.head_n} in dict'
            assert set(head_rst_d.keys()).issuperset(set(rst_keys)), f'must include keys {rst_keys}, but given {head_rst_d.keys()}'
            head_dict['result'][head_cfg.head_n] = head_rst_d
            if head_k and head_k != head_cfg.head_n:  # get the task head - flat & overridable
                if head_k in head_dict['result']:
                    warnings.warn(f'duplicate task head {head_k} in dict, override by {head_cfg.head_n}')
                head_dict['result'][head_k] = {k: head_rst_d[k][head_k] if isinstance(head_rst_d[k], dict) else head_rst_d[k] for k in head_rst_d}
            # config
            head_dict['config'][head_cfg.head_n] = head_cfg
            head_dict['config'][head_k] = head_cfg

        if verbose:
            print('\n\n')
        return head_dict


    @tf_scope
    def build_loss(self, scope=None, head_dict=None):
        # finalizing loss_dict
        if head_dict is None:
            head_dict = self.head_dict
        loss_dict = head_dict['loss']
        sum_fn = tf.accumulate_n if len(self.config.gpu_devices) else tf.add_n  # accumulate_n seems not working with cpu-only???

        # get the collection, filtering by 'scope'
        l2_loss = tf.get_collection('weight_losses', scope)
        if l2_loss:
            loss_dict['l2_loss'] = sum_fn(l2_loss, name='l2_loss')  # L2

        # sum total loss
        loss = sum_fn(list(loss_dict.values()), name='loss')

        # reconstruct loss dict - reorder & incldue total loss
        main_n = {'seg': ['S3DIS', 'ScanNet', 'Semantic3D', 'NPM3D']}
        main_n = {v: k for k, lst in main_n.items() for v in lst}[self.config.dataset]
        loss_dict = {
            'loss': loss,
            # # should have one and only one 'main' loss
            # # TODO: may introduce cls & seg head at the same time? => each task a main?
            # main_n: loss_dict.pop(main_n),
            **loss_dict,
        }
        head_dict['loss'] = loss_dict
        return loss_dict

class SceneSegModel(Model):
    def __init__(self, flat_inputs, is_training, config, scope=None, verbose=True):
        self.config = config
        self.is_training = is_training
        self.scope = scope
        self.verbose = verbose

        with tf.variable_scope('inputs'):
            self.inputs = self.get_inputs(flat_inputs)

            self.num_layers = config.num_layers
            self.labels = self.inputs['point_labels']
            self.down_list = [{'p_sample': None, 'f_sample': None, 'p_out': None, 'f_out': None} for i in range(self.num_layers)]
            self.up_list = [{'p_sample': None, 'f_sample': None, 'p_out': None, 'f_out': None} for i in range(self.num_layers)]
            self.stage_list = self.inputs['stage_list'] = {'down': self.down_list, 'up': self.up_list}
            self.head_dict = self.inputs['head_dict'] = {'loss': {}, 'result': {}, 'config': {}}

            for i, p in enumerate(self.inputs['points']):  # fill points
                self.down_list[i]['p_out'] = p
                # up 0 = the most upsampled, num_layers-1 the upsampled pt from the most downsampled
                self.up_list[i]['p_out'] = p if i < self.num_layers - 1 else None

        if config.dense_by_conv:
            dense_layer.config = config

        with tf.variable_scope('model'):
            fdim = config.first_features_dim
            r = config.first_subsampling_dl * config.density_parameter
            features = self.inputs['features']

            F = resnet_backbone(config, self.inputs, features, base_radius=r, base_fdim=fdim,
                                bottleneck_ratio=config.bottleneck_ratio, depth=config.depth,
                                is_training=is_training, init=config.init, weight_decay=config.weight_decay,
                                activation_fn=config.activation_fn, bn=True, bn_momentum=config.bn_momentum,
                                bn_eps=config.bn_eps)

            F_up, head = resnet_scene_segmentation_head(config, self.inputs, F, base_fdim=fdim,
                                                        is_training=is_training, init=config.init,
                                                        weight_decay=config.weight_decay,
                                                        activation_fn=config.activation_fn,
                                                        bn=True, bn_momentum=config.bn_momentum, bn_eps=config.bn_eps)

            for i, p in enumerate(self.inputs['points']):  # fill features
                self.down_list[i]['f_out'] = F[i]
                # F_up reversed - 0 = the most upsampled, num_layers-1 the upsampled pt from the most downsampled
                self.up_list[i]['f_out'] = F_up[i] if i < len(F_up) else None
            self.up_list[-1] = self.down_list[-1]  # align the most-downsampled layer
            if head is not None:
                latent, logits = head
                self.up_list[0]['latent'] = latent
                self.up_list[0]['logits'] = logits

            self.head_dict = self.build_head(self.config.arch_out, verbose=verbose)
            self.loss_dict = self.build_loss(scope)
        return


def get_model(flat_inputs, is_training, config, scope=None, verbose=True):

    model = get_inference_model(flat_inputs, is_training, config, scope=scope, verbose=verbose)
    return model

def get_inference_model(flat_inputs, is_training, config, scope=None, verbose=True):
    Model = globals()[config.builder]
    return Model(flat_inputs, is_training, config, scope=scope, verbose=verbose)
