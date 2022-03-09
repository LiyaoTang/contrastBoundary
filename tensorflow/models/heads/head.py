import tensorflow as tf
if tf.__version__.split('.')[0] == '2':
    tf = tf.compat.v1
    tf.disable_v2_behavior()

import re
from .utils import *
from .basic_operators import *
from .basic_operators import _eps, _inf
fetch_supports = fetch_supports_stage

# ---------------------------------------------------------------------------- #
# heavy & re-uesable func
# ---------------------------------------------------------------------------- #

def get_scene_label(*args, infer=None, **kwargs):
    infer = infer if infer else 'infer'
    func = {
        'infer': get_scene_label_infer,
        'recur': get_scene_label_recursive,
        'nst': get_scene_label_nearest,
    }[infer]
    return func(*args, **kwargs)

def get_scene_label_infer(inputs, stage_n, stage_i, ftype, config, reduction='max', extend=False):
    """ collect label for sub-sampled points via accumulated sampling ratio
    """
    scene_neighbor = get_sample_idx(inputs, 'U0', (stage_n, stage_i), ftype, config)  # NOTE: must have ftype at U0
    if scene_neighbor is None and extend:  # extend first-layer (no sub-sampling) with neighbor_idx
        scene_neighbor = inputs['neighbors'][stage_i]
    if scene_neighbor is None:
        return None

    _glb = inputs['_glb']
    ftype, ptype = get_ftype(ftype)
    key = f'{stage_n}{stage_i}/{ptype}/scene_label-{reduction}'
    if key not in _glb:
        # gather labels
        key_gather = f'{stage_n}{stage_i}/{ptype}/scene_label-gather'
        if key_gather not in _glb:
            labels = tf_gather(inputs['point_labels'], scene_neighbor, shadow_fn=-1, get_mask=False)  # [BxN, k] - invalid label (-1) one_hot to 0s
            valid_mask = tf.greater_equal(labels, 0)
            labels = tf.one_hot(labels, depth=config.num_classes, axis=-1)
            _glb[key_gather] = (labels, valid_mask)
        labels, valid_mask = _glb[key_gather]
        # summarize labels
        labels = get_neighbor_summary(tf.cast(labels, tf.float32), valid_mask=valid_mask, reduction=reduction)
        _glb[key] = labels
    return _glb[key]  # [BxN, 1/num_classes]

def get_scene_label_recursive(inputs, stage_n, stage_i, ftype, config, reduction='soft', extend=False):

    key_n = f'scene_label_recur-{reduction}'
    key = f'{stage_i}/{key_n}'
    _glb = inputs['_glb']

    if stage_i > 0:
        # fetch from last layer
        scene_neighbor = inputs['sample_idx']['down'][stage_i - 1]
        last_labels = get_scene_label_recursive(inputs, stage_n, stage_i - 1, ftype, config, reduction=reduction, extend=extend)
    elif extend:
        # extend first-layer (no sub-sampling) with neighbor_idx
        scene_neighbor = inputs['neighbors'][stage_i]
        last_labels = None
    else:
        # first stage with no extend
        return None
    if key not in _glb:
        # gather labels
        if last_labels is None:  # stage-1 (last stage 0) / stage-0 with extend
            labels = tf_gather(inputs['point_labels'], scene_neighbor, shadow_fn=-1, get_mask=False)
            valid_mask = tf.greater_equal(labels, 0)
            labels = tf.one_hot(labels, depth=config.num_classes, axis=-1)
        else:
            labels, valid_mask = tf_gather(last_labels, scene_neighbor)  # [BxN, k, ncls]
        # summarize labels
        labels = get_neighbor_summary(tf.cast(labels, tf.float32), valid_mask=valid_mask, reduction=reduction)  # [BxN, 1/ncls]
        if labels.shape[-1] == 1:
            labels = tf.one_hot(tf.cast(tf.squeeze(labels, axis=-1), tf.int32), depth=config.num_classes, axis=-1)
        _glb[key] = labels
    return _glb[key]  # [BxN, num_classes]

def get_scene_label_nearest(inputs, stage_n, stage_i, ftype, config, reduction=None, extend=False):
    scene_neighbor = get_sample_idx(inputs, 'U0', (stage_n, stage_i), ftype, config, kr=1)  # NOTE: must have ftype at U0
    if scene_neighbor is None and extend:  # extend first-layer (no sub-sampling) with neighbor_idx
        scene_neighbor = inputs['neighbors'][stage_i][..., 0]
    if scene_neighbor is None:
        return None
    scene_neighbor = tf.expand_dims(scene_neighbor, axis=-1)
    labels = tf_gather(inputs['point_labels'], scene_neighbor, shadow_fn=-1, get_mask=False)  # [BxN, 1] - invalid label (-1) one_hot to 0s
    return labels


def get_scene_features(features, inputs, stage_from, stage_to, ftype, config, reduction='soft', mask=None, extend=False, name='features'):
    """ cross-stage sample-summarize features, assume features - [BxN, d]
    """
    n_from, i_from = parse_stage(stage_from, config.num_layers)[0] if isinstance(stage_from, str) else stage_from
    n_to, i_to = parse_stage(stage_to, config.num_layers)[0] if isinstance(stage_to, str) else stage_to

    scene_neighbor = get_sample_idx(inputs, stage_from, stage_to, ftype, config)
    if scene_neighbor is None and extend:  # same layer, but extend with neighbor_idx
        scene_neighbor = inputs['neighbors'][i_to]
    if scene_neighbor is None:
        return features

    _glb = inputs['_glb']
    key = f'{n_from}{i_from}-{n_to}{i_to}/scene_{name}'
    if key not in _glb:
        features, valid_mask = tf_gather(features, scene_neighbor)  # [BxN, k, d]
        if valid_mask is not None:
            mask = tf.logical_and(mask, valid_mask) if mask is not None else valid_mask
        features = get_neighbor_summary(features, valid_mask=mask, reduction=reduction)
        _glb[key] = features
    return _glb[key]  # [BxN, 1/num_classes]


def get_neighbor_summary(neighbor_label, valid_mask=None, reduction=None):
    # assume neighbor_label is one-hot - [BxN, k, num_classes]
    if reduction in ['soft', 'mean']:  # soft - [BxN, num_classes]
        if valid_mask is None:
            labels = tf.reduce_mean(neighbor_label, axis=-2)
        else:
            labels = tf.reduce_sum(neighbor_label, axis=-2)  # assume filling 0s
            labels = labels / (tf.reduce_sum(tf.cast(valid_mask, tf.float32), axis=-1, keepdims=True) + _eps)
    elif reduction in ['cnt', 'max']:  # cnt - [BxN, 1]
        labels = tf.reduce_sum(neighbor_label, axis=-2, keepdims=True)  # [BxN, 1, num_classes]
        with tf.device('/cpu:0'):
            labels = tf.argmax(labels, axis=-1)  # [BxN, 1] - argmax only on cpu
    else:
        raise NotImplementedError(f'not supported label reduction = {reduction}')
    return labels

def get_sample_idx(inputs, stage_from, stage_to, ftype, config, kr=None):
    """collect neighbors from up/sub-sampled points, by accumulated sub-sampling ratio
        => neighbor_idx indexing stage_from pts - [stage_to pts, num of neighbor in stage_from pts]
        i.e. stage_from = support, stage_to = queries
    """
    _, ptype = get_ftype(ftype)
    # assert ftype != 'f_sample', f'not considering to use p_sample in cross-stage sampling'
    n_from, i_from = parse_stage(stage_from, config.num_layers)[0] if isinstance(stage_from, str) else stage_from
    n_to, i_to = parse_stage(stage_to, config.num_layers)[0] if isinstance(stage_to, str) else stage_to

    pts_from = inputs['stage_list'][n_from][i_from][ptype]
    pts_to = inputs['stage_list'][n_to][i_to][ptype]

    if i_from - i_to == 0:  # no sample
        assert pts_from == pts_to, f'pts have changed from {stage_from} to {stage_to} ({ftype})'
        return None

    # upsampling: i_to closer to end - up0, compared with e.g. down4/up4
    # downsampling: i_to closer to last - down4, compared with e.g. down0
    updown = 'up' if i_to < i_from else 'down'
    # assert n_to == 'up' or n_from == n_to, f'not supported sampling from {stage_from} to {stage_to}'

    if abs(i_from - i_to) == 1:  # sample to next/last stage
        neighbor_idx = inputs['sample_idx'][updown][i_from]
        return neighbor_idx[..., 0] if kr == 1 else neighbor_idx[...,:kr] if isinstance(kr, int) else neighbor_idx

    # down/up-sample more than 1 stage
    _glb = inputs['_glb']
    key = f'{n_from}{i_from}-{n_to}{i_to}/{ftype}/sample_neighbor'
    if key not in _glb:

        from ops import get_tf_func
        search_func = get_tf_func(config.search)
        # search kr depending on (cumulated) radius/ratio in sub-sampling points

        # grid - down: r_sample[i_to - 1]; up: r_sample[i_from - 1]  => using the larger one
        # knn - donw: r_sample[i_from:i_to]; up: r_sample[i_to:i_from]  => smaller to larger
        i_min, i_max = min(i_from, i_to), max(i_from, i_to)

        kr_search = config.r_sample[i_max - 1] if config.sample == 'grid' else kr if kr else np.prod(config.r_sample[i_min:i_max])
        args = [inputs['batches_len'][i_to], inputs['batches_len'][i_from], kr_search] if config.sample == 'grid' else [kr_search]
        neighbor_idx = search_func(pts_to, pts_from, *args, device='/cpu:0')  # queries, supports, ...

        _glb[key] = neighbor_idx[..., 0] if kr == 1 else neighbor_idx
    return _glb[key]


def calc_dist(f_from, f_to, dist, align=True, keepdims=True):
    # f_from / f_to - prepared & normalized in advance
    assert dist in calc_dist.valid
    if dist in ['l2', 'norml2']:
        dist = tf.reduce_sum((f_from - f_to)**2, axis=-1, keepdims=keepdims)
        dist = tf.sqrt(tf.maximum(dist, _eps))  # avoid 0-distance - nan due to tf.sqrt numerical unstable at close-0
        # dist = tf.where(tf.greater(dist, 0.0), tf.sqrt(dist), dist)
    elif dist in ['l2square']:
        dist = tf.reduce_sum((f_from - f_to)**2, axis=-1, keepdims=keepdims)
    elif dist in ['l1', 'norml1']:
        dist = tf.reduce_sum(tf.abs(f_from - f_to), axis=-1, keepdims=keepdims)
    elif dist in ['dot', 'normdot']:  # normdot = raw cos
        dist = tf.reduce_sum(f_from * f_to, axis=-1, keepdims=keepdims)
        dist = -dist if align else dist  # revert to (-inf i.e. small <- similar, dissim -> inf i.e. large)
    elif dist == 'cos':
        dist = tf.reduce_sum(f_from * f_to, axis=-1, keepdims=keepdims)
        # NOTE: matmul seems incorrectly used - training diverge
        # dist = tf.matmul(f_to, f_from, transpose_b=True)  # [k, 1] = [k, d] @ [1, d]^T - similar = smaller
        dist = -dist if align else dist  # revert to (-1 <- similar, dis-similar -> 1)
        dist = (1 + dist) / 2  # rescale to (0, 1)
    elif dist == 'kl':
        # f_from/to need to be a distribution - (0 <- sim, dis -> inf)
        dist = tf.reduce_sum(tf.math.xlogy(f_from, f_from / tf.maximum(f_to, _eps)), axis=-1, keepdims=keepdims)
    else:
        raise NotImplementedError(f'not supported dist = {dist}')
    return dist  # [BxN, k, 1] / [BxN, 1]
calc_dist.valid = ['cos', 'l2', 'norml2', 'l1', 'norml1', 'dot', 'normdot', 'l2square', 'kl']

def calc_loss(loss, labels, logits, config, num_classes=None, mask=None, name=None, reduce_loss=True, raise_not_support=True):
    name = name if name else 'cross_entropy'

    labels = tf.stop_gradient(labels)
    def masking(labels, logits, mask):
        labels = tf.boolean_mask(labels, mask) if mask is not None else labels
        logits = tf.boolean_mask(logits, mask) if mask is not None else logits
        return labels, logits

    if loss.startswith('softmax') or loss == 'xen':  # pixel-level loss
        # flatten to avoid explicit tf.assert in cross_entropy, which supports cpu only
        labels, logits = masking(labels, logits, mask)
        if len(labels.shape) == len(logits.shape):  # one-hot label
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits, name=name)
        else:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(labels, [-1]),
                                                                logits=tf.reshape(logits, [-1, int(logits.shape[-1])]),
                                                                name=name)
    elif loss.startswith('sigmoid'):  # pixel-level loss
        labels, logits = masking(labels, logits, mask)
        num_classes = num_classes if num_classes else config.num_classes
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(tf.reshape(labels, [-1]), logits.dtype),
                                                    logits=tf.reshape(logits, [-1]),
                                                    name=name)
    elif raise_not_support:
        raise NotImplementedError(f'not supported loss = {loss}')
    else:
        loss = None

    if loss is not None and reduce_loss:
        loss = tf.reduce_mean(loss)
    return loss


# ---------------------------------------------------------------------------- #
# pred head & loss
# ---------------------------------------------------------------------------- #


class mlp_head(object):
    valid_weight = ['center', 'class', 'batch']  # valid spcial weight (str)

    def __call__(self, inputs, head_cfg, config, is_training):
        # assert head_cfg.ftype == None and head_cfg.stage == None
        drop = head_cfg.drop if head_cfg.drop else config.drop
        logits, labels, loss = mlp_head.pred(inputs, 'up', 0, head_cfg, config, is_training, drop=drop)
        latent = inputs['stage_list']['up'][0]['latent']
        probs = tf.nn.softmax(logits, axis=-1)
        inputs['stage_list']['up'][0]['probs'] = probs
        return {'loss': loss, 'latent': latent, 'logits': logits, 'probs': probs, 'labels': labels}

    @staticmethod
    @tf_scope
    def get_branch_head(rst_dict, head_cfg, config, is_training, fkey, drop=None):
        # get features of current branch & storing into rst, upto probs
        ftype = get_ftype(fkey)[0]
        if ftype == 'f_out': return rst_dict['f_out']

        ftype_list = ['latent', 'logits', 'probs',]
        assert ftype in ftype_list, f'not supported ftype = {ftype} from fkey = {fkey}'
        if isinstance(drop, str):
            drop = re.search('\.\d+', drop)  # str float / None
            drop = float(drop.group) if drop else None

        d_out = head_cfg.d_out if head_cfg.d_out else config.first_features_dim
        num_classes = head_cfg.num_classes if head_cfg.num_classes else config.num_classes

        idx = max([ftype_list.index(k) + 1 if k in ftype_list else 0 for k in rst_dict])  # only consider fkey after existing keys
        if 'latent' not in rst_dict and ftype in ftype_list[idx:]:
            features = rst_dict['f_out']
            ops = head_cfg.mlp if head_cfg.mlp else fkey[len(ftype):] if fkey != ftype else 'mlp'
            kwargs = get_kwargs_mlp(head_cfg, config, is_training, _cnt=0)
            features = mlps_by_ops(features, d_out, d_out, ops=ops, kwargs=kwargs)
            rst_dict['latent'] = features
            idx += 1

        if drop is not None:
            features = rst_dict['latent']  # must have latent if applying dropout
            rst_dict['latent'] = dropout(features, rate=drop, is_training=is_training, name='dropout')

        if 'logits' not in rst_dict and ftype in ftype_list[idx:]:
            kwargs = get_kwargs(head_cfg, config, is_training)
            features = dense_layer(rst_dict['latent'], num_classes, f'linear', None, True, False, **kwargs)
            rst_dict['logits'] = features
            idx += 1

        if 'probs' not in rst_dict and ftype in ftype_list[idx:]:
            features = tf.nn.softmax(rst_dict['logits'], axis=-1)
            rst_dict['probs'] = features
            idx += 1

        return rst_dict[ftype]

    @staticmethod
    @tf_scope
    def pred(inputs, stage_n, stage_i, head_cfg, config, is_training, rst_dict=None, drop=None, labels='U0', labels_ext=False, reduce_loss=True):
        # TODO: may use rst_dict to pass on latent? - since may introduce mask-level supervision
        rst_dict = inputs['stage_list'][stage_n][stage_i] if rst_dict is None else rst_dict
        logits = mlp_head.get_branch_head(rst_dict, head_cfg, config, is_training, 'logits', drop=drop)

        # match pred (logits) to labels
        upsample_idx = get_sample_idx(inputs, f'{stage_n}{stage_i}', labels, head_cfg.ftype, config, kr=1)
        if upsample_idx is not None:
            logits = tf_gather(logits, upsample_idx, get_mask=False)
        full_logits = logits

        # get labels
        label_n, label_i = parse_stage(labels, config.num_layers)[0] if isinstance(labels, str) else labels
        labels = get_scene_label(inputs, label_n, label_i, head_cfg.ftype, config, reduction='soft', extend=labels_ext)
        labels = labels if labels is not None else inputs['point_labels']  # [BxN, /num_classes]
        full_labels = labels

        # collect mask
        mask = tf.greater_equal(labels, 0) if len(config.ignored_labels) > 0 else None
        if isinstance(head_cfg.weight, str) and 'center' in head_cfg.weight:  # retain sample center only
            raise NotImplementedError(f'not compatible with refine')
            pts = inputs['stage_list']['up']['p_out']
            dist = tf.reduce_sum(pts ** 2, axis=-1)  # [BxN] - squared l2 dist
            if 'in_batches' in inputs:
                dist, valid_mask = tf_gather(dist, inputs['in_batches'], shadow_fn=-1, get_mask=True)  # [B, N]
            thr = float(re.search('center\.\d+', head_cfg.weight).group(0)) * tf.reduce_max(dist, axis=-2, keepdims=True)  # [B, 1]
            thr_mask = tf.less(dist, thr ** 2)
            if 'in_batches' in inputs:
                thr_mask = tf.boolean_mask(thr_mask, valid_mask)  # [BxN]
            # update - crop
            mask = tf.logical_and(mask, thr_mask) if mask is not None else thr_mask
            full_logits = tf.boolean_mask(full_logits, thr_mask, names='center_logits')
            full_labels = tf.boolean_mask(full_labels, thr_mask, names='center_labels')
            if stage_n == 'up' and stage_i == 0:
                inputs['point_inds'] = tf.boolean_mask(inputs['point_inds'], thr_mask)
                inputs['stage_list']['up'][0]['p_out'] = tf.boolean_mask(pts, thr_mask)

        # match valid labels
        labels = tf.boolean_mask(labels, mask) if mask is not None else labels
        logits = tf.boolean_mask(logits, mask) if mask is not None else logits

        # calc loss
        loss = head_cfg.loss if head_cfg.loss else 'xen'  # default to softmax cross-entropy
        loss = calc_loss(loss, labels, logits, config, reduce_loss=False, raise_not_support=False)
        if loss is None and head_cfg.loss != 'none':
            raise NotImplementedError(f'not supported loss type = {head_cfg.loss}')

        # extra weight
        weight = None
        if isinstance(head_cfg.weight, float):
            weight = head_cfg.weight
        elif 'class' in head_cfg.weight:  # class weighting
            weight = get_class_weight(config.dataset, labels)
        elif 'batch' in head_cfg.weight:  # batch-size weighting => mean inside cloud, then over batches
            weight = inputs['batch_weights']
            weight = tf.boolean_mask(weight, mask) if mask is not None else weight
        elif 'center' in head_cfg.weight:
            pass
        elif head_cfg.weight.startswith('w'):
            weight = float(head_cfg.weight[1:])
        elif head_cfg.weight:
            raise NotImplementedError(f'not supported weight = {head_cfg.weight}')

        loss = loss * weight if weight is not None else loss
        if reduce_loss:
            loss = tf.reduce_mean(loss)
        return full_logits, full_labels, loss


class multiscale_head(object):
    f_list = {}

    def __call__(self, inputs, head_cfg, config, is_training):
        # config - per-branch ops
        branch = head_cfg.branch.split('-')
        branch_stages = branch[0]
        fkey = branch[1] if len(branch) > 1 else None
        drop = float(re.search('\.\d+', branch[2]).group()) if len(branch) > 2 else None
        stage_list = parse_stage(branch_stages, config.num_layers)
        stage_list = sorted(stage_list, key=lambda t: (t[0], (t[1] if t[0] == 'down' else -t[1])))  # D[0...5], U[5...0]
        assert all([t[0] == 'up' for t in stage_list]), f'not supported conditioning in downsampling stage yet'

        # config - conditioning
        condition = None
        if head_cfg.condition:
            cond_ops, cond_fkey = head_cfg.condition.split('-')
            cond_i = 1
            if cond_ops[-1].isdigit():
                cond_ops = cond_ops[:-1]
                cond_i = int(cond_ops[-1])
            elif cond_ops[-1] == 'A':
                cond_ops = cond_ops[:-1]
                cond_i = config.num_layers
            condition = (cond_ops, cond_i, cond_fkey)
            fkey = cond_fkey if fkey is None else fkey
        assert fkey is not None or not head_cfg.branch

        # apply per-branch ops & conditioning
        f_list = inputs['stage_list']  # TODO: use local dict?
        head_dict = {'logits': {}, 'probs': {}, 'loss': {'seg': None}, 'labels': {}}
        for n, i in stage_list:  # from input -> down -> up -> out
            key = f'mlp-{to_valid_stage(n)}{i}'
            with tf.variable_scope(key):
                if condition:  # combine the condition
                    self.combine_condition(inputs, f_list, stage_list, (n, i), head_cfg, config, is_training, condition, name='combine')

                # get current features to desired fkey
                if fkey.startswith('loss'):
                    labels = (n, i) if fkey.startswith('lossSub') else 'U0'
                    logits, labels, loss = mlp_head.pred(inputs, n, i, head_cfg, config, is_training, f_list[n][i], drop=drop, labels=labels)
                    head_dict['loss'][key] = loss
                    head_dict['logits'][key] = logits
                    head_dict['probs'][key] = tf.nn.softmax(logits, axis=-1)
                    head_dict['labels'][key] = labels
                else:
                    mlp_head.get_branch_head(f_list[n][i], head_cfg, config, is_training, fkey, drop=drop)

        # get main
        with tf.variable_scope('main'):
            if '-' in head_cfg.main:
                # stand-along branch
                comb_st, comb_ops, comb_fkey = head_cfg.main.split('-')[:3]
                comb_drop = head_cfg.main.split('-')[3] if len(head_cfg.main.split('-')) > 3 else None
                comb_st = parse_stage(comb_st, config.num_layers)
                comb_fype = get_ftype(comb_fkey)[0]
                feat_list = self.collect_and_match(inputs, f_list, comb_st, comb_fkey, 'U0', head_cfg, config, is_training)
                assert len(feat_list) > 1
                features = combine(feat_list, comb_ops, kwargs=get_kwargs(head_cfg, config, is_training, act=True))
                if comb_fype not in head_dict:  # may be fout/latent
                    head_dict[comb_fype] = {'seg': features}
                logits, _, loss = mlp_head.pred(inputs, 'up', 0, head_cfg, config, is_training, rst_dict={comb_fype: features}, drop=comb_drop)
                probs = tf.nn.softmax(logits, axis=-1)
            else:
                # using existing branch
                stage = parse_stage(head_cfg.main, config.num_layers)
                assert len(stage) == 1, f'specified main stage = {stage}, (either empty or more than one)'
                n, i = stage[0]
                key = f'mlp-{to_valid_stage(n)}{i}'
                if key in head_dict['loss']:
                    logits, probs, loss = head_dict['logits'].pop(key), head_dict['probs'].pop(key), head_dict['loss'].pop(key)
                else:
                    logits, _, loss = mlp_head.pred(inputs, n, i, head_cfg, config, is_training, rst_dict=f_list[n][i])
                    probs = tf.nn.softmax(logits, axis=-1)

        # apply loss weight - not using existing branch as main branch
        if fkey and fkey.startswith('loss'):
            loss_w = float(re.search('\.\d+', fkey).group())
            for loss_k, v in head_dict['loss'].items():
                if v is not None:
                    head_dict['loss'][loss_k] *= loss_w
        head_dict['logits']['seg'] = logits
        head_dict['probs']['seg'] = probs
        head_dict['loss']['seg'] = loss
        head_dict['labels']['seg'] = inputs['point_labels']

        return head_dict

    @tf_scope
    def combine_condition(self, inputs, f_list, stage_list, stage, head_cfg, config, is_training, condition):
        n, i = stage
        rst_dict = f_list[n][i]

        # get previous features & upsample
        cond_ops, cond_i, cond_fkey = condition
        idx = stage_list.index(stage)  # current idx
        idx_list = list(range(max(0, idx - cond_i), idx))
        stage_list = [stage_list[i] for i in idx_list]
        feat_list = self.collect_and_match(inputs, f_list, stage_list, cond_fkey, stage, head_cfg, config, is_training)

        # get current till cond_fkey
        features = mlp_head.get_branch_head(rst_dict, head_cfg, config, is_training, cond_fkey)

        # combine all
        feat_list = [*feat_list, features]
        kwargs = get_kwargs(head_cfg, config, is_training)
        features = combine(feat_list, cond_ops, kwargs=kwargs) if len(feat_list) > 1 else feat_list[0]

        # update combined
        rst_dict[cond_fkey] = features
        return features

    def collect_and_match(self, inputs, f_list, stage_list, f_key, stage, head_cfg, config, is_training):
        # collect from f from f_list[stage_list] and match to stage
        feat_list = []
        for n, i in stage_list:
            f = mlp_head.get_branch_head(f_list[n][i], head_cfg, config, is_training, f_key, name=f'{n}{i}')
            upsample_idx = get_sample_idx(inputs, stage_from=(n, i), stage_to=stage, ftype=head_cfg.ftype, config=config, kr=1)
            if upsample_idx is not None:
                f = tf.gather(f, upsample_idx)
            feat_list += [f]
        return feat_list


class contrast_head(object):
    def __call__(self, inputs, head_cfg, config, is_training):
        head_dict = {'logits': {}, 'probs': {}, 'loss': {}, 'labels': {}}
        fhead, ftype = head_cfg.ftype.split('-') if '-' in head_cfg.ftype else (None, head_cfg.ftype)
        ftype = get_ftype(ftype)[0]
        for n, i in parse_stage(head_cfg.stage, config.num_layers):
            # sample pos-neg
            samples = contrast_head.sample_labels(inputs, n, i, head_cfg.sample, ftype, config, name=f'{n}{i}/sample')
            samples = (*samples, n, i)
            # get the features from existing head if specified
            stage = inputs['head_dict']['result'][fhead] if fhead is not None else inputs['stage_list'][n][i]
            features = stage[ftype] if ftype in stage and stage[ftype] is not None else mlp_head.get_branch_head(stage, head_cfg, config, is_training, ftype, name=f'{n}{i}')
            if head_cfg.project:
                # projection head - contrast on projected features
                kwargs = get_kwargs_mlp(head_cfg, config, is_training)
                features = mlps_by_ops(features, features.shape[-1], features.shape[-1], head_cfg.project, kwargs=kwargs, name=f'{n}{i}-proj')
            for contrast in head_cfg.contrast.split('-'):  # may have multiple contrast - e.g. sim-diff
                rst = contrast_head.contrast(features, samples, contrast, inputs, head_cfg, config, name=f'{n}{i}/{contrast}')
                for k in ['logits', 'probs', 'loss', 'labels']:
                    head_dict[k][f'{contrast}-{n}{i}'] = rst[k]
        return head_dict

    @staticmethod
    @tf_scope
    def collect_labels(inputs, stage_n, stage_i, ftype, config, neighbor_idx, criterion, annotation='label'):
        mask = None  # valid mask
        # get label
        infer = criterion.split('_')[-1] if '_' in criterion else ''
        assert infer in ['', 'nst', 'recur', 'recurhard']
        criterion = criterion[len('label'):]
        if criterion and re.match('[a-z]+', criterion):
            # e.g. labelkl / labelabs
            mask_c = mask_n = None
            labels = get_scene_label(inputs, stage_n, stage_i, ftype, config, reduction='soft', extend=False, infer=infer)
            if labels is None:
                labels = tf.cast(tf.one_hot(inputs['point_labels'], depth=config.num_classes), tf.float32)  # [BxN, num_cls]
            if config.ignored_labels:
                mask_c = tf.greater_equal(labels, 0)

            neighbor_label = tf_gather(labels, neighbor_idx, shadow_fn=0, get_mask=bool)  # [BxN, k, num_cls] - 0 to be used in dist meastures
            if isinstance(neighbor_label, tuple):
                neighbor_label, mask_n = neighbor_label
            
            if mask_c is None and mask_n is not None:
                mask = mask_n
            elif mask_c is not None and mask_n is None:
                mask = tf.expand_dims(mask_c, axis=-1)
            elif mask_c is not None and mask_n is not None:
                mask = tf.logical_and(mask_c, mask_n)

            labels = tf.expand_dims(labels, axis=-2)
            if criterion.startswith('kl'):  # xlogy=0 if y=0, but need to have large value if labels=1 and neighbor_label=0
                if criterion.startswith('klR'):
                    dist = calc_dist(neighbor_label, labels, dist='kl', keepdims=False)
                else:
                    dist = calc_dist(labels, neighbor_label, dist='kl', keepdims=False)
                criterion = re.search('\.{0,1}\d+', criterion)
                criterion = float(criterion.group()) if criterion else None
            elif criterion.startswith('abs'):
                dist = tf.reduce_sum(tf.abs(labels - neighbor_label), axis=-1)
                criterion = float(criterion[3:])
            else:
                raise NotImplementedError(f'not supported criterion = {criterion}')
            posneg = tf.less(dist, criterion) if criterion is not None else dist

        else:
            # gather as hard (discrete) label
            reduction = 'max'
            if infer == 'recur':  # recursive with soft label
                reduction = 'soft'
            elif infer == 'recurhard':  # recursive with hard label
                infer = 'recur'
            # default: infering with hard (argmax) label
            labels = get_scene_label(inputs, stage_n, stage_i, ftype, config, reduction=reduction, extend=False, infer=infer)  # None / [BxN, 1/ncls]
            if labels is None:
                labels = inputs['point_labels']
            if labels.shape[-1] == config.num_classes:
                with tf.device('/cpu:0'):
                    labels = tf.argmax(labels, axis=-1)  # [BxN, 1] - argmax only on cpu
            if labels.shape[-1] == 1:
                labels = tf.squeeze(labels, axis=-1)
            # posneg from discrete neighbors
            neighbor_label = tf_gather(labels, neighbor_idx, shadow_fn=-1, get_mask=False)  # [BxN, k]
            posneg = tf.equal(tf.expand_dims(labels, axis=-1), neighbor_label)

            if config.search == 'radius' or config.ignored_labels:
                mask_n = tf.greater_equal(neighbor_label, 0)  # [BxN, k] - not consider invalid point as neighbors
                mask_c = tf.greater_equal(labels, 0) if config.ignored_labels else None  # [BxN, 1] - not consider invalid point as center
                mask = tf.logical_and(mask_c, mask_n) if mask_c is not None else mask_n
        return posneg, mask

    @staticmethod
    @tf_scope
    def sample_labels(inputs, stage_n, stage_i, sample, ftype, config):

        # sampling pos-neg - changing the neighbor_idx for collection
        _glb = inputs['_glb']
        key = f'{stage_n}{stage_i}/{sample}'
        if key in _glb:
            return _glb[key]

        # get label
        # labels = get_scene_label(inputs, stage_n, stage_i, ftype, config, reduction='max', extend=False)
        # labels = tf.squeeze(labels, axis=-1) if labels is not None else inputs['point_labels']  # [BxN]

        neighbor_idx = inputs['neighbors'][stage_i]
        neighbor_idx = neighbor_idx[..., 1:]  # exclude self-loop

        sample_idx = []
        sample_mask = []  # pos=True, neg=False
        valid_mask = None
        anchor_mask = None  # mask on sample_mask, to select anchor point
        if 'label' in sample and (config.search == 'radius' or config.ignored_labels):
            valid_mask = []  # need to collect valid_mask
        for s in sample.split('-'):

            # Sampling
            if s.startswith('label'):  # using label - pos-neg
                sample_idx += [neighbor_idx]

            elif s.startswith('nn'):  # spatially closest as pos
                nn = neighbor_idx[..., :int(s[2:])]  # assume enough neighbor
                sample_idx += [nn]

            elif s.startswith('rand'):  # rand sampling as neg
                n_neg = int(re.search('\d+', s).group(0))
                if 'G' in s:  # cross-sample sampling (global)
                    BN = tf.shape(neighbor_idx)
                    if 'batches_len' in inputs:
                        rand_idx = tf.random.uniform(shape=[BN[0], n_neg], minval=0, maxval=BN[0], dtype=neighbor_idx.dtype)
                    else:
                        rand_idx = tf.random.uniform(shape=[BN[0], BN[1], n_neg], minval=0, maxval=BN[0] * BN[1], dtype=neighbor_idx.dtype)  # [B, N] - indexing into BxN
                        raise
                else:  # per-sample sampling
                    if 'batches_len' in inputs:
                        BN = inputs['batches_len'][stage_i]  # B - [B] indicating point num in each example
                        rand_idx_0 = tf.zeros([0, n_neg], dtype=neighbor_idx.dtype)
                        B = tf.shape(BN)[0]

                        def body(batch_i, rand_idx):
                            N = BN[batch_i]
                            cur_idx = tf.random.uniform(shape=[N, n_neg], minval=0, maxval=N, dtype=neighbor_idx.dtype)
                            rand_idx = tf.concat([rand_idx, cur_idx + tf.reduce_sum(BN[:batch_i])], axis=0)
                            batch_i += 1
                            return batch_i, rand_idx

                        def cond(batch_i, rand_idx): return tf.less(batch_i, B)

                        _, rand_idx = tf.while_loop(cond=cond, body=body, loop_vars=[0, rand_idx_0],
                                                    shape_invariants=[tf.TensorShape([]), tf.TensorShape([None, n_neg])], name='rand')
                    else:
                        BN = tf.shape(neighbor_idx)  # [B, N, n]
                        rand_idx = tf.random.uniform(shape=[BN[0], BN[1], n_neg], minval=0, maxval=BN[1], dtype=neighbor_idx.dtype)
                sample_idx += [rand_idx]  # [BxN, n]

            else:
                raise NotImplementedError(f'not supported sample = {s} in {sample}')

            mask = None
            cur_idx = sample_idx[-1]

            # sample_mask - pos-neg mask for sampled points
            if 'label' in s:
                posneg, mask = contrast_head.collect_labels(inputs, stage_n, stage_i, ftype, config, cur_idx, s)
                sample_mask += [posneg]  # NOTE: maybe of float if using labelkl
            elif s.startswith('nn'):
                sample_mask += [tf.ones(shape=tf.shape(cur_idx), dtype=tf.bool)]
            elif s.startswith('rand'):
                sample_mask += [tf.zeros(shape=tf.shape(cur_idx), dtype=tf.bool)]

            # anchor_mask - anchor selection from sample_mask
            if 'M' in s:  # if masked by existing mask (i.e. label) - exclude from anchor selection
                assert len(sample_mask) > 1
                anchor_mask = [] if anchor_mask is None else anchor_mask
                anchor_mask += [len(sample_mask) - 1]  # exclude current sample in selecting anchor

            # valid_mask - only valid points for contrast
            if 'label' in s and mask is not None:
                valid_mask += [mask]
            elif 'R' in s:  # reject neighbors
                valid_mask = [] if valid_mask is None else valid_mask
                mask = tf.not_equal(tf.expand_dims(cur_idx, axis=-1), tf.expand_dims(neighbor_idx, axis=-2))  # [BxN, n, k] = ([BxN, n, 1] != [BxN, 1, k])
                mask = tf.reduce_all(mask, axis=-1)  # [BxN, n] - sampled idx should != any of the neighbor
                valid_mask += [mask]
            elif valid_mask is not None:
                valid_mask += [tf.ones(shape=tf.shape(cur_idx), dtype=tf.bool)]


        if anchor_mask is not None:
            anchor_mask = [(tf.zeros if i in anchor_mask else tf.ones)(shape=tf.shape(m)[-1], dtype=tf.bool) for i, m in enumerate(sample_mask)]
            anchor_mask = tf.squeeze(tf.where(tf.concat(anchor_mask, axis=0)), axis=-1)  # per-point col idx on sample mask for anchor selection
        if valid_mask is not None:
            valid_mask = tf.concat(valid_mask, axis=-1) if len(valid_mask) > 1 else valid_mask[0]

        sample_idx = tf.concat(sample_idx, axis=-1) if len(sample_idx) > 1 else sample_idx[0]
        sample_mask = tf.concat(sample_mask, axis=-1) if len(sample_mask) > 1 else sample_mask[0]

        _glb[key] = (sample_idx, sample_mask, anchor_mask, valid_mask)
        return sample_idx, sample_mask, anchor_mask, valid_mask

    @staticmethod
    @tf_scope
    def solve_samples_mask(contrast, sample_mask, anchor_mask, valid_mask, config):
        # get pos-neg mask & point mask - points with desired pos/neg pairs
        pos_mask = neg_mask = pos_point = neg_point = None
        if contrast in ['triplet', 'softnn', 'simdiff', 'nce', 'bce', 'sim']:
            pos_mask = tf.logical_and(sample_mask, valid_mask) if valid_mask is not None else sample_mask  # [BxN, k] - valid eq in samples/neighbors
            pos_point = tf.gather(pos_mask, anchor_mask, axis=-1) if anchor_mask is not None else pos_mask
            pos_point = tf.reduce_any(pos_point, axis=-1, keepdims=True)  # [BxN, 1]
        if contrast in ['triplet', 'softnn', 'simdiff', 'nce', 'bce', 'diff']:
            neg_mask = tf.logical_and(tf.logical_not(sample_mask), valid_mask) if valid_mask is not None else tf.logical_not(sample_mask)  # valid neq
            neg_point = tf.gather(neg_mask, anchor_mask, axis=-1) if anchor_mask is not None else neg_mask
            neg_point = tf.reduce_any(neg_point, axis=-1, keepdims=True)
        if contrast in ['simplain']:
            pos_mask = tf.logical_or(sample_mask, tf.logical_not(valid_mask)) if valid_mask is not None else sample_mask  # [BxN, k] - valid -> eq
            pos_point = tf.gather(pos_mask, anchor_mask, axis=-1) if anchor_mask is not None else pos_mask
            pos_point = tf.logical_and(tf.reduce_all(pos_point, axis=-1, keepdims=True), tf.reduce_any(valid_mask, axis=-1, keepdims=True))  # [BxN, 1]
        if contrast in ['kl', 'ncekl', 'normkl', 'sigkl', 'msekl']:  # sample_mask - float
            pos_mask = sample_mask  # label kl dist
            neg_mask = tf.cast(tf.logical_not(valid_mask), pos_mask.dtype) if valid_mask is not None else None  # per-position IN-valid mask
            if contrast in ['sigkl', 'msekl']:
                pos_point = tf.reduce_any(valid_mask, axis=-1, keepdims=True)  # as regression
            else:
                pos_point = tf.cast(valid_mask, tf.float32, axis=-1, keepdims=True) > 1  # need >1 for normalization

        point_mask = [m for m in [pos_point, neg_point] if m is not None]
        if len(point_mask) == 1:
            point_mask = point_mask[0]
        else:
            point_mask = tf.logical_and(*point_mask)
            pos_mask = tf.logical_and(pos_mask, point_mask)
            neg_mask = tf.logical_and(neg_mask, point_mask)
        point_mask = tf.squeeze(point_mask, axis=-1)  # [BxN]
        if config.debug:
            print('pos/neg mask = ', pos_mask, neg_mask)
            print('point_mask = ', point_mask)
            # if anchor_mask is not None:
            #     point_mask = tf_Print(point_mask, [' anchor mask ', anchor_mask, ' sample_mask shape ', tf.shape(sample_mask)])
            #     point_mask = tf_Print(point_mask, [' select ', tf.reduce_sum(tf.cast(point_mask, tf.int32), axis=-1), 'from features ', tf.shape(features)])
        return pos_mask, neg_mask, point_mask

    @staticmethod
    @tf_scope
    def contrast(features, samples, contrast, inputs, head_cfg, config):

        sample_idx, sample_mask, anchor_mask, valid_mask, stage_n, stage_i = samples
        pos_mask, neg_mask, point_mask = contrast_head.solve_samples_mask(contrast, sample_mask, anchor_mask, valid_mask, config)

        def false_fn():
            if contrast in ['sim', 'diff', 'simdiff', 'simplain'] or len(head_cfg.dist.split('-')) == 1:
                # if sim/diff or not using reduce - dist = [BxN, k]
                mask = neg_mask if contrast in ['diff'] else pos_mask
                dist = tf.zeros(tf.shape(tf.boolean_mask(mask, point_mask)))
            else:
                # else, i.e. triplet/softnn and using reduce - dist = [pos=[BxN], neg]
                shape = tf.shape(point_mask)
                dist = [tf.zeros(shape=shape), tf.zeros(shape=shape)]
            return dist, 0.0

        def true_fn():
            d, l = contrast_head.calc_loss_from_sample(features, sample_idx, pos_mask, neg_mask, point_mask, contrast, inputs, (stage_n, stage_i), head_cfg, config)
            l = contrast_head.apply_weights(l, point_mask, inputs, stage_n, stage_i, head_cfg, config, name='weights')
            return d, l

        # NOTE: point_mask could be None => exclude unnecessary boolean_mask if all True
        dist, loss = tf.cond(tf.reduce_any(point_mask), true_fn=true_fn, false_fn=false_fn) if 'label' in head_cfg.sample else true_fn()
        return {'logits': dist, 'probs': dist, 'loss': loss, 'labels': tf.constant(0.0)}

    @staticmethod
    def calc_loss_from_sample(features, sample_idx, pos_mask, neg_mask, point_mask, contrast, inputs, stage, head_cfg, config):
        rst = contrast_head.calc_dist_from_sample(features, sample_idx, pos_mask, neg_mask, point_mask, head_cfg, config, name='dist')
        dist, pos_mask, neg_mask = rst  # dist & pos/neg mask selected by point mask
        loss = contrast_head.calc_loss_from_dist(dist, features, (pos_mask, neg_mask, point_mask), contrast, inputs, stage, head_cfg, config, name='loss')
        return dist, loss

    @staticmethod
    @tf_scope
    def calc_dist_from_sample(features, sample_idx, pos_mask, neg_mask, point_mask, head_cfg, config):

        key_dist = ['cos', 'l2', 'norml2', 'l1', 'norml1', 'dot', 'normdot', 'l2square']
        def calc_dist_sample(features, dist, sample_features=None):
            if 'norm' in dist or dist in ['cos']:
                features = tf.nn.l2_normalize(features, axis=-1, epsilon=_eps)
                if isinstance(sample_features, list) :  # list of [BxN, d] - reduced features
                    sample_features = [tf.nn.l2_normalize(f, axis=-1, epsilon=_eps) for f in sample_features if f is not None]

            if sample_features is None:  # [BxN, k, d] - gather
                sample_features = tf_gather(features, sample_idx, get_mask=False)
                features = tf.expand_dims(features, axis=-2)
                features = tf.boolean_mask(features, point_mask, name='mask_f')  # calc only the valid row
                sample_features = tf.boolean_mask(sample_features, point_mask, name='mask_neighbor')
                dist = calc_dist(features, sample_features, dist)
            else:
                assert isinstance(sample_features, list)
                features = tf.boolean_mask(features, point_mask, name='mask_f')  # calc only the valid row
                dist = [calc_dist(features, tf.boolean_mask(f, point_mask, name='mask_sample'), dist) for f in sample_features]
            # remove invalid row in mask
            nonlocal pos_mask, neg_mask  # bind to the enclosing scope
            pos_mask = tf.boolean_mask(pos_mask, point_mask, name='mask_pos') if pos_mask is not None else None
            neg_mask = tf.boolean_mask(neg_mask, point_mask, name='mask_neg') if neg_mask is not None else None
            return dist  # [BxN, k, 1] / [[BxN, 1], ...]

        key_reduction = ['minmax', 'mean']
        def calc_reduction(features, reduction):
            # [BxN, k, d/1] - reduce features [BxN, d] / dist [BxN, k, 1]
            sample_features = tf_gather(features, sample_idx, get_mask=False) if len(features.shape) <= len(sample_idx.shape) else features

            pos_msk = tf.cast(tf.expand_dims(pos_mask, axis=-1), tf.float32) if pos_mask is not None else None  # [BxN, k, 1]
            neg_msk = tf.cast(tf.expand_dims(neg_mask, axis=-1), tf.float32) if neg_mask is not None else None

            pos = neg = None
            if reduction == 'minmax':  # max pos & min neg - hardest pair
                assert int(features.shape[-1]) == 1, f'min-max should be applied on distance, not features'
                if pos_msk is not None:
                    pos = tf.reduce_max(sample_features * pos_msk + (tf.reduce_min(sample_features, axis=-2, keepdims=True) * (1 - pos_msk)), axis=-2)
                if neg_msk is not None:
                    neg = tf.reduce_min(sample_features * neg_msk + (tf.reduce_max(sample_features, axis=-2, keepdims=True) * (1 - neg_msk)), axis=-2)
            elif reduction == 'mean':
                pos = tf.reduce_sum(sample_features * pos_msk, axis=-2) / tf.reduce_sum(pos_msk, axis=-2) if pos_msk is not None else None
                neg = tf.reduce_sum(sample_features * neg_msk, axis=-2) / tf.reduce_sum(neg_msk, axis=-2) if neg_msk is not None else None
            else:
                raise NotImplementedError(f'not supported reduction = {reduction}')
            return [pos, neg]  # [BxN, 1/d]

        # calc dist - dist & reduce
        #   dist -> reduce: [BxN, d] -> [BxN, k, 1] -> [BxN, 1]...
        #   reduce -> dist: [BxN, d] -> [BxN, d]... -> [BxN, 1]...
        #   dist: [BxN, d] -> [BxN, k, 1]
        sample_features = None
        for ops in head_cfg.dist.split('-'):
            if ops in key_dist:
                # calculate distance: [BxN, d] -> [BxN, k, 1], or, list of [BxN, d] -> list of [BxN, 1]
                dist = calc_dist_sample(features, ops, sample_features=sample_features)
                features = dist
            elif ops in key_reduction:
                # reduce by mask: [BxN, k, 1/d] -> list of [BxN, 1/d]...
                dist = sample_features = calc_reduction(features, ops)
            else:
                raise NotImplementedError(f'not supported ops = {ops} in head_cfg.dist = {head_cfg.dist}')
        # [BxN, k] / [pos=[BxN], neg]
        dist = [tf.squeeze(d, axis=-1) for d in dist] if isinstance(dist, list) else tf.squeeze(dist, axis=-1)
        return dist, pos_mask, neg_mask

    @staticmethod
    @tf_scope
    def calc_loss_from_dist(dist, features, masks, contrast, inputs, stage, head_cfg, config):
        pos_mask, neg_mask, point_mask = masks

        # solve margin - mask - other shared aug
        margin = head_cfg.margin if head_cfg.margin else None  # e.g. m.1, m.1d, mI
        if margin and margin[-1] == 'd':
            margin = tf.reduce_mean(dist) * float(margin[:-1])
        elif margin and not re.fullmatch('[IELSP]*(T([pvl]|)\.{0,1}\d*|)', margin):
            margin = float(margin)
        masking = head_cfg.mask if head_cfg.mask else None  # e.g. m.1-mask, mask.1, mI-mask
        if masking and not masking[4:] and margin is not None:  # tf/str
            masking = margin
        elif masking and masking[4:]:  # float
            masking = float(masking[4:])
        contrast_aug = head_cfg.contrast_aug  # e.g. p2
        if config.debug:
            print('margin = ', margin); print('masking = ', masking)

        temperature = None
        if margin is not None and 'T' in margin:  # temperature
            temperature = margin[margin.index('T') + 1:]
            if temperature == 'v':
                temperature = tf_get_variable('temprature', [], initializer='ones')
                # temperature = tf_get_variable('temperature', [], initializer='ones')
            elif temperature.startswith('p'):
                f = tf.boolean_mask(features, point_mask)
                f = tf.nn.sigmoid(dense_layer(f, 1, f'linear', None, True, False, initializer='fanin'))
                temperature = f if not temperature[1:] else f * float(temperature[1:])
            elif temperature.startswith('l'):
                temperature = float(temperature[1:] if temperature[1:] else 1) * (stage[1] + 1)
            else:
                temperature =  float(temperature)

        def apply_masking(dist, valid_mask):
            dist = tf.cond(tf.reduce_any(valid_mask), 
                            true_fn=lambda: tf.boolean_mask(dist, valid_mask), false_fn=lambda: 0.0, name='masking')
            return dist

        # contrast on pos-neg dist - [BxN, k], or, [pos=[BxN], neg=[BxN]]
        if contrast == 'softnn':
            assert masking is None, f'pos-neg mask already masked by point mask'
            if isinstance(dist, list):
                dist = tf.stack(dist, axis=-1)  # [BxN, 2]

            dist = -dist  # take negative
            if temperature is not None:
                dist /= temperature
            dist = dist - tf.reduce_max(dist, axis=-1, keepdims=True)  # numerical stability
            exps = tf.exp(dist)
            pos = tf.reduce_sum(exps * tf.cast(pos_mask, dist.dtype), axis=-1)  # [BxN] - sum [pos exps] / max exps
            neg = tf.reduce_sum(exps * tf.cast(neg_mask, dist.dtype), axis=-1)  # [BxN] - sum [neg exps] / max exps

            if margin is not None and 'S' in margin:  # pos / neg - separate
                dist = pos / tf.maximum(neg, _eps)
            else:  # pos / (pos + neg)
                dist = pos / (pos + neg)

            # maximize pos <=> minimize neg
            # NOTE - pos/neg = exp[-dist]
            if margin is not None and 'I' in margin:  # inverse
                dist = tf.boolean_mask(dist, tf.greater(dist, 0.0)) if masking else dist + _eps  # avoid inf
                dist = 1 / dist
            elif margin is not None and 'P' in margin:  # as probability
                dist = 1 - dist
            else:
                dist = tf.boolean_mask(dist, tf.greater(dist, 0.0)) if masking else dist + _eps
                dist = -tf.log(dist)

        elif contrast == 'nce':
            if isinstance(dist, list):
                dist = tf.stack(dist, axis=-1)  # [BxN, 2]

            dist = -dist  # take negative (to align with the minimizing of pos pair dist - as xen maximize the dist at pos_mask)
            if temperature is not None:
                dist /= temperature
            dist = dist - tf.reduce_max(dist, axis=-1, keepdims=True)  # numerical stability
            exps = tf.exp(dist)

            # contrasting against (per-pos + neg) / (all pos + neg)
            if margin is not None and 'S' in margin:  # separate pos term
                under = exps * tf.cast(neg_mask, dist.dtype)
                under = exps + tf.reduce_sum(under, axis=-1, keepdims=True)  # [BxN, k]
            else:
                under = tf.reduce_sum(exps * tf.cast(tf.logical_or(pos_mask, neg_mask), dist.dtype), axis=-1, keepdims=True)  # [BxN, 1]

            # maximize pos <=> minimize neg
            dist = exps / under  # [BxN, k]
            if masking:
                dist = -tf.log(tf.boolean_mask(dist, pos_mask) + _eps)  # [num of all pos term] - each log term an example for CL
            else:
                dist = -tf.reduce_sum(tf.log(dist + _eps) * tf.cast(pos_mask, dist.dtype), axis=-1)  # [BxN]

        else:
            raise NotImplementedError(f'not supported contrast = {contrast}')

        for aug in contrast_aug.split('-'):
            if not aug: continue
            if aug[0] == 'p': dist = tf.pow(dist, float(aug[1:]))  # power - >1 to enhance larger loss, <1 for more close-to-margin loss
            else: raise NotImplementedError(f'not supported post-ops = {aug} in contrast_aug = {contrast_aug}')

        loss = dist  # [BxN]
        return loss

    @staticmethod
    @tf_scope
    def apply_weights(loss, point_mask, inputs, stage_n, stage_i, head_cfg, config):
        # assume loss - [BxN]
        weight = head_cfg.weight
        if isinstance(weight, str):
            raise NotImplementedError
        else:
            wfloat = weight
        loss = tf.reduce_mean(loss) * wfloat
        return loss



def get_head_ops(head_n, raise_not_found=True):

    # head_n == idx_name_pre
    if head_n == 'mlp':
        head_ops = mlp_head()
    elif head_n == 'multiscale':
        head_ops = multiscale_head()
    elif head_n == 'contrast':
        head_ops = contrast_head()

    # raise or skip
    elif raise_not_found:
        raise NotImplementedError(f'not supported head_n = {head_n}')
    else:
        head_ops = None
    return head_ops

def apply_head_ops(inputs, head_cfg, config, is_training):
    head_ops = get_head_ops(head_cfg.head_n)
    rst = head_ops(inputs, head_cfg, config, is_training)
    return rst
