import os, re, json
import numpy as np
import tensorflow as tf

def restore(session, restore_snap, except_list=None, select_list=None, restore_vars=None, raise_if_not_found=False, saver=None, verbose=True):
    """
    restore_vars: the dict for tf saver.restore => a dict {name in ckpt : var in current graph}
    """
    except_list = [] if except_list is None else except_list
    select_list = [] if select_list is None else select_list
    restore_vars = {} if restore_vars is None else restore_vars

    save_file = restore_snap
    if not os.path.exists(save_file) and raise_if_not_found:
        raise Exception('File %s not found' % save_file)
    # load stored model
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()  # dict {op name : shape in list}

    # matching vars in current graph to vars in file
    restored_var_names = set([v.name.split(':')[0] for v in restore_vars.keys()])
    # restored_var_new_shape = []
    if verbose:
        print('Restoring:')
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        for var in tf.global_variables():
            op_name = var.name.split(':')[0]

            # filter out wanted/unwanted ops
            if except_list and any([bool(re.fullmatch(expr, op_name)) for expr in except_list]):
                continue
            if select_list and not any([bool(re.fullmatch(expr, op_name)) for expr in select_list]):
                continue

            if op_name not in saved_shapes:  # not found in saved file
                continue
            v_shape = var.get_shape().as_list()  # checking shape
            if v_shape == saved_shapes[op_name]:
                restored_var_names.add(op_name)
                restore_vars[op_name] = var
            else:
                print('Shape mismatch for var', op_name, 'expected', v_shape, 'got', saved_shapes[op_name])
                # restored_var_new_shape.append((saved_var_name, cur_var, reader.get_tensor(saved_var_name)))
                # print('bad things')
    # print info
    if verbose:
        for k, v in restore_vars.items():
            v_name = v.name.split(':')[0]
            v_shape = v.get_shape().as_list()
            v_size = int(np.prod(v_shape) * 4 / 10 ** 6)
            print(f'\t{k} \t -> \t {v_name}, shape {v_shape} = {v_size}MB')

    ignored_var_names = sorted(list(set(saved_shapes.keys()) - restored_var_names))
    missing_var_names = sorted(list(set(restored_var_names - saved_shapes.keys())))
    if verbose:
        print('\n')
        if len(ignored_var_names):
            print('left-over ckpt variables:\n\t' + '\n\t'.join(ignored_var_names))
        else:
            print('All ckpt variables restored into graph')
        if len(missing_var_names):
            print('left-over graph variables:\n\t' + '\n\t'.join(missing_var_names))
        else:
            print('All graph variables restored from ckpt')

    if restore_vars and session != None:
        # NOTE: may need to check saver's vars v.s. restore_vars
        saver = saver if saver else tf.train.Saver(restore_vars)
        saver.restore(session, save_file)
    """
    if len(restored_var_new_shape) > 0:
        print('trying to restore misshapen variables')
        assign_ops = []
        for name, kk, vv in restored_var_new_shape:
            copy_sizes = np.minimum(kk.get_shape().as_list(), vv.shape)
            slices = [slice(0,cs) for cs in copy_sizes]
            print('copy shape', name, kk.get_shape().as_list(), '->', copy_sizes.tolist())
            new_arr = session.run(kk)
            new_arr[slices] = vv[slices]
            assign_ops.append(tf.assign(kk, new_arr))
        session.run(assign_ops)
        print('Copying unmatched weights done')
    """
    if verbose:
        print('Restored %s' % save_file)
    try:
        start_iter = int(save_file.split('-')[-1])  # get global_step
    except ValueError:
        print('Could not parse start iter, assuming 0')
        start_iter = 0
    return start_iter, ignored_var_names, missing_var_names


class TimeLiner(object):

    def __init__(self):
        super(TimeLiner, self).__init__()
        self._timeline_dict = None

    def update_timeline(self, chrome_trace):
        # convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)
        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict
        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)


def get_boundary_mask(labels, neighbor_label=None, neighbor_idx=None, valid_mask=None, iterative=False, use_np=False, **kwargs):
    """ wrapping & adding np implementation """
    assert neighbor_label is not None or neighbor_idx is not None
    if use_np:
        func = _get_boundary_mask_np_iter if iterative else _get_boundary_mask_np
    else:
        from models.basic_operators import get_boundary_mask as get_boundary_mask_tf
        func = lambda *args, **kwargs: get_boundary_mask_tf(*args, get_plain=True, **kwargs)
    return func(labels, neighbor_label, neighbor_idx, valid_mask, **kwargs)

def _get_boundary_mask_np_iter(labels, neighbor_label=None, neighbor_idx=None, valid_mask=None, batch=1e4, posneg=False):
    print('using iter', flush=True)
    bound = np.zeros(labels.shape, dtype=bool)
    plain = np.zeros(labels.shape, dtype=bool)
    if posneg:
        posneg = [[], []]

    if valid_mask is not None:
        assert valid_mask.shape == labels.shape, f'invalid shape - labels {labels.shape}, valid_mask {valid_mask.shape}'

    N = len(labels)
    if neighbor_label is None:
        if neighbor_idx.max() >= N:
            labels = np.concatenate([labels, [-1]], axis=0)  # -1 for invalid

    idx = 0
    batch = int(batch)
    while idx < N:  # per-entry (potentially batched)
        neighbor_label = np.vstack([labels[i] for i in neighbor_idx[idx:idx+batch]])
        valid_neighbor = neighbor_label >= 0

        cur_label = np.expand_dims(labels[idx:idx+batch], axis=-1)
        eq = cur_label == neighbor_label
        neq = cur_label != neighbor_label

        eq = np.logical_or(eq, np.logical_not(valid_neighbor))  # valid -> all equal => plain if all neighbors invalid
        neq = np.logical_and(neq, valid_neighbor)  # mask out invalid
        if posneg:
            posneg[0].append(eq)
            posneg[1].append(neq)

        eq = eq.all(axis=-1)
        neq = neq.any(axis=-1)

        if valid_mask is not None:  # [BxN] - assume label mask
            valid_mask_i = valid_mask[idx:idx+batch]
            eq = np.logical_and(eq, valid_mask_i)
            neq = np.logical_and(neq, valid_mask_i)

        bound[idx:idx+batch] = neq
        plain[idx:idx+batch] = eq
        idx += batch
    if posneg:
        posneg[0] = np.concatenate(posneg[0], axis=0)
        posneg[1] = np.concatenate(posneg[1], axis=0)
        return bound, plain, posneg
    return bound, plain

def _get_boundary_mask_np(labels, neighbor_label=None, neighbor_idx=None, valid_mask=None, posneg=False):
    if neighbor_label == None:
        if neighbor_idx.max() >= len(labels):
            labels = np.concatenate([labels, [-1]], axis=0)  # -1 for invalid
            neighbor_label = np.vstack([labels[idx] for idx in neighbor_idx])
            labels = labels[:-1]
        else:
            neighbor_label = np.vstack([labels[idx] for idx in neighbor_idx])  # [N, k] as using per-cloud label

    # valid_mask = labels >= 0
    valid_neighbor = neighbor_label >= 0
    labels = np.expand_dims(labels, axis=-1)  # [N, 1]

    eq = labels == neighbor_label
    neq = labels != neighbor_label

    eq = np.logical_or(eq, np.logical_not(valid_neighbor))  # valid -> all equal
    neq = np.logical_and(neq, valid_neighbor)  # mask out invalid

    bound = np.any(neq, axis=-1)
    plain = np.all(eq, axis=-1)

    if valid_mask is not None:
        bound = np.logical_and(bound, valid_mask)
        plain = np.logical_and(plain, valid_mask)

    if posneg:
        return bound, plain, (eq, neq)
    return bound, plain
