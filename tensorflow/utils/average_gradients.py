import tensorflow as tf


def average_gradients(tower_grads, grad_norm, raise_on_none=True, grad_reduce=None, device=None):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    From tensorflow tutorial: cifar10/cifar10_multi_gpu_train.py
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
            - [[(g,v), ... at gpu 0], ..., [(g,v), ... at gpu N]]
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    if device:
        with tf.device(device):
            return average_gradients(tower_grads, grad_norm, raise_on_none, grad_reduce, None)

    use_clip = grad_norm and grad_norm > 0
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars containes (grad, var) calculated at each gpu, looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, v in grad_and_vars:
            if g is not None:
                if use_clip:
                    g = tf.clip_by_norm(g, grad_norm)
            elif raise_on_none:
                raise ValueError(f'variable {v} got None gradients')
            else:
                continue
                # g = tf.zeros_like(v)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(g)

        # Average over the 'tower' dimension.
        if len(grads) > 1 and (grad_reduce == 'concat' or not grad_reduce):
            # Add 0 dimension to the gradients to represent the tower.
            grads = [tf.expand_dims(g, 0) for g in grads]
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)
        elif len(grads) > 1 and grad_reduce == 'mean':
            # Direct mean
            grad = tf.accumulate_n(grads) / len(grads)
        elif len(grads) == 1:
            # skip if only 1 gpu
            grad = grads[0]
        elif len(grads) == 0:
            grad = None
        else:
            raise ValueError(f'not support grad_reduce = {config.grad_reduce}')

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
