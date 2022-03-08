import os
import numpy as np
import tensorflow as tf
if tf.__version__.split('.')[0] == '2':
    tf = tf.compat.v1
    tf.disable_v2_behavior()
OPS_DIR = os.path.dirname(os.path.abspath(__file__))

from models.utils import tf_device

class TF_OPS(object):

    modules = {
        'grid_preprocess': None,

        'grid': None,
        'random': None,
        'farthest': None,
        'farthest_gpu': None,

        'knn': None,
        'radius': None,
    }

    @staticmethod
    def get_tf_func(key, verbose=True):

        modules = TF_OPS.modules
        assert key in modules, f'invalide key for tf ops: {key} not in {modules.keys()}'

        # init module
        if modules[key] is None:

            # preprocessing methdos
            if key == 'grid_preprocess':
                import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
                modules[key] = cpp_subsampling

            # sampling methods
            elif key == 'grid':
                modules[key] = tf.load_op_library(os.path.join(OPS_DIR, 'tf_custom_ops/tf_batch_subsampling.so'))
            elif key == 'random':
                pass
            elif key == 'farthest':
                from sampling import farthest_sample
                modules[key] = farthest_sample
            elif key == 'farthest_gpu':
                modules[key] = tf.load_op_library(os.path.join(OPS_DIR, 'sampling/tf_sampling_so.so'))

            # search methods
            elif key == 'knn':
                import nearest_neighbors.lib.python.nearest_neighbors as nearest_neighbors
                modules[key] = nearest_neighbors
            elif key == 'radius':
                modules[key] = tf.load_op_library(os.path.join(OPS_DIR, 'tf_custom_ops/tf_batch_neighbors.so'))
            else:
                raise NotImplementedError(f'not supported module init for key = {key}')

        # return func
        func = {
            'grid_preprocess': TF_OPS.grid_subsampling,

            'grid': TF_OPS.tf_batch_subsampling,
            'random': TF_OPS.tf_random_sampling_fixed_size,
            'farthest': TF_OPS.tf_farthest_point_sampling,
            'farthest_gpu': TF_OPS.tf_farthest_point_sampling_gpu,

            'knn': TF_OPS.tf_knn_search,
            'radius': TF_OPS.tf_batch_neighbors,
        }[key]
        if verbose:
            print(f'loading tf func [{key}] = {func}')
        return func


    # ---------------------------------------------------------------------------- #
    # ops for dataset pre-processing
    # ---------------------------------------------------------------------------- #


    @staticmethod
    def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
        """CPP wrapper for a grid subsampling (method = barycenter for points and features
        Args:
            points: (N, 3) matrix of input points
            features: optional (N, d) matrix of features (floating number)
            labels: optional (N,) matrix of integer labels
            sampleDl: parameter defining the size of grid voxels
            verbose: 1 to display

        Returns:
            subsampled points, with features and/or labels depending of the input
        """
        cpp_subsampling = TF_OPS.modules['grid_preprocess']
        if (features is None) and (labels is None):
            return cpp_subsampling.compute(points, sampleDl=sampleDl, verbose=verbose)
        elif (labels is None):
            return cpp_subsampling.compute(points, features=features, sampleDl=sampleDl, verbose=verbose)
        elif (features is None):
            return cpp_subsampling.compute(points, classes=labels, sampleDl=sampleDl, verbose=verbose)
        else:
            return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=sampleDl, verbose=verbose)


    # ---------------------------------------------------------------------------- #
    # ops for model down-sampling
    # ---------------------------------------------------------------------------- #


    @staticmethod
    def _knn_search(query_pts, support_pts, k):
        neighbor_idx = TF_OPS.modules['knn'].knn_batch(support_pts, query_pts, k, omp=True)
        return neighbor_idx.astype(np.int32)

    @staticmethod
    @tf_device
    def tf_knn_search(query_pts, support_pts, k, device=None):
        """
        Args:
            query_pts   : [B, M, 3] - query points
            support_pts : [B, N, 3] - the point cloud
        Returns:
            neighbor_idx    : [B, M, k] - idx of k-neighbors of each query point
        """
        py_function = tf.py_function if hasattr(tf, 'py_function') else tf.py_func
        neighbor_idx = py_function(TF_OPS._knn_search, [query_pts, support_pts, k], tf.int32)  # [B, M, k]
        neighbor_idx = tf.reshape(neighbor_idx, tf.concat([tf.shape(query_pts)[:-1], [k]], axis=0))  # explicitly set shape
        neighbor_idx = tf.stop_gradient(neighbor_idx)
        return neighbor_idx


    @staticmethod
    def tf_farthest_point_sampling_gpu(points, sample_num):
        """ farthest sampling (cuda ops) on [B, N, 3]
        """
        ret = TF_OPS.modules['farthest_gpu'].farthest_point_sample(points, sample_num)
        ret = tf.stop_gradient(ret)
        return ret

    @staticmethod
    def tf_farthest_point_sampling(points, sample_num):
        """ farthest sampling (tf while ops) on [B, N, 3]
        """
        ret = TF_OPS.modules['farthest'].farthest_sample(points, sample_num)
        ret = tf.stop_gradient(ret)
        return ret

    @staticmethod
    def tf_random_sampling_fixed_size(points, sample_num, shuffle=False):
        """ random sampling on [B, N, 3]
        """
        if shuffle:
            points = tf.random.shuffle(points)
        return points[:, :sample_num]


    @staticmethod
    def tf_batch_subsampling(points, batches_len, sampleDl):
        """ grid subsampling for [BxN, ...]
        """
        return TF_OPS.modules['grid'].batch_grid_subsampling(points, batches_len, sampleDl)

    @staticmethod
    @tf_device
    def tf_batch_neighbors(queries, supports, q_batches, s_batches, radius):
        """ radius neighbors for [BxN, ...]
        """
        return TF_OPS.modules['radius'].batch_ordered_neighbors(queries, supports, q_batches, s_batches, radius)
