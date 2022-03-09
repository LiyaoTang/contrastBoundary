#
#
#      0=========================0
#      |    Kernel Point CNN     |
#      0=========================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Handle NPM3D dataset in a class
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Basic libs
import os, sys, time, pickle
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KDTree

# PLY reader
from utils.ply import read_ply, write_ply
from utils.mesh import rasterize_mesh

# OS functions
from os import makedirs, listdir
from os.path import exists, join, isfile, isdir

# Dataset parent class
from .base import Dataset

# Subsampling extension
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#

def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.compute(points, sampleDl=sampleDl, verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.compute(points, features=features, sampleDl=sampleDl, verbose=verbose)
    elif (features is None):
        return cpp_subsampling.compute(points, classes=labels, sampleDl=sampleDl, verbose=verbose)
    else:
        return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=sampleDl, verbose=verbose)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Class Definition
#       \***************/
#


class NPM3DDataset(Dataset):
    """
    Class to handle S3DIS dataset for segmentation task.
    """
    label_to_names = {0: 'unclassified',
                      1: 'ground',
                      2: 'buildings',
                      3: 'poles',
                      4: 'bollards',
                      5: 'trash_cans',
                      6: 'barriers',
                      7: 'pedestrians',
                      8: 'cars',
                      9: 'natural'}
    ignored_labels = np.sort([0])

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, config, verbose=True, input_threads=8):
        super(NPM3DDataset, self).__init__(config)
        self.verbose = verbose

        # Initiate a bunch of variables concerning class labels
        self.init_labels()
        self.num_threads = max([input_threads, config.gpu_num, config.num_threads])

        # Load test set or train set?
        self.load_test = 'test' in config.mode

        # Path of the folder containing ply files
        self.path = self.data_path
        # Path of the training files
        self.train_path = os.path.join(self.path, 'training_10_classes')
        self.test_path = os.path.join(self.path, 'test_10_classes')
        # List of training and test files
        self.train_files = np.sort([os.path.join(self.train_path, f) for f in os.listdir(self.train_path) if f[-4:] == '.ply'])
        self.test_files = np.sort([os.path.join(self.test_path, f) for f in os.listdir(self.test_path) if f[-4:] == '.ply'])

        # Proportion of validation scenes
        self.all_splits = [0, 1, 2, 3]
        self.validation_split = -1

        self.load_subsampled_clouds(config.first_subsampling_dl, verbose=verbose)

    def load_subsampled_clouds(self, subsampling_parameter, verbose=True):
        """
        Presubsample point clouds and load into memory (Load KDTree for neighbors searches
        """

        if 0 < subsampling_parameter <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')

        from ops import get_tf_func
        grid_subsampling = get_tf_func('grid_preprocess')

        # Create path for files
        tree_path = os.path.join(self.path, 'input_{:.3f}'.format(subsampling_parameter))
        if not os.path.exists(tree_path):
            os.makedirs(tree_path)

        # All training and test files
        files = np.hstack((self.train_files, self.test_files))

        # Initiate containers
        self.input_files = {'training': [], 'validation': [], 'test': []}
        self.input_trees = {'training': [], 'validation': [], 'test': []}
        self.input_colors = {'training': [], 'validation': [], 'test': []}  # actually, reflectance
        self.input_labels = {'training': [], 'validation': []}

        # Advanced display
        N = len(files)
        progress_n = 30
        fmt_str = '[{:<' + str(progress_n) + '}] {:5.1f}%'

        if self.verbose:
            print('\nPreparing KDTree for all scenes, subsampled at {:.3f}'.format(subsampling_parameter))

        for i, file_path in enumerate(files):

            # Restart timer
            t0 = time.time()

            # get cloud name and split
            cloud_name = file_path.split('/')[-1][:-4]
            cloud_folder = file_path.split('/')[-2]
            if 'train' in cloud_folder:
                if self.all_splits[i] == self.validation_split:
                    cloud_split = 'validation'
                else:
                    cloud_split = 'training'
            else:
                cloud_split = 'test'

            # if (cloud_split != 'test' and self.load_test) or (cloud_split == 'test' and not self.load_test):
            if cloud_split == 'test' and not self.load_test:
                continue

            # Name of the input files
            KDTree_file = os.path.join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = os.path.join(tree_path, '{:s}.ply'.format(cloud_name))

            # Check if inputs have already been computed
            if isfile(KDTree_file):

                # read ply with data
                data = read_ply(sub_ply_file)
                sub_reflectance = np.expand_dims(data['reflectance'], 1)
                if cloud_split == 'test':
                    sub_labels = None
                else:
                    sub_labels = data['class']

                # Read pkl with search tree
                with open(KDTree_file, 'rb') as f:
                    search_tree = pickle.load(f)

            else:

                # Read ply file
                data = read_ply(file_path)
                points = np.vstack((data['x'], data['y'], data['z'])).astype(np.float32).T
                reflectance = np.expand_dims(data['reflectance'], 1).astype(np.float32)
                if cloud_split == 'test':
                    int_features = None
                else:
                    int_features = data['class']

                # Saturate reflectance
                reflectance = np.minimum(reflectance, 50.0)

                # Subsample cloud
                sub_data = grid_subsampling(points,
                                            features=reflectance,
                                            labels=int_features,
                                            sampleDl=subsampling_parameter)

                # Rescale and saturate float reflectance
                sub_reflectance = sub_data[1] / 50.0

                # Get chosen neighborhoods
                search_tree = KDTree(sub_data[0], leaf_size=50)

                # Save KDTree
                with open(KDTree_file, 'wb') as f:
                    pickle.dump(search_tree, f)

                # Save ply
                if cloud_split == 'test':
                    sub_labels = None
                    write_ply(sub_ply_file,
                              [sub_data[0], sub_reflectance],
                              ['x', 'y', 'z', 'reflectance'])
                else:
                    sub_labels = np.squeeze(sub_data[2])
                    write_ply(sub_ply_file,
                              [sub_data[0], sub_reflectance, sub_labels],
                              ['x', 'y', 'z', 'reflectance', 'class'])

            # Fill data containers
            self.input_files[cloud_split] += [file_path]
            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_reflectance]
            if cloud_split in ['training', 'validation']:
                self.input_labels[cloud_split] += [sub_labels]

            if verbose and sys.stdout.isatty():
                print('', end='\r')
                print(fmt_str.format('#' * (((i+1) * progress_n) // N), 100 * (i+1) / N), end='', flush=True)

        # joint training on train-val for submission
        if self.validation_split not in self.all_splits:
            self.input_files['validation'] = self.input_files['training']
            self.input_trees['validation'] = self.input_trees['training']
            self.input_colors['validation'] = self.input_colors['training']
            self.input_labels['validation'] = self.input_labels['training']

        # Get number of clouds
        self.num_training = len(self.input_trees['training'])
        self.num_validation = len(self.input_trees['validation'])
        self.num_test = len(self.input_trees['test'])

        # Get validation and test reprojection indices
        self.validation_proj = []
        self.validation_labels = []
        self.test_proj = []
        self.test_labels = []
        i_val = 0
        i_test = 0

        # Advanced display
        N = max(self.num_validation + self.num_test, 1)
        if verbose and sys.stdout.isatty():
            print('', end='\r')
            print(fmt_str.format('#' * progress_n, 100), flush=True)
        if verbose:
            print('\nPreparing reprojection indices for validation and test')

        for i, file_path in enumerate(files):

            # get cloud name and split
            cloud_name = file_path.split('/')[-1][:-4]
            cloud_folder = file_path.split('/')[-2]

            # Validation projection and labels
            if 'train' in cloud_folder and file_path in self.input_files['validation']:
                proj_file = os.path.join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                if isfile(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds, labels = pickle.load(f)
                else:

                    # Get original points
                    data = read_ply(file_path)
                    points = np.vstack((data['x'], data['y'], data['z'])).T
                    labels = data['class']

                    # Compute projection inds
                    proj_inds = np.squeeze(self.input_trees['validation'][i_val].query(points, return_distance=False))
                    proj_inds = proj_inds.astype(np.int32)

                    # Save
                    with open(proj_file, 'wb') as f:
                        pickle.dump([proj_inds, labels], f)

                self.validation_proj += [proj_inds]
                self.validation_labels += [labels]
                i_val += 1

            # Test projection
            if self.load_test and 'test' in cloud_folder:
                proj_file = os.path.join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                if isfile(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds = pickle.load(f)
                else:

                    # Get original points
                    data = read_ply(file_path)
                    points = np.vstack((data['x'], data['y'], data['z'])).T

                    # Compute projection inds
                    proj_inds = np.squeeze(self.input_trees['test'][i_test].query(points, return_distance=False))
                    proj_inds = proj_inds.astype(np.int32)

                    # Save
                    with open(proj_file, 'wb') as f:
                        pickle.dump(proj_inds, f)

                self.test_proj += [proj_inds]
                self.test_labels += [np.zeros(0, dtype=np.int32)]
                i_test += 1

            if verbose and sys.stdout.isatty():
                print('', end='\r')
                print(fmt_str.format('#' * (((i_val + i_test) * progress_n) // N), 100 * (i_val + i_test) / N), end='', flush=True)
        if verbose:
            print('\n')

        # class cnt over all clouds in val set (w/o invalid class)
        self.val_proportions_full = np.array([np.sum([np.sum(labels == label_val) for labels in self.validation_labels]) for label_val in self.label_values])
        self.val_proportions = np.array([p for l, p in zip(self.label_values, self.val_proportions_full) if l not in self.ignored_labels])
        return

    # Utility methods
    # ------------------------------------------------------------------------------------------------------------------
    def get_batch_gen_radius(self, split):
        """
        A function defining the batch generator for each split. Should return the generator, the generated types and
        generated shapes
        :param split: string in "training", "validation" or "test"
        :param config: configuration file
        :return: gen_func, gen_types, gen_shapes
        """

        ############
        # Parameters
        ############

        config = self.config
        epoch_n = self.init_sampling(split)
        data_split = split

        ##########################
        # Def generators functions
        ##########################
        def spatially_regular_gen():

            # Initiate concatanation lists
            p_list = []
            c_list = []
            pl_list = []
            pi_list = []
            ci_list = []

            batch_n = 0

            # Generator loop
            for i in range(epoch_n):

                # Choose a random cloud
                cloud_ind = int(np.argmin(self.min_potentials[split]))

                # Choose point ind as minimum of potentials
                point_ind = np.argmin(self.potentials[split][cloud_ind])

                # Get points from tree structure
                points = np.array(self.input_trees[data_split][cloud_ind].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                if split != 'ERF':
                    noise = np.random.normal(scale=config.in_radius/10, size=center_point.shape)
                    pick_point = center_point + noise.astype(center_point.dtype)
                else:
                    pick_point = center_point

                # Indices of points in input region
                input_inds = self.input_trees[data_split][cloud_ind].query_radius(pick_point,
                                                                             r=config.in_radius)[0]

                # Number collected
                n = input_inds.shape[0]

                # Update potentials (Tuckey weights)
                if split != 'ERF':
                    dists = np.sum(np.square((points[input_inds] - pick_point).astype(np.float32)), axis=1)
                    tukeys = np.square(1 - dists / np.square(config.in_radius))
                    tukeys[dists > np.square(config.in_radius)] = 0
                    self.potentials[split][cloud_ind][input_inds] += tukeys
                    self.min_potentials[split][cloud_ind] = float(np.min(self.potentials[split][cloud_ind]))

                    # Safe check for very dense areas
                    if n > self.batch_limit:
                        input_inds = np.random.choice(input_inds, size=int(self.batch_limit)-1, replace=False)
                        n = input_inds.shape[0]

                # Collect points and colors
                input_points = (points[input_inds] - pick_point).astype(np.float32)
                input_colors = self.input_colors[data_split][cloud_ind][input_inds]  # actually, reflectance
                if split in ['test', 'ERF']:
                    input_labels = np.zeros(input_points.shape[0])
                else:
                    input_labels = self.input_labels[data_split][cloud_ind][input_inds]
                    input_labels = np.array([self.label_to_idx[l] for l in input_labels])

                # In case batch is full, yield it and reset it
                if batch_n + n > self.batch_limit and batch_n > 0:
                    yield (np.concatenate(p_list, axis=0),
                           np.concatenate(c_list, axis=0),
                           np.concatenate(pl_list, axis=0),
                           np.array([tp.shape[0] for tp in p_list]),
                           np.concatenate(pi_list, axis=0),
                           np.array(ci_list, dtype=np.int32))

                    p_list = []
                    c_list = []
                    pl_list = []
                    pi_list = []
                    ci_list = []
                    batch_n = 0

                # Add data to current batch
                if n > 0:
                    p_list += [input_points]
                    c_list += [np.hstack((input_colors, input_points + pick_point))]  # [BxN, 4 = reflectance-XYZ]
                    pl_list += [input_labels]
                    pi_list += [input_inds]
                    ci_list += [cloud_ind]

                # Update batch size
                batch_n += n

            if batch_n > 0:
                yield (np.concatenate(p_list, axis=0),
                       np.concatenate(c_list, axis=0),
                       np.concatenate(pl_list, axis=0),
                       np.array([tp.shape[0] for tp in p_list]),
                       np.concatenate(pi_list, axis=0),
                       np.array(ci_list, dtype=np.int32))

        ###################
        # Choose generators
        ###################

        # Define the generator that should be used for this split
        if split == 'training':
            gen_func = spatially_regular_gen
        elif split == 'validation':
            gen_func = spatially_regular_gen
        elif split in ['test', 'ERF']:
            gen_func = spatially_regular_gen
        else:
            raise ValueError('Split argument in data generator should be "training", "validation" or "test"')

        # Define generated types and shapes
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 4], [None], [None], [None], [None])

        return gen_func, gen_types, gen_shapes

    def get_tf_mapping_radius(self):
        config = self.config

        # Returned mapping function
        def tf_map(stacked_points, stacked_colors, point_labels, stacks_lengths, point_inds, cloud_inds):
            """
            [None, 3], [None, 3], [None], [None]
            """

            # Get batch indice for each point
            batch_inds = self.tf_get_batch_inds(stacks_lengths)

            # Augment input points
            stacked_points, scales, rots = self.tf_augment_input(stacked_points, batch_inds)

            # First add a column of 1 as feature for the network to be able to learn 3D shapes
            stacked_features = tf.ones((tf.shape(stacked_points)[0], 1), dtype=tf.float32)

            # Get coordinates and colors
            stacked_original_coordinates = stacked_colors[:, 1:]  # 
            stacked_colors = stacked_colors[:, :1]

            # Augmentation : randomly drop colors
            in_features = config.in_features.replace('-', '')
            if 'rgb' in in_features and config.augment_color:
                num_batches = batch_inds[-1] + 1
                s = tf.cast(tf.less(tf.random_uniform((num_batches,)), config.augment_color), tf.float32)
                stacked_s = tf.gather(s, batch_inds)
                stacked_colors = stacked_colors * tf.expand_dims(stacked_s, axis=1)

            # Then use positions or not
            if in_features == '1':
                pass
            elif in_features == '1Z':
                stacked_features = tf.concat((stacked_features, stacked_original_coordinates[:, 2:]), axis=1)
            elif in_features == '1rZ':
                stacked_features = tf.concat((stacked_features, stacked_colors, stacked_original_coordinates[:, 2:]), axis=1)
            elif in_features == '1rXYZ':
                stacked_features = tf.concat((stacked_features, stacked_colors), axis=1)
            else:
                raise ValueError(f'invalid in_features = {in_features}')

            # Get the whole input list
            inputs = self.tf_segmentation_inputs_radius(stacked_points,
                                                        stacked_features,
                                                        point_labels,
                                                        stacks_lengths,
                                                        batch_inds)

            if isinstance(inputs, dict):
                inputs.update({
                    'augment_scales': scales, 'augment_rotations': rots,
                    'point_inds': point_inds, 'cloud_inds': cloud_inds,
                })
            else:
                # Add scale and rotation for testing
                inputs += [scales, rots]
                inputs += [point_inds, cloud_inds]

            return inputs

        return tf_map

    def load_evaluation_points(self, file_path):
        """
        Load points (from test or validation split) on which the metrics should be evaluated
        """

        data = read_ply(file_path)
        return np.vstack((data['x'], data['y'], data['z'])).astype(np.float32).T

    # Debug methods
    # ------------------------------------------------------------------------------------------------------------------

    def check_input_pipeline_timing(self, config):

        # Create a session for running Ops on the Graph.
        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Initialise iterator with train data
        self.sess.run(self.train_init_op)

        # Run some epochs
        n_b = config.batch_num
        t0 = time.time()
        mean_dt = np.zeros(2)
        last_display = t0
        epoch = 0
        training_step = 0
        while epoch < 100:

            try:
                # Run one step of the model.
                t = [time.time()]
                ops = self.flat_inputs

                # Get next inputs
                np_flat_inputs = self.sess.run(ops)
                t += [time.time()]

                # Restructure flatten inputs
                points = np_flat_inputs[:config.num_layers]
                neighbors = np_flat_inputs[config.num_layers:2 * config.num_layers]
                batches = np_flat_inputs[-7]
                n_b = 0.99 * n_b + 0.01 * batches.shape[0]
                t += [time.time()]

                # Average timing
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Console display
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Step {:08d} : timings {:4.2f} {:4.2f} - {:d} x {:d} => b = {:.1f}'
                    print(message.format(training_step,
                                         1000 * mean_dt[0],
                                         1000 * mean_dt[1],
                                         neighbors[0].shape[0],
                                         neighbors[0].shape[1],
                                         n_b))

                training_step += 1

            except tf.errors.OutOfRangeError:
                print('End of train dataset')
                self.sess.run(self.train_init_op)
                epoch += 1

        return

    def check_input_pipeline_batches(self, config):

        # Create a session for running Ops on the Graph.
        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Initialise iterator with train data
        self.sess.run(self.train_init_op)

        # Run some epochs
        mean_b = 0
        min_b = 1000000
        max_b = 0
        t0 = time.time()
        mean_dt = np.zeros(2)
        last_display = t0
        epoch = 0
        training_step = 0
        while epoch < 100:

            try:
                # Run one step of the model.
                t = [time.time()]
                ops = self.flat_inputs

                # Get next inputs
                np_flat_inputs = self.sess.run(ops)
                t += [time.time()]

                # Restructure flatten inputs
                points = np_flat_inputs[:config.num_layers]
                neighbors = np_flat_inputs[config.num_layers:2 * config.num_layers]
                batches = np_flat_inputs[-7]

                max_ind = np.max(batches)
                batches_len = [np.sum(b < max_ind-0.5) for b in batches]

                for b_l in batches_len:
                    mean_b = 0.99 * mean_b + 0.01 * b_l
                max_b = max(max_b, np.max(batches_len))
                min_b = min(min_b, np.min(batches_len))

                print('{:d} < {:.1f} < {:d} /'.format(min_b, mean_b, max_b),
                      self.training_batch_limit,
                      batches_len)

                t += [time.time()]

                # Average timing
                mean_dt = 0.01 * mean_dt + 0.99 * (np.array(t[1:]) - np.array(t[:-1]))

                training_step += 1

            except tf.errors.OutOfRangeError:
                print('End of train dataset')
                self.sess.run(self.train_init_op)
                epoch += 1

        return

    def check_input_pipeline_neighbors(self, config):

        # Create a session for running Ops on the Graph.
        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Initialise iterator with train data
        self.sess.run(self.train_init_op)

        # Run some epochs
        hist_n = 500
        neighb_hists = np.zeros((config.num_layers, hist_n), dtype=np.int32)
        t0 = time.time()
        mean_dt = np.zeros(2)
        last_display = t0
        epoch = 0
        training_step = 0
        while epoch < 100:

            try:
                # Run one step of the model.
                t = [time.time()]
                ops = self.flat_inputs

                # Get next inputs
                np_flat_inputs = self.sess.run(ops)
                t += [time.time()]

                # Restructure flatten inputs
                points = np_flat_inputs[:config.num_layers]
                neighbors = np_flat_inputs[config.num_layers:2 * config.num_layers]
                batches = np_flat_inputs[-7]

                for neighb_mat in neighbors:
                    print(neighb_mat.shape)

                counts = [np.sum(neighb_mat < neighb_mat.shape[0], axis=1) for neighb_mat in neighbors]
                hists = [np.bincount(c, minlength=hist_n) for c in counts]

                neighb_hists += np.vstack(hists)

                print('***********************')
                dispstr = ''
                fmt_l = len(str(int(np.max(neighb_hists)))) + 1
                for neighb_hist in neighb_hists:
                    for v in neighb_hist:
                        dispstr += '{num:{fill}{width}}'.format(num=v, fill=' ', width=fmt_l)
                    dispstr += '\n'
                print(dispstr)
                print('***********************')

                t += [time.time()]

                # Average timing
                mean_dt = 0.01 * mean_dt + 0.99 * (np.array(t[1:]) - np.array(t[:-1]))

                training_step += 1

            except tf.errors.OutOfRangeError:
                print('End of train dataset')
                self.sess.run(self.train_init_op)
                epoch += 1

        return

    def check_input_pipeline_colors(self, config):

        # Create a session for running Ops on the Graph.
        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Initialise iterator with train data
        self.sess.run(self.train_init_op)

        # Run some epochs
        t0 = time.time()
        mean_dt = np.zeros(2)
        epoch = 0
        training_step = 0
        while epoch < 100:

            try:
                # Run one step of the model.
                t = [time.time()]
                ops = self.flat_inputs

                # Get next inputs
                np_flat_inputs = self.sess.run(ops)
                t += [time.time()]

                # Restructure flatten inputs
                stacked_points = np_flat_inputs[:config.num_layers]
                stacked_colors = np_flat_inputs[-9]
                batches = np_flat_inputs[-7]
                stacked_labels = np_flat_inputs[-5]

                # Extract a point cloud and its color to save
                max_ind = np.max(batches)
                for b_i, b in enumerate(batches):

                    # Eliminate shadow indices
                    b = b[b < max_ind-0.5]

                    # Get points and colors (only for the concerned parts)
                    points = stacked_points[0][b]
                    colors = stacked_colors[b]
                    labels = stacked_labels[b]

                    write_ply('input_{:d}.ply'.format(b_i),
                              [points, colors[:, 1:4], labels],
                              ['x', 'y', 'z', 'red', 'green', 'blue', 'labels'])

                a = 1/0



                t += [time.time()]

                # Average timing
                mean_dt = 0.01 * mean_dt + 0.99 * (np.array(t[1:]) - np.array(t[:-1]))

                training_step += 1

            except tf.errors.OutOfRangeError:
                print('End of train dataset')
                self.sess.run(self.train_init_op)
                epoch += 1

        return

    def check_debug_input(self, config, path):

        # Get debug file
        file = join(path, 'all_debug_inputs.pkl')
        with open(file, 'rb') as f1:
            inputs = pickle.load(f1)

        #Print inputs
        nl = config.num_layers
        for layer in range(nl):

            print('Layer : {:d}'.format(layer))

            points = inputs[layer]
            neighbors = inputs[nl + layer]
            pools = inputs[2*nl + layer]
            upsamples = inputs[3*nl + layer]

            nan_percentage = 100 * np.sum(np.isnan(points)) / np.prod(points.shape)
            print('Points =>', points.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(neighbors)) / np.prod(neighbors.shape)
            print('neighbors =>', neighbors.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(pools)) / (np.prod(pools.shape) +1e-6)
            print('pools =>', pools.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(upsamples)) / (np.prod(upsamples.shape) +1e-6)
            print('upsamples =>', upsamples.shape, '{:.1f}% NaN'.format(nan_percentage))

        ind = 4 * nl
        features = inputs[ind]
        nan_percentage = 100 * np.sum(np.isnan(features)) / np.prod(features.shape)
        print('features =>', features.shape, '{:.1f}% NaN'.format(nan_percentage))
        ind += 1
        batch_weights = inputs[ind]
        ind += 1
        in_batches = inputs[ind]
        max_b = np.max(in_batches)
        print(in_batches.shape)
        in_b_sizes = np.sum(in_batches < max_b - 0.5, axis=-1)
        print('in_batch_sizes =>', in_b_sizes)
        ind += 1
        out_batches = inputs[ind]
        max_b = np.max(out_batches)
        print(out_batches.shape)
        out_b_sizes = np.sum(out_batches < max_b - 0.5, axis=-1)
        print('out_batch_sizes =>', out_b_sizes)
        ind += 1
        point_labels = inputs[ind]
        ind += 1
        if config.dataset.startswith('ShapeNetPart_multi'):
            object_labels = inputs[ind]
            nan_percentage = 100 * np.sum(np.isnan(object_labels)) / np.prod(object_labels.shape)
            print('object_labels =>', object_labels.shape, '{:.1f}% NaN'.format(nan_percentage))
            ind += 1
        augment_scales = inputs[ind]
        ind += 1
        augment_rotations = inputs[ind]
        ind += 1

        print('\npoolings and upsamples nums :\n')

        #Print inputs
        nl = config.num_layers
        for layer in range(nl):

            print('\nLayer : {:d}'.format(layer))

            neighbors = inputs[nl + layer]
            pools = inputs[2*nl + layer]
            upsamples = inputs[3*nl + layer]

            max_n = np.max(neighbors)
            nums = np.sum(neighbors < max_n - 0.5, axis=-1)
            print('min neighbors =>', np.min(nums))

            if np.prod(pools.shape) > 0:
                max_n = np.max(pools)
                nums = np.sum(pools < max_n - 0.5, axis=-1)
                print('min pools =>', np.min(nums))

            if np.prod(upsamples.shape) > 0:
                max_n = np.max(upsamples)
                nums = np.sum(upsamples < max_n - 0.5, axis=-1)
                print('min upsamples =>', np.min(nums))


        print('\nFinished\n\n')

