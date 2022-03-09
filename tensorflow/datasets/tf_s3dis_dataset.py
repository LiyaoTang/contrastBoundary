import os, re, sys, glob, time, pickle
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KDTree

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

from utils.ply import read_ply, write_ply
from .base import Dataset

class S3DISDataset(Dataset):

    # Dict from labels to names
    label_to_names = {
        0: 'ceiling',
        1: 'floor',
        2: 'wall',
        3: 'beam',
        4: 'column',
        5: 'window',
        6: 'door',
        7: 'chair',
        8: 'table',
        9: 'bookcase',
        10: 'sofa',
        11: 'board',
        12: 'clutter',
        # 13: 'empty',
    }
    ignored_labels = np.sort([])  # 13

    label_to_rgb = np.array([
        [233, 229, 107],    # 'ceiling'     ->  yellow
        [95, 156, 196],     # 'floor'       ->  blue
        [179, 116, 81],     # 'wall'        ->  brown
        [241, 149, 131],    # 'beam'        ->  salmon
        [81, 163, 148],     # 'column'      ->  bluegreen
        [77, 174, 84],      # 'window'      ->  bright green
        [108, 135, 75],     # 'door'        ->  dark green
        [41, 49, 101],      # 'chair'       ->  darkblue
        [79, 79, 76],       # 'table'       ->  dark grey
        [223, 52, 52],      # 'bookcase'    ->  red
        [89, 47, 95],       # 'sofa'        ->  purple
        [81, 109, 114],     # 'board'       ->  grey
        [233, 233, 229],    # 'clutter'     ->  light grey
        # [0, 0, 0,],         # 'empty'       ->  dark
    ])

    def __init__(self, config, verbose=True, input_threads=8, parse=True):
        """Class to handle S3DIS dataset for scene segmentation task.

        Args:
            config: config file
            input_threads: the number elements to process in parallel
        """
        super(S3DISDataset, self).__init__(config)
        self.config = config
        self.verbose = verbose

        # Initiate a bunch of variables concerning class labels
        self.init_labels()

        # Number of input threads
        self.num_threads = max([input_threads, config.gpu_num, config.num_threads])

        version = config.version if config.version else 'full'
        if version == 'aligned':
            version = 'Stanford3dDataset_v1.2_Aligned_Version'
            self.prepare_S3DIS_ply = self.prepare_S3DIS_ply_aligned
            self.load_subsampled_clouds = self.load_subsampled_clouds_aligned
        else:
            assert version == 'full'
            version = 'Stanford3dDataset_v1.2'

        # Path of the folder containing ply files
        self.path = os.path.join(self.data_path, version)
        # Path of the training files
        self.train_path = 'original_ply'
        # List of files to process
        ply_path = os.path.join(self.path, self.train_path)
        # Proportion of validation scenes
        self.cloud_names = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6']
        self.all_splits = [0, 1, 2, 3, 4, 5]
        self.validation_split = int(config.validation_split) if config.validation_split else 4
        # List of training files
        self.train_files = [os.path.join(ply_path, f + '.ply') for f in self.cloud_names]

        # Some configs
        self.gpu_num = config.gpu_num
        self.in_features_dim = config.in_features_dim
        self.num_layers = config.num_layers
        self.augment_scale_anisotropic = config.augment_scale_anisotropic
        self.augment_symmetries = config.augment_symmetries
        self.augment_rotation = config.augment_rotation
        self.augment_scale_min = config.augment_scale_min
        self.augment_scale_max = config.augment_scale_max
        self.augment_noise = config.augment_noise
        self.augment_color = config.augment_color
        self.epoch_steps = config.epoch_steps
        self.validation_steps = config.validation_steps

        # prepare ply file
        self.prepare_S3DIS_ply(verbose=verbose)

        # input subsampling
        self.load_subsampled_clouds(config.first_subsampling_dl, verbose=verbose)

        # # initialize op - manual call
        # self.initialize(verbose=verbose)


    def prepare_S3DIS_ply(self, verbose):

        if verbose:
            print('\nPreparing ply files')
        t0 = time.time()

        # Folder for the ply files
        ply_path = os.path.join(self.path, self.train_path)
        os.makedirs(ply_path, exist_ok=True)

        for cloud_name in self.cloud_names:

            # Pass if the cloud has already been computed
            cloud_file = os.path.join(ply_path, cloud_name + '.ply')
            if os.path.exists(cloud_file):
                continue

            # Get rooms of the current cloud
            cloud_folder = os.path.join(self.path, cloud_name)
            room_folders = [os.path.join(cloud_folder, room) for room in os.listdir(cloud_folder) if
                            os.path.isdir(os.path.join(cloud_folder, room))]

            # Initiate containers
            cloud_points = np.empty((0, 3), dtype=np.float32)
            cloud_colors = np.empty((0, 3), dtype=np.uint8)
            cloud_classes = np.empty((0, 1), dtype=np.int32)

            # Loop over rooms
            for i, room_folder in enumerate(room_folders):

                if verbose:
                    print('Cloud %s - Room %d/%d : %s' % (cloud_name, i + 1, len(room_folders), room_folder.split('\\')[-1]))

                for object_name in os.listdir(os.path.join(room_folder, 'Annotations')):

                    if object_name[-4:] == '.txt':

                        # Text file containing point of the object
                        object_file = os.path.join(room_folder, 'Annotations', object_name)

                        # Object class and ID
                        tmp = object_name[:-4].split('_')[0]
                        if tmp in self.name_to_label:
                            object_class = self.name_to_label[tmp]
                        elif tmp in ['stairs']:
                            object_class = self.name_to_label['clutter']
                        else:
                            raise ValueError('Unknown object name: ' + str(tmp))

                        # Read object points and colors
                        with open(object_file, 'r') as f:
                            try:
                                object_data = np.array([[float(x) for x in line.split()] for line in f])
                            except:
                                # NOTE: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
                                for i, l in enumerate(open(object_file, 'r').read().split('\n')):
                                    try:
                                        l = [float(x) for x in l.split()]
                                    except:
                                        print(f'parsing line {i} of file {object_file}')
                                        raise
                                raise

                        # Stack all data
                        cloud_points = np.vstack((cloud_points, object_data[:, 0:3].astype(np.float32)))
                        cloud_colors = np.vstack((cloud_colors, object_data[:, 3:6].astype(np.uint8)))
                        object_classes = np.full((object_data.shape[0], 1), object_class, dtype=np.int32)
                        cloud_classes = np.vstack((cloud_classes, object_classes))

            # Save as ply
            write_ply(cloud_file,
                      (cloud_points, cloud_colors, cloud_classes),
                      ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])
        if verbose:
            print('Done in {:.1f}s'.format(time.time() - t0))

    def load_subsampled_clouds(self, subsampling_parameter, verbose=True):
        """
        Presubsample point clouds and load into memory (Load KDTree for neighbors searches)
        """

        if 0 < subsampling_parameter <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')

        from ops import get_tf_func
        grid_subsampling = get_tf_func('grid_preprocess')

        # Create path for files
        folder_name = 'input_{:.3f}'.format(subsampling_parameter)
        tree_path = os.path.join(self.path, folder_name)
        if not os.path.exists(tree_path):
            os.makedirs(tree_path)
        if verbose:
            print(f'looking into {tree_path}...')

        # Initiate containers
        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}

        for i, file_path in enumerate(self.train_files):

            # Restart timer
            t0 = time.time()

            # get cloud name and split
            cloud_name = file_path.split(os.sep)[-1][:-4]
            if self.all_splits[i] == self.validation_split:
                cloud_split = 'validation'
            else:
                cloud_split = 'training'

            # Name of the input files
            KDTree_file = os.path.join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = os.path.join(tree_path, '{:s}.ply'.format(cloud_name))

            # Check if inputs have already been computed
            if verbose:
                action = 'Found' if os.path.isfile(KDTree_file) else 'Preparing'
                print(f'\n{action} KDTree for cloud {cloud_name}, subsampled at {subsampling_parameter:.3f}')

            if os.path.isfile(KDTree_file):
                # read ply with data
                data = read_ply(sub_ply_file)
                sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
                sub_labels = data['class']

                # Read pkl with search tree
                with open(KDTree_file, 'rb') as f:
                    search_tree = pickle.load(f)

            else:
                # Read ply file
                data = read_ply(file_path)
                points = np.vstack((data['x'], data['y'], data['z'])).T
                colors = np.vstack((data['red'], data['green'], data['blue'])).T
                labels = data['class']

                # Subsample cloud
                sub_points, sub_colors, sub_labels = grid_subsampling(points,
                                                                      features=colors,
                                                                      labels=labels,
                                                                      sampleDl=subsampling_parameter)

                # Rescale float color and squeeze label
                sub_colors = sub_colors / 255
                sub_labels = np.squeeze(sub_labels)

                # Get chosen neighborhoods
                search_tree = KDTree(sub_points, leaf_size=50)

                # Save KDTree
                with open(KDTree_file, 'wb') as f:
                    pickle.dump(search_tree, f)

                # Save ply
                write_ply(sub_ply_file,
                          [sub_points, sub_colors, sub_labels],
                          ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

            # Fill data containers
            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]
            self.input_names[cloud_split] += [cloud_name]

            size = sub_colors.shape[0] * 4 * 7
            if verbose:
                print(f'{size * 1e-6:.1f} MB loaded in {time.time() - t0:.1f}s')

        if verbose:
            print('\nPreparing reprojection indices for testing')

        # Get number of clouds
        self.num_training = len(self.input_trees['training'])
        self.num_validation = len(self.input_trees['validation'])

        # Get validation and test reprojection indices
        self.validation_proj = []
        self.validation_labels = []
        i_val = 0
        for i, file_path in enumerate(self.train_files):

            # Restart timer
            t0 = time.time()

            # Get info on this cloud
            cloud_name = file_path.split(os.sep)[-1][:-4]

            # Validation projection and labels
            if self.all_splits[i] == self.validation_split:
                proj_file = os.path.join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                if os.path.isfile(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds, labels = pickle.load(f)
                else:
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
                if verbose:
                    print(f'{cloud_name} done in {time.time() - t0:.1f}s')

        # class cnt over all clouds in val set (w/o invalid class)
        self.val_proportions_full = np.array([np.sum([np.sum(labels == label_val) for labels in self.validation_labels]) for label_val in self.label_values])
        self.val_proportions = np.array([p for l, p in zip(self.label_values, self.val_proportions_full) if l not in self.ignored_labels])

        return

    def init_sampling(self, split):
        ############
        # Parameters
        ############

        # Initiate parameters depending on the chosen split
        if split == 'training':  # First compute the number of point we want to pick in each cloud set - num of samples
            epoch_n = self.epoch_steps * self.config.epoch_batch
        elif split == 'validation':
            epoch_n = self.validation_steps * self.config.epoch_batch
        elif split == 'test':
            epoch_n = self.validation_steps * self.config.epoch_batch
        else:
            raise ValueError('Split argument in data generator should be "training", "validation" or "test"')

        # Initiate potentials for regular generation
        if not hasattr(self, 'potentials'):
            self.potentials = {}
            self.min_potentials = {}

        # Reset potentials
        self.potentials[split] = []
        self.min_potentials[split] = []
        for i, tree in enumerate(self.input_trees[split]):
            self.potentials[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_potentials[split] += [float(np.min(self.potentials[split][-1]))]

        return epoch_n


    def get_batch_gen_radius(self, split):
        """
        A function defining the batch generator for each split. Should return the generator, the generated types and
        generated shapes
        """
        if split == 'test':
            return None, None, None
        return super(S3DISDataset, self).get_batch_gen_radius(split)

        epoch_n = self.init_sampling(split)
        data_split = split

        ##########################
        # Def generators
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
                noise = np.random.normal(scale=self.config.in_radius / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)

                # Indices of points in input region
                input_inds = self.input_trees[data_split][cloud_ind].query_radius(pick_point, r=self.config.in_radius)[0]

                # Number collected
                n = input_inds.shape[0]

                # Update potentials (Tuckey weights)
                dists = np.sum(np.square((points[input_inds] - pick_point).astype(np.float32)), axis=1)
                tukeys = np.square(1 - dists / np.square(self.config.in_radius))
                tukeys[dists > np.square(self.config.in_radius)] = 0
                self.potentials[split][cloud_ind][input_inds] += tukeys
                self.min_potentials[split][cloud_ind] = float(np.min(self.potentials[split][cloud_ind]))

                # Safe check for very dense areas
                if n > self.batch_limit:
                    input_inds = np.random.choice(input_inds, size=int(self.batch_limit) - 1, replace=False)
                    n = input_inds.shape[0]

                # Collect points and colors
                input_points = (points[input_inds] - pick_point).astype(np.float32)
                input_colors = self.input_colors[data_split][cloud_ind][input_inds]
                if split == 'test':
                    input_labels = np.zeros(input_points.shape[0])
                else:
                    input_labels = self.input_labels[data_split][cloud_ind][input_inds]
                    input_labels = self.label_to_idx[input_labels]
                    # input_labels = np.array([self.label_to_idx[l] for l in input_labels])

                # In case batch is full, yield it and reset it
                if batch_n + n > self.batch_limit and batch_n > 0:
                    yield (np.concatenate(p_list, axis=0),  # [BxN, 3]  - xyz in sample
                           np.concatenate(c_list, axis=0),  # [BxN, 6 (RGB + global xyz in whole cloud)]
                           np.concatenate(pl_list, axis=0), # [BxN]     - labels
                           np.array([tp.shape[0] for tp in p_list]),    # [B]    - size (point num) of each batch
                           np.concatenate(pi_list, axis=0),             # [B, N] - point idx in each of its point cloud
                           np.array(ci_list, dtype=np.int32))           # [B]    - cloud idx

                    p_list = []
                    c_list = []
                    pl_list = []
                    pi_list = []
                    ci_list = []
                    batch_n = 0

                # Add data to current batch
                if n > 0:
                    p_list += [input_points]
                    c_list += [np.hstack((input_colors, input_points + pick_point))]
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

        # Define the generator that should be used for this split
        if split == 'training':
            gen_func = spatially_regular_gen
        elif split == 'validation':
            gen_func = spatially_regular_gen
        elif split == 'test':
            gen_func = spatially_regular_gen
        else:
            raise ValueError('Split argument in data generator should be "training", "validation" or "test"')

        # Define generated types and shapes
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 6], [None], [None], [None], [None])

        return gen_func, gen_types, gen_shapes

    def get_batch_gen_fixed_size(self, split):

        if split == 'test':
            return None, None, None

        epoch_n = self.init_sampling(split)

        def spatially_regular_gen():
            # Generator loop
            for i in range(epoch_n):

                # Choose a random cloud
                cloud_ind = int(np.argmin(self.min_potentials[split]))

                # Choose point ind as minimum of potentials
                point_ind = np.argmin(self.potentials[split][cloud_ind])

                # Get points from tree structure
                points = np.array(self.input_trees[split][cloud_ind].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=self.config.noise_init / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)

                # Check if the number of points in the selected cloud is less than the predefined num_points
                k = min(len(points), self.config.in_points)

                # Query all points / the predefined number within the cloud
                input_inds = self.input_trees[split][cloud_ind].query(pick_point, k=k)[1][0]

                # Shuffle index
                np.random.shuffle(input_inds)

                # Collect points and colors
                input_points = (points[input_inds] - pick_point).astype(np.float32)
                input_colors = self.input_colors[split][cloud_ind][input_inds]
                if split == 'test':
                    input_labels = np.zeros(input_points.shape[0])
                else:
                    input_labels = self.input_labels[split][cloud_ind][input_inds]
                    input_labels = self.label_to_idx[input_labels]

                # Update potentials (Tuckey weights)
                dists = np.sum(np.square((points[input_inds] - pick_point).astype(np.float32)), axis=1)
                tukeys = np.square(1 - dists / dists.max())
                # # weighted update
                # tukeys_cls_w = class_weight[split][input_labels] if split == 'train' else 1  # per-pt class weight
                self.potentials[split][cloud_ind][input_inds] += tukeys
                self.min_potentials[split][cloud_ind] = float(np.min(self.potentials[split][cloud_ind]))

                # up_sampled with replacement
                if len(input_points) < self.config.in_points:
                    dup_idx = np.random.choice(len(points), self.config.in_points - len(points))
                    dup_idx = np.concatenate([np.arange(len(points)), dup_idx])  # [original, dup]
                    input_points = input_points[dup_idx]
                    input_colors = input_colors[dup_idx]
                    input_labels = input_labels[dup_idx]
                    input_inds = input_inds[dup_idx]

                # sampled point cloud
                yield (input_points.astype(np.float32),  # centered xyz
                        np.hstack([input_colors, input_points + pick_point]).astype(np.float32),  # colors, original xyz
                        input_labels,  # label
                        input_inds.astype(np.int32),  # points idx in cloud
                        int(cloud_ind)  # cloud idx
                        # np.array([cloud_ind], dtype=np.int32)
                    )

        # Define the generator that should be used for this split
        valid_split = ('training', 'validation', 'test')
        assert split in valid_split, ValueError(f'invalid split = {split} not in {valid_split}')

        # Define generated types and shapes
        gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)
        # N = None
        N = self.config.in_points
        gen_shapes = ([N, 3], [N, 6], [N], [N], [])  # after batch : [B, N, 3], [B, N, 6], [B, N], [B, N], [B]
        # gen_shapes = ([N, 3], [N, 6], [N], [N], [1])
        return gen_func, gen_types, gen_shapes


    def get_tf_mapping_radius(self, **kwargs):

        config = self.config
        # Returned mapping function
        def tf_map(stacked_points, stacked_colors, point_labels, stacks_lengths, point_inds, cloud_inds, v_list=None):
            """
            Args:
                stacked_points  : [BxN, 3]  - xyz in sample
                stacked_colors  : [BxN, 6]  - rgb,  xyz in whole cloud (global xyz)
                point_labels    : [BxN]     - labels
                stacks_lengths  : [B]    - size (point num) of each batch
                point_inds      : [BxN]  - point idx in each of its point cloud
                cloud_inds      : [B]    - cloud idx
            """

            # Get batch indice for each point
            batch_inds = self.tf_get_batch_inds(stacks_lengths)  # [BxN]

            # Augmentation: input points (xyz) - rotate, scale (flip), jitter
            stacked_points, scales, rots = self.tf_augment_input(stacked_points,
                                                                 batch_inds)

            # First add a column of 1 as feature for the network to be able to learn 3D shapes
            stacked_features = tf.ones((tf.shape(stacked_points)[0], 1), dtype=tf.float32)

            # Get coordinates and colors
            stacked_original_coordinates = stacked_colors[:, 3:]
            stacked_colors = stacked_colors[:, :3]
            stacked_rgb = stacked_colors

            # Augmentation : randomly drop colors
            in_features = config.in_features.replace('-', '')
            if 'rgb' in in_features and config.augment_color:
                num_batches = batch_inds[-1] + 1
                s = tf.cast(tf.less(tf.random.uniform((num_batches,)), config.augment_color), tf.float32)
                stacked_s = tf.gather(s, batch_inds)
                stacked_colors = stacked_colors * tf.expand_dims(stacked_s, axis=1)

            # Then use positions or not
            if in_features == '1':
                pass
            elif in_features == '1Z':
                stacked_features = tf.concat((stacked_features, stacked_original_coordinates[..., 2:]), axis=1)
            elif in_features == 'rgb':
                stacked_features = stacked_colors
            elif in_features == '1rgb':
                stacked_features = tf.concat((stacked_features, stacked_colors), axis=1)
            elif in_features == '1rgbZ':  # used in provided cfgs - [1, rgb, Z]
                stacked_features = tf.concat((stacked_features, stacked_colors, stacked_original_coordinates[..., 2:]), axis=1)
            elif in_features == '1rgbxyz':
                stacked_features = tf.concat((stacked_features, stacked_colors, stacked_points), axis=1)
            elif in_features == '1rgbxyzZ':
                stacked_features = tf.concat((stacked_features, stacked_colors, stacked_points, stacked_original_coordinates[..., 2:]), axis=1)
            else:
                raise ValueError('Only accepted input dimensions are 1, 3, 4 and 7 (without and with rgb/xyz)')

            # Get the whole input list
            inputs = self.tf_segmentation_inputs_radius(stacked_points,
                                                        stacked_features,  # [BxN, d]
                                                        point_labels,
                                                        stacks_lengths,
                                                        batch_inds)

            kv = {}
            if v_list:
                assert 'k_list' in kwargs, f'missing key \'k_list\' from {kwargs}'
                kv.update(dict(zip(kwargs['k_list'], v_list)))

            if config.info_gen and 'rgb' in config.info_gen:  # original rgb
                kv['rgb'] = stacked_rgb
            if config.info_gen and 'xyz' in config.info_gen:  # original xyz (global)
                kv['xyz'] = stacked_original_coordinates
            if config.info_gen and 'overlap' in config.info_gen:  # overlap idx
                from models.utils import tf_overlap_points
                impl = config.info_gen.split('overlap')[-1].split('-')[0]
                impl = impl if impl else 'tfmap'  # seems to be faster than 'tfwhile', but has problem with gpu
                kv['overlap_idx'] = tf_overlap_points(point_inds, cloud_inds, stacks_lengths, config, impl=impl, impl_cloud='hash')

            # Add scale and rotation for testing
            if isinstance(inputs, dict):
                inputs.update({
                    'augment_scales': scales, 'augment_rotations': rots,
                    'point_inds': point_inds, 'cloud_inds': cloud_inds,
                })
                if kv:
                    inputs.update(kv)
            else:
                inputs += [scales, rots]
                inputs += [point_inds, cloud_inds]

            return inputs

        return tf_map

    def get_tf_mapping_fixed_size(self):

        config = self.config
        # Returned mapping function
        def tf_map(stacked_points, stacked_colors, point_labels, point_inds, cloud_inds):
            """
            Args:
                stacked_points  : [B, N, 3]  - xyz in sample
                stacked_colors  : [B, N, 6]  - rgb,  xyz in whole cloud (global xyz)
                point_labels    : [B, N]     - labels
                point_inds      : [B, N] - point idx in each of its point cloud
                cloud_inds      : [B]    - cloud idx
            """

            # Get batch indice for each point
            BN = tf.shape(stacked_points)
            batch_inds = tf.range(BN[0])  # [B] - may have leftover sample (to avoid: set drop_reminder=True)

            # Augmentation: input points (xyz) - rotate, scale (flip), jitter
            stacked_points, scales, rots = self.tf_augment_input(stacked_points, batch_inds)

            # First add a column of 1 as feature for the network to be able to learn 3D shapes
            stacked_features = tf.ones((BN[0], BN[1], 1), dtype=tf.float32)

            # Get coordinates and colors
            stacked_original_coordinates = stacked_colors[:, :, 3:]
            stacked_colors = stacked_colors[:, :, :3]
            stacked_rgb = stacked_colors

            # Augmentation : randomly drop colors
            in_features = config.in_features.replace('-', '')
            if 'rgb' in in_features and config.augment_color:
                s = tf.cast(tf.less(tf.random.uniform(BN[:1]), config.augment_color), tf.float32)  # [B]
                # stacked_s = tf.gather(s, batch_inds)
                stacked_s = tf.expand_dims(s, axis=-1)  # [B, 1]
                stacked_colors = stacked_colors * tf.expand_dims(stacked_s, axis=-1)

            stacked_fin = []  # default to '1rgbZ'
            if not (in_features and re.fullmatch('(|1)(|rgb)(|xyz)(|XYZ)(|Z)', in_features)):
                raise ValueError(f'Not accepting in features = {in_features}')
            if '1' in in_features:
                # First add a column of 1 as feature for the network to be able to learn 3D shapes
                stacked_fin += [tf.ones((BN[0], BN[1], 1), dtype=tf.float32)]
            if 'rgb' in in_features:
                stacked_fin += [stacked_colors]
            if 'xyz' in in_features:
                stacked_fin += [stacked_points]
            if 'XYZ' in in_features:
                stacked_fin += [stacked_original_coordinates]
            if 'Z' in in_features and 'XYZ' not in in_features:
                stacked_fin += [stacked_original_coordinates[..., 2:]]
            stacked_features = tf.concat(stacked_fin, axis=-1) if len(stacked_fin) > 1 else stacked_fin[0]

            # Get the whole input list
            inputs = self.tf_segmentation_inputs_fixed_size(stacked_points,    # [B, N, 3]
                                                            stacked_features,  # [B, N, d]
                                                            point_labels)

            # Add scale and rotation for testing
            if isinstance(inputs, dict):
                inputs.update({
                    'rgb': stacked_rgb,
                    'augment_scales': scales, 'augment_rotations': rots,
                    'point_inds': point_inds, 'cloud_inds': cloud_inds,
                })
            else:
                inputs += [scales, rots]
                inputs += [point_inds, cloud_inds]

            return inputs

        return tf_map


    def load_evaluation_points(self, file_path):
        """
        Load points (from test or validation split) on which the metrics should be evaluated
        """

        # Get original points
        data = read_ply(file_path)
        return np.vstack((data['x'], data['y'], data['z'])).T


    def tf_segmentation_inputs_radius(self,
                                      stacked_points,
                                      stacked_features,
                                      point_labels,
                                      stacks_lengths,
                                      batch_inds):
        from ops import get_tf_func
        tf_batch_subsampling = get_tf_func(self.config.sample, verbose=self.verbose)
        tf_batch_neighbors = get_tf_func(self.config.search, verbose=self.verbose)

        # Batch weight at each point for loss (inverse of stacks_lengths for each point)
        min_len = tf.reduce_min(stacks_lengths, keepdims=True)
        batch_weights = tf.cast(min_len, tf.float32) / tf.cast(stacks_lengths, tf.float32)
        stacked_weights = tf.gather(batch_weights, batch_inds)
        # Starting radius of convolutions
        dl = self.config.first_subsampling_dl
        dp = self.config.density_parameter
        r = dl * dp / 2.0
        # Lists of inputs
        num_layers = self.config.num_layers
        downsample_times = num_layers - 1
        input_points = [None] * num_layers
        input_neighbors = [None] * num_layers
        input_pools = [None] * num_layers
        input_upsamples = [None] * num_layers
        input_batches_len = [None] * num_layers

        input_upsamples[0] = tf.zeros((0, 1), dtype=tf.int32)  # no upsample for input pt
        for dt in range(0, downsample_times):
            neighbors_inds = tf_batch_neighbors(stacked_points, stacked_points, stacks_lengths, stacks_lengths, r)
            pool_points, pool_stacks_lengths = tf_batch_subsampling(stacked_points, stacks_lengths, sampleDl=2 * dl)
            pool_inds = tf_batch_neighbors(pool_points, stacked_points, pool_stacks_lengths, stacks_lengths, r)
            up_inds = tf_batch_neighbors(stacked_points, pool_points, stacks_lengths, pool_stacks_lengths, 2 * r)

            neighbors_inds = self.big_neighborhood_filter(neighbors_inds, dt)
            pool_inds = self.big_neighborhood_filter(pool_inds, dt)
            up_inds = self.big_neighborhood_filter(up_inds, dt)

            input_points[dt] = stacked_points
            input_neighbors[dt] = neighbors_inds
            input_pools[dt] = pool_inds
            input_upsamples[dt + 1] = up_inds
            input_batches_len[dt] = stacks_lengths
            stacked_points = pool_points
            stacks_lengths = pool_stacks_lengths
            r *= 2
            dl *= 2

        # last (downsampled) layer points
        neighbors_inds = tf_batch_neighbors(stacked_points, stacked_points, stacks_lengths, stacks_lengths, r)
        neighbors_inds = self.big_neighborhood_filter(neighbors_inds, downsample_times)
        input_points[downsample_times] = stacked_points
        input_neighbors[downsample_times] = neighbors_inds
        input_pools[downsample_times] = tf.zeros((0, 1), dtype=tf.int32)
        input_batches_len[downsample_times] = stacks_lengths

        # Batch unstacking (with first layer indices for optional classif loss) - in_batches - input stage
        stacked_batch_inds_0 = self.tf_stack_batch_inds(input_batches_len[0])
        # Batch unstacking (with last layer indices for optional classif loss) - out_batches - most down-sampled stage
        stacked_batch_inds_1 = self.tf_stack_batch_inds(input_batches_len[-1])

        # list of network inputs
        input_dict = {
            'points': tuple(input_points),
            'neighbors': tuple(input_neighbors),
            'pools': tuple(input_pools),
            'upsamples': tuple(input_upsamples),
            'batches_len': tuple(input_batches_len),  # [[B], ...] - size of each point cloud in current batch
            'features': stacked_features,
            'batch_weights': stacked_weights,
            'in_batches': stacked_batch_inds_0,
            'out_batches': stacked_batch_inds_1,
            'point_labels': point_labels,
        }

        return input_dict
        # li = input_points + input_neighbors + input_pools + input_upsamples
        # li += [stacked_features, stacked_weights, stacked_batch_inds_0, stacked_batch_inds_1]
        # li += [point_labels]
        # return li


    def prepare_S3DIS_ply_aligned(self, verbose):

        original_pc_folder = os.path.join(self.path, 'original_ply')
        os.makedirs(original_pc_folder, exist_ok=True)

        for anno_path in glob.glob(f'{self.path}/Area_*/*/Annotations'):
            elements = str(anno_path).split(os.sep)
            out_file_name = elements[-3] + '_' + elements[-2] + '.ply'  # ".../Area_1/hallway_1" becomes ".../Area_1_hallway_1.ply"
            save_path = os.path.join(original_pc_folder, out_file_name)

            if os.path.isfile(save_path):
                continue

            if verbose:
                print(f'prepare {anno_path} into {out_file_name}')

            cloud_points = np.empty((0, 3), dtype=np.float32)
            cloud_colors = np.empty((0, 3), dtype=np.uint8)
            cloud_classes = np.empty((0, 1), dtype=np.int32)

            for object_file in glob.glob(os.path.join(anno_path, '*.txt')):
                # label denoted by file name; point [xyz-rgb] stored as line in the file

                class_name = os.path.basename(object_file).split('_')[0]
                if class_name == 'staris':  # note: in some room there is 'staris' class..
                    class_name = 'clutter'  # clutter as 'others'

                if class_name in self.name_to_label:
                    object_class = self.name_to_label[class_name]
                elif class_name in ['stairs']:
                    object_class = self.name_to_label['clutter']
                else:
                    raise ValueError('Unknown object name: ' + str(class_name))

                with open(object_file, 'r') as f:
                    try:
                        object_data = np.array([[float(x) for x in line.split()] for line in f if line.strip()])
                        eq = [len(object_data[0]) == len(object_data[i]) for i in range(len(object_data))]
                        if not all(eq):
                            print(object_file, cloud_points.shape, object_data.shape)
                            print(eq.index(False))
                            print([len(i) for i in object_data])
                    except:
                        # NOTE: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
                        for i, l in enumerate(open(object_file, 'r').read().split('\n')):
                            try:
                                l = [float(x) for x in l.split()]
                            except:
                                print(f'parsing line {i} of file {object_file}')
                                raise
                        raise

                # Stack all data
                cloud_points = np.vstack((cloud_points, object_data[:, 0:3].astype(np.float32)))
                cloud_colors = np.vstack((cloud_colors, object_data[:, 3:6].astype(np.uint8)))
                object_classes = np.full((object_data.shape[0], 1), object_class, dtype=np.int32)
                cloud_classes = np.vstack((cloud_classes, object_classes))

            xyz_min = np.amin(cloud_points, axis=0)
            cloud_points -= xyz_min
            write_ply(save_path, (cloud_points, cloud_colors, cloud_classes), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])
        return

    def load_subsampled_clouds_aligned(self, subsampling_parameter, verbose=True):

        config = self.config
        val_name = f'Area_{self.validation_split + 1}'
        all_files = glob.glob(os.path.join(self.path, 'original_ply', '*.ply'))

        folder_name = 'input_{:.3f}'.format(subsampling_parameter)
        tree_path = os.path.join(self.path, folder_name)
        os.makedirs(tree_path, exist_ok=True)
        if verbose:
            print('loading with path =', tree_path)

        # if set(glob.glob(os.path.join(tree_path, '*.ply'))).issuperset(set(all_files)):
        #     # if using processed files only
        #     all_files = glob.glob(os.path.join(tree_path, '*.ply'))

        # Initiate containers
        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}

        from ops import get_tf_func
        grid_subsampling = get_tf_func('grid_preprocess')

        all_files = sorted(all_files)  # fixed order
        for i, file_path in enumerate(all_files):
            t0 = time.time()
            cloud_name = file_path.split(os.sep)[-1][:-4]

            if val_name in cloud_name:
                cloud_split = 'validation'
            else:
                cloud_split = 'training'

            sub_ply_file = os.path.join(tree_path, '{:s}.ply'.format(cloud_name))
            kd_tree_file = os.path.join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))

            if not os.path.exists(sub_ply_file) or not os.path.exists(kd_tree_file):
                # Sub-sample the points it not exists
                data = read_ply(file_path)
                xyz = np.vstack((data['x'], data['y'], data['z'])).T
                colors = np.vstack((data['red'], data['green'], data['blue'])).T
                labels = data['class']
                sub_xyz, sub_colors, sub_labels = grid_subsampling(xyz, colors, labels, subsampling_parameter)

                sub_labels = np.squeeze(sub_labels)
                write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

                # Build KDTree & proj if not exists
                search_tree = KDTree(sub_xyz)
                with open(kd_tree_file, 'wb') as f:
                    pickle.dump(search_tree, f)

            else:
                # Read from file
                data = read_ply(sub_ply_file)
                sub_xyz = np.vstack((data['x'], data['y'], data['z'])).T
                sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
                sub_colors = sub_colors / 255.0
                sub_labels = data['class']

                # Read pkl with search tree
                with open(kd_tree_file, 'rb') as f:
                    search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]
            self.input_names[cloud_split] += [cloud_name]

            size = sub_colors.shape[0] * 4 * 7
            if verbose:
                print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))

        if verbose:
            print('\nPreparing reprojected indices for testing')

        # Get number of clouds
        self.num_training = len(self.input_trees['training'])
        self.num_validation = len(self.input_trees['validation'])

        # Get validation and test reprojected indices
        self.validation_proj = []
        self.validation_labels = []
        i_val = 0
        for i, file_path in enumerate(all_files):
            cloud_name = file_path.split(os.sep)[-1][:-4]

            # Validation projection and labels
            if val_name not in cloud_name:
                continue

            t0 = time.time()
            proj_file = os.path.join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
            if not os.path.exists(proj_file):
                data = read_ply(file_path)
                xyz = np.vstack((data['x'], data['y'], data['z'])).T
                labels = data['class']

                search_tree = self.input_trees['validation'][i_val]
                proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
                proj_idx = proj_idx.astype(np.int32)
                with open(proj_file, 'wb') as f:
                    pickle.dump([proj_idx, labels], f)

            else:
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)

            i_val += 1
            self.validation_proj += [proj_idx]
            self.validation_labels += [labels]
            if verbose:
                print(f'{cloud_name} done in {time.time() - t0:.1f}s')

        # class cnt over all clouds in val set (w/o invalid class)
        self.val_proportions_full = np.array([np.sum([np.sum(labels == label_val) for labels in self.validation_labels]) for label_val in self.label_values])
        self.val_proportions = np.array([p for l, p in zip(self.label_values, self.val_proportions_full) if l not in self.ignored_labels])


