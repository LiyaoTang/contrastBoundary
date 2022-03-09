import os, re, sys, json, time, pickle
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KDTree
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

from utils.ply import read_ply, write_ply
from utils.mesh import rasterize_mesh
from .base import Dataset

class Semantic3DDataset(Dataset):
    """
    Class to handle Semantic3D dataset for segmentation task.
    """

    label_to_names = {0: 'unlabeled',
                      1: 'man-made terrain',
                      2: 'natural terrain',
                      3: 'high vegetation',
                      4: 'low vegetation',
                      5: 'buildings',
                      6: 'hard scape',
                      7: 'scanning artefacts',
                      8: 'cars'}
    ignored_labels = np.sort([0])

    def __init__(self, config, verbose=True, input_threads=8):
        super(Semantic3DDataset, self).__init__(config)
        self.verbose = verbose

        # Initiate a bunch of variables concerning class labels
        self.init_labels()

        # Number of input threads
        self.num_threads = max([input_threads, config.gpu_num, config.num_threads])

        # Path of the folder containing ply files
        self.path = self.data_path
        # Original data path
        self.original_folder = 'original_data'
        # Path of the training files
        self.original_ply = 'original_ply'
        # self.train_path = os.path.join(self.path, 'ply_subsampled/train')
        # self.test_path = os.path.join(self.path, 'ply_full/reduced-8')
        #self.test_path = os.path.join(self.path, 'ply_full/semantic-8')

        # Proportion of validation scenes (chose validation split = 6 for final networks)
        # self.all_splits=[0, 1, 4, 2, 3, 4, 3, 0, 1, 2, 3, 4, 2, 0, 1]
        self.all_splits = [0, 1, 4, 5, 3, 4, 3, 0, 1, 2, 3, 4, 2, 0, 5]
        self.validation_split = int(config.validation_split) if config.validation_split else 5

        self.version = config.version if config.version else 'reduced'
        assert self.version in ['reduced', 'full']

        # Ascii files dict for testing
        self.ascii_files = {'MarketplaceFeldkirch_Station4_rgb_intensity-reduced.ply': 'marketsquarefeldkirch4-reduced.labels',
                            'sg27_station10_rgb_intensity-reduced.ply': 'sg27_10-reduced.labels',
                            'sg28_Station2_rgb_intensity-reduced.ply': 'sg28_2-reduced.labels',
                            'StGallenCathedral_station6_rgb_intensity-reduced.ply': 'stgallencathedral6-reduced.labels',
                            'birdfountain_station1_xyz_intensity_rgb.ply': 'birdfountain1.labels',
                            'castleblatten_station1_intensity_rgb.ply': 'castleblatten1.labels',
                            'castleblatten_station5_xyz_intensity_rgb.ply': 'castleblatten5.labels',
                            'marketplacefeldkirch_station1_intensity_rgb.ply': 'marketsquarefeldkirch1.labels',
                            'marketplacefeldkirch_station4_intensity_rgb.ply': 'marketsquarefeldkirch4.labels',
                            'marketplacefeldkirch_station7_intensity_rgb.ply': 'marketsquarefeldkirch7.labels',
                            'sg27_station10_intensity_rgb.ply': 'sg27_10.labels',
                            'sg27_station3_intensity_rgb.ply': 'sg27_3.labels',
                            'sg27_station6_intensity_rgb.ply': 'sg27_6.labels',
                            'sg27_station8_intensity_rgb.ply': 'sg27_8.labels',
                            'sg28_station2_intensity_rgb.ply': 'sg28_2.labels',
                            'sg28_station5_xyz_intensity_rgb.ply': 'sg28_5.labels',
                            'stgallencathedral_station1_intensity_rgb.ply': 'stgallencathedral1.labels',
                            'stgallencathedral_station3_intensity_rgb.ply': 'stgallencathedral3.labels',
                            'stgallencathedral_station6_intensity_rgb.ply': 'stgallencathedral6.labels'}

        # Prepare ply files - as well as finding train/test files
        self.prepare_data()

        # input subsampling
        self.load_subsampled_clouds(config.first_subsampling_dl, verbose=verbose)
        # # initialize op - manual call
        # self.initialize(verbose=verbose)

    def prepare_data(self):
        """
        Download and precompute Seamntic3D point clouds
        """

        from ops import get_tf_func
        grid_subsampling = get_tf_func('grid_preprocess')

        # Folder names
        old_folder = os.path.join(self.path, self.original_folder)
        ply_folder = os.path.join(self.path, self.original_ply)
        os.makedirs(ply_folder, exist_ok=True)

        self.train_files = []
        self.test_files = []

        # Text files containing points
        cloud_names = [file_name[:-4] for file_name in os.listdir(old_folder) if file_name[-4:] == '.txt']

        for cloud_name in cloud_names:

            # Name of the files
            txt_file = os.path.join(old_folder, cloud_name + '.txt')
            label_file = os.path.join(old_folder, cloud_name + '.labels')

            ply_file_full = os.path.join(ply_folder, cloud_name + '.ply')
            if os.path.exists(label_file):
                self.train_files += [ply_file_full]
            elif not self._check_not_required('test', ply_file_full):
                self.test_files += [ply_file_full]

            # Pass if already done
            if os.path.exists(ply_file_full):
                print('{:s} already done'.format(cloud_name))
                continue

            print('Preparation of {:s}'.format(cloud_name))

            data = np.loadtxt(txt_file)

            points = data[:, :3].astype(np.float32)
            colors = data[:, 4:7].astype(np.uint8)

            if os.path.exists(label_file):

                # Load labels
                labels = np.loadtxt(label_file, dtype=np.int32)

                # Subsample to save space
                sub_points, sub_colors, sub_labels = grid_subsampling(points,
                                                                      features=colors,
                                                                      labels=labels,
                                                                      sampleDl=0.01)
                # Write the subsampled ply file
                write_ply(ply_file_full, (sub_points, sub_colors, sub_labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

            else:

                # Write the full ply file
                write_ply(ply_file_full, (points, colors), ['x', 'y', 'z', 'red', 'green', 'blue'])

    def _check_not_required(self, split, file_path):
        return (self.version == 'reduced' and 'reduced' not in file_path and split == 'test') or \
            (self.version == 'full' and 'reduced' in file_path and split == 'test')

    def load_subsampled_clouds(self, subsampling_parameter, verbose=True):
        """
        Presubsample point clouds and load into memory (Load KDTree for neighbors searches
        """

        if 0 < subsampling_parameter <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')

        # Create path for files
        tree_path = os.path.join(self.path, 'input_{:.3f}'.format(subsampling_parameter))
        os.makedirs(tree_path, exist_ok=True)

        # All training and test files
        files = np.hstack((self.train_files, self.test_files))

        # Initiate containers
        self.input_trees = {'training': [], 'validation': [], 'test': []}
        self.input_colors = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': []}

        # Advanced display
        N = len(files)
        progress_n = 30
        fmt_str = '[{:<' + str(progress_n) + '}] {:5.1f}%'
        print('\nPreparing KDTree for all scenes, subsampled at {:.3f}'.format(subsampling_parameter))

        for i, file_path in enumerate(files):

            # Restart timer
            t0 = time.time()

            # get cloud name and split
            cloud_name = file_path.split('/')[-1][:-4]
            cloud_folder = file_path.split('/')[-2]
            if file_path in self.train_files:
                if self.all_splits[i] == self.validation_split:
                    cloud_split = 'validation'
                else:
                    cloud_split = 'training'
            else:
                cloud_split = 'test'
                if self._check_not_required(cloud_split, file_path):
                    continue

            # Name of the input files
            KDTree_file = os.path.join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = os.path.join(tree_path, '{:s}.ply'.format(cloud_name))

            # Check if inputs have already been computed
            if os.path.isfile(KDTree_file):

                # read ply with data
                data = read_ply(sub_ply_file)
                sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
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
                points = np.vstack((data['x'], data['y'], data['z'])).T
                colors = np.vstack((data['red'], data['green'], data['blue'])).T
                if cloud_split == 'test':
                    int_features = None
                else:
                    int_features = data['class']

                # Subsample cloud
                sub_data = grid_subsampling(points,
                                            features=colors,
                                            labels=int_features,
                                            sampleDl=subsampling_parameter)

                # Rescale float color and squeeze label
                sub_colors = sub_data[1] / 255

                # Get chosen neighborhoods
                search_tree = KDTree(sub_data[0], leaf_size=50)

                # Save KDTree
                with open(KDTree_file, 'wb') as f:
                    pickle.dump(search_tree, f)

                # Save ply
                if cloud_split == 'test':
                    sub_labels = None
                    write_ply(sub_ply_file,
                              [sub_data[0], sub_colors],
                              ['x', 'y', 'z', 'red', 'green', 'blue'])
                else:
                    sub_labels = np.squeeze(sub_data[2])
                    write_ply(sub_ply_file,
                              [sub_data[0], sub_colors, sub_labels],
                              ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

            # Fill data containers
            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            if cloud_split in ['training', 'validation']:
                self.input_labels[cloud_split] += [sub_labels]

            if verbose and sys.stdout.isatty():
                print('', end='\r')
                print(fmt_str.format('#' * ((i * progress_n) // N), 100 * i / N), end='', flush=True)

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
        N = self.num_validation + self.num_test
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
            if file_path in self.train_files and self.all_splits[i] == self.validation_split:
                proj_file = os.path.join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                if os.path.isfile(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds, labels = pickle.load(f)
                else:
                    if verbose:
                        print('proj - val ', file_path)

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
            if file_path in self.test_files:
                if self._check_not_required(cloud_split, file_path):
                    i_test += 1
                    continue

                proj_file = os.path.join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                if os.path.isfile(proj_file):  # built
                    with open(proj_file, 'rb') as f:
                        proj_inds, labels = pickle.load(f)
                else:
                    if verbose:
                        print('proj - test', file_path)
                    # Get original points
                    full_ply_path = file_path
                    # full_ply_path = file_path.split(os.sep)
                    # full_ply_path = os.sep.join(full_ply_path)
                    data = read_ply(full_ply_path)
                    points = np.vstack((data['x'], data['y'], data['z'])).T
                    labels = np.zeros(points.shape[0], dtype=np.int32)

                    # Compute projection inds
                    proj_inds = np.squeeze(self.input_trees['test'][i_test].query(points, return_distance=False))
                    proj_inds = proj_inds.astype(np.int32)

                    # Save
                    with open(proj_file, 'wb') as f:
                        pickle.dump([proj_inds, labels], f)

                self.test_proj += [proj_inds]
                self.test_labels += [labels]
                i_test += 1

        self.val_proportions_full = np.array([np.sum([np.sum(labels == label_val) for labels in self.validation_labels]) for label_val in self.label_values])
        self.val_proportions = np.array([p for l, p in zip(self.label_values, self.val_proportions_full) if l not in self.ignored_labels])

        if verbose and sys.stdout.isatty():
            print('', end='\r')
            print(fmt_str.format('#' * (((i_val + i_test) * progress_n) // N), 100 * (i_val + i_test) / N), end='', flush=True)
        if verbose:
            print('\n')

        return

    # Utility methods
    # ------------------------------------------------------------------------------------------------------------------

    def get_batch_gen(self, split, config):
        """
        A function defining the batch generator for each split. Should return the generator, the generated types and
        generated shapes
        :param split: string in "training", "validation" or "test"
        :param config: configuration file
        :return: gen_func, gen_types, gen_shapes
        """

        config = self.config
        epoch_n = self.init_sampling(split)
        data_split = split

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
                input_colors = self.input_colors[data_split][cloud_ind][input_inds]
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
        gen_shapes = ([None, 3], [None, 6], [None], [None], [None], [None])

        return gen_func, gen_types, gen_shapes

    def get_tf_mapping_radius(self):

        # Returned mapping function
        def tf_map(stacked_points, stacked_colors, point_labels, stacks_lengths, point_inds, cloud_inds):
            """
            Args:
                stacked_points  : [BxN, 3]  - xyz in sample
                stacked_colors  : [BxN, 6]  - rgb,  xyz in whole cloud (global xyz)
                point_labels    : [BxN]     - labels
                stacks_lengths  : [B]    - size (point num) of each batch
                point_inds      : [B, N] - point idx in each of its point cloud
                cloud_inds      : [B]    - cloud idx
            """
            config = self.config

            # Get batch indice for each point
            batch_inds = self.tf_get_batch_inds(stacks_lengths)  # [BxN]

            # Augmentation: input points (xyz) - rotate, scale (flip), jitter
            stacked_points, scales, rots = self.tf_augment_input(stacked_points, batch_inds)

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
                s = tf.cast(tf.less(tf.random_uniform((num_batches,)), config.augment_color), tf.float32)
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
            elif in_features == '1rgbZ':  # used in provided cfgs - [1, rgb, z]
                stacked_features = tf.concat((stacked_features, stacked_colors, stacked_original_coordinates[..., 2:]), axis=1)
            elif in_features == '1rgbxyz':
                stacked_features = tf.concat((stacked_features, stacked_colors, stacked_points), axis=1)
            elif in_features == '1rgbxyzZ':
                stacked_features = tf.concat([stacked_features, stacked_colors, stacked_points, stacked_original_coordinates[..., 2:]], axis=1)
            else:
                raise ValueError('Only accepted input dimensions are 1, 3, 4 and 7 (without and with rgb/xyz)')

            # Get the whole input list
            inputs = self.tf_segmentation_inputs_radius(stacked_points,
                                                        stacked_features,
                                                        point_labels,
                                                        stacks_lengths,
                                                        batch_inds)
            if isinstance(inputs, dict):
                inputs.update({
                    'rgb': stacked_rgb,
                    'augment_scales': scales, 'augment_rotations': rots,
                    'point_inds': point_inds, 'cloud_inds': cloud_inds,
                })
            else:
                # Add scale and rotation for testing
                inputs += [scales, rots]
                inputs += [point_inds, cloud_inds]

            return inputs

        return tf_map

    def get_tf_mapping_fixed_size(self):

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
            config = self.config

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

                    write_ply('Semantic3D_input_{:d}.ply'.format(b_i),
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
        file = os.path.join(path, 'all_debug_inputs.pkl')
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

