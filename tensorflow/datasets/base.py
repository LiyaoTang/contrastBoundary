import os
import sys
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

OPS_DIR = os.path.join(ROOT_DIR, 'ops')
sys.path.insert(0, OPS_DIR)

from ops import get_tf_func

class Dataset(object):
    def __init__(self, config):
        self.config = config

        # path
        self.data_path = config.data_path if config.data_path else 'Data'
        self.data_path = f'{self.data_path}/{config.dataset}'
        # interface - init op
        self.train_init_op = None
        self.val_init_op = None
        self.test_init_op = None

    @property
    def info(self):
        return {
            'ignored_labels': self.ignored_labels,
            'label_names': self.label_names,
        }

    def valid_split(self, split, short=False):
        assert split in ['train', 'training', 'val', 'validation', 'test'], f'invalid split = {split}'
        if split.startswith('train'):
            return 'train' if short else 'training'
        elif split.startswith('val'):
            return 'val' if short else 'validation'
        else:
            return 'test'

    def init_labels(self):
        """
        Initiate all label parameters given the label_to_names dict
        """
        self.num_classes = len(self.label_to_names) - len(self.ignored_labels)
        assert self.config.num_classes == self.num_classes

        self.label_values = np.sort([k for k, v in self.label_to_names.items()])  # may not be consecutive or start from 0
        self.label_names = [self.label_to_names[k] for k in self.label_values]
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}

        # original label value <-> idx of valid label
        self.label_to_idx = []
        idx = 0
        for l in self.label_values:
            while len(self.label_to_idx) < l:
                self.label_to_idx += [None]  # skipped labels - not even invalid i.e. should not exists in label idx
            self.label_to_idx += [idx] if l not in self.ignored_labels else [-1]
            idx += l not in self.ignored_labels
        self.label_to_idx = np.array(self.label_to_idx)
        self.idx_to_label = np.array([l for l in self.label_values if l not in self.ignored_labels])

    def initialize(self, verbose=True):
        config = self.config
        # initialize op
        if config.search == 'radius':
            self.initialize_radius(verbose=verbose)
        elif config.search == 'knn':
            self.initialize_fixed_size(verbose=verbose)
        else:
            raise NotImplementedError(f'not supported methods: sampling = {config.sample}; searching = {config.search}')

    def initialize_radius(self, verbose=True):
        config = self.config
        self.batch_limit = self.calibrate_batches('training', config.batch_size)  # max num points [BxN] of a batch - used in get_batch_gen
        self.batch_limit_val = self.calibrate_batches('validation', config.batch_size_val) if config.batch_size_val else None
        # neighbor_limits - used in base.big_neighborhood_filter => set neighbor_idx shape
        self.neighborhood_limits = config.neighborhood_limits if config.neighborhood_limits else self.calibrate_neighbors('training')
        if config.max_neighborhood_limits:
            self.neighborhood_limits = [min(i, config.max_neighborhood_limits) for i in self.neighborhood_limits]
        self.neighborhood_limits = [int(l * config.density_parameter // 5) for l in self.neighborhood_limits]
        if verbose:
            print("batch_limit: ", self.batch_limit)
            print("neighborhood_limits: ", self.neighborhood_limits)

        # Get generator and mapping function
        gen_function, gen_types, gen_shapes = self.get_batch_gen_radius('training')
        gen_function_val, _, _ = self.get_batch_gen_radius('validation')
        gen_function_test, _, _ = self.get_batch_gen_radius('test')
        kwargs = gen_function.kwargs if hasattr(gen_function, 'kwargs') else {}
        map_func = self.get_tf_mapping_radius(**kwargs)

        self.train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes)
        self.train_data = self.train_data.map(map_func=map_func, num_parallel_calls=self.num_threads)
        # self.train_data = self.train_data.apply(tf.data.experimental.copy_to_device('/gpu:0'))
        self.train_data = self.train_data.prefetch(tf.data.experimental.AUTOTUNE)
        # self.train_data = self.train_data.apply(tf.data.experimental.prefetch_to_device('/gpu:0'))

        self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)
        self.val_data = self.val_data.map(map_func=map_func, num_parallel_calls=self.num_threads)
        self.val_data = self.val_data.prefetch(tf.data.experimental.AUTOTUNE)

        self.test_data = None
        if gen_function_test is not None:
            self.test_data = tf.data.Dataset.from_generator(gen_function_test, gen_types, gen_shapes)
            self.test_data = self.test_data.map(map_func=map_func, num_parallel_calls=self.num_threads)
            self.test_data = self.test_data.prefetch(tf.data.experimental.AUTOTUNE)

        # create a iterator of the correct shape and type
        iter = tf.data.Iterator.from_structure(self.train_data.output_types, self.train_data.output_shapes)
        # independent stream for each gpus
        self.flat_inputs = [iter.get_next() for i in range(config.gpu_num)]
        # create the initialisation operations
        self.train_init_op = iter.make_initializer(self.train_data)
        self.val_init_op = iter.make_initializer(self.val_data)
        self.test_init_op = iter.make_initializer(self.test_data) if self.test_data is not None else None

    def initialize_fixed_size(self, verbose=True):
        config = self.config
        if verbose:
            print('\n\t'.join(['k-nn & ratio:'] + [f'{a} = {getattr(config, a)}' for a in ['kr_search', 'kr_sample', 'kr_sample_up', 'r_sample']]))

        # Get generator and mapping function
        gen_function, gen_types, gen_shapes = self.get_batch_gen_fixed_size('training')
        gen_function_val, _, _ = self.get_batch_gen_fixed_size('validation')
        gen_function_test, _, _ = self.get_batch_gen_fixed_size('test')
        map_func = self.get_tf_mapping_fixed_size()

        self.train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes)
        self.train_data = self.train_data.batch(config.batch_size)
        self.train_data = self.train_data.map(map_func=map_func, num_parallel_calls=self.num_threads)
        self.train_data = self.train_data.prefetch(tf.data.experimental.AUTOTUNE)

        self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)
        self.val_data = self.val_data.batch(config.batch_size_val if config.batch_size_val else config.batch_size)
        self.val_data = self.val_data.map(map_func=map_func, num_parallel_calls=self.num_threads)
        self.val_data = self.val_data.prefetch(tf.data.experimental.AUTOTUNE)

        self.test_data = None
        if gen_function_test is not None:
            self.test_data = tf.data.Dataset.from_generator(gen_function_test, gen_types, gen_shapes)
            self.test_data = self.test_data.batch(config.batch_size_val if config.batch_size_val else config.batch_size)
            self.test_data = self.test_data.map(map_func=map_func, num_parallel_calls=self.num_threads)
            self.test_data = self.test_data.prefetch(tf.data.experimental.AUTOTUNE)

        # create a iterator of the correct shape and type
        iter = tf.data.Iterator.from_structure(self.train_data.output_types, self.train_data.output_shapes)
        # independent stream for each gpus
        self.flat_inputs = [iter.get_next() for i in range(config.gpu_num)]
        # create the initialisation operations
        self.train_init_op = iter.make_initializer(self.train_data)
        self.val_init_op = iter.make_initializer(self.val_data)
        self.test_init_op = iter.make_initializer(self.test_data) if self.test_data is not None else None


    def calibrate_batches(self, split=None, batch_size=None):
        s = 'training' if len(self.input_trees['training']) > 0 else 'test'
        split = split if split else s
        batch_size = batch_size if batch_size else self.config.batch_size

        N = (10000 // len(self.input_trees[split])) + 1
        sizes = []
        # Take a bunch of example neighborhoods in all clouds
        for i, tree in enumerate(self.input_trees[split]):
            # Randomly pick points
            points = np.array(tree.data, copy=False)
            rand_inds = np.random.choice(points.shape[0], size=N, replace=False)
            rand_points = points[rand_inds]
            noise = np.random.normal(scale=self.config.in_radius / 4, size=rand_points.shape)
            rand_points += noise.astype(rand_points.dtype)
            neighbors = tree.query_radius(points[rand_inds], r=self.config.in_radius)
            # Only save neighbors lengths
            sizes += [len(neighb) for neighb in neighbors]
        sizes = np.sort(sizes)
        # Higher bound for batch limit
        lim = sizes[-1] * batch_size
        # Biggest batch size with this limit
        sum_s = 0
        max_b = 0
        for i, s in enumerate(sizes):
            sum_s += s
            if sum_s > lim:
                max_b = i
                break
        # With a proportional corrector, find batch limit which gets the wanted batch_num
        estim_b = 0
        for i in range(10000):
            # Compute a random batch
            rand_shapes = np.random.choice(sizes, size=max_b, replace=False)
            b = np.sum(np.cumsum(rand_shapes) < lim)
            # Update estim_b (low pass filter istead of real mean
            estim_b += (b - estim_b) / min(i + 1, 100)
            # Correct batch limit
            lim += 10.0 * (self.config.batch_size - estim_b)
        return lim

    def calibrate_neighbors(self, split, keep_ratio=0.8, samples_threshold=10000):

        # Create a tensorflow input pipeline
        # **********************************
        import time
        config = self.config
        assert split in ['training', 'test']

        # From config parameter, compute higher bound of neighbors number in a neighborhood
        hist_n = int(np.ceil(4 / 3 * np.pi * (config.density_parameter + 1) ** 3))

        # Initiate neighbors limit with higher bound
        self.neighborhood_limits = np.full(config.num_layers, hist_n, dtype=np.int32)

        # Init batch limit if not done
        self.batch_limit = self.batch_limit if hasattr(self, 'batch_limit') else self.calibrate_batches()

        # Get mapping function
        gen_function, gen_types, gen_shapes = self.get_batch_gen_radius(split)
        kwargs = gen_function.kwargs if hasattr(gen_function, 'kwargs') else {}
        map_func = self.get_tf_mapping_radius(**kwargs)

        # Create batched dataset from generator
        train_data = tf.data.Dataset.from_generator(gen_function,
                                                    gen_types,
                                                    gen_shapes)

        train_data = train_data.map(map_func=map_func, num_parallel_calls=self.num_threads)

        # Prefetch data
        train_data = train_data.prefetch(10)

        # create a iterator of the correct shape and type
        iter = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
        flat_inputs = iter.get_next()

        # create the initialisation operations
        train_init_op = iter.make_initializer(train_data)

        # Create a local session for the calibration.
        cProto = tf.ConfigProto()
        cProto.gpu_options.allow_growth = True
        with tf.Session(config=cProto) as sess:

            # Init variables
            sess.run(tf.global_variables_initializer())

            # Initialise iterator with train data
            sess.run(train_init_op)

            # Get histogram of neighborhood sizes in 1 epoch max
            # **************************************************

            neighb_hists = np.zeros((config.num_layers, hist_n), dtype=np.int32)
            t0 = time.time()
            mean_dt = np.zeros(2)
            last_display = t0
            epoch = 0
            training_step = 0
            while epoch < 1 and np.min(np.sum(neighb_hists, axis=1)) < samples_threshold:
                try:

                    # Get next inputs
                    t = [time.time()]
                    ops = flat_inputs['neighbors']
                    neighbors = sess.run(ops)
                    t += [time.time()]

                    # Update histogram
                    counts = [np.sum(neighb_mat < neighb_mat.shape[0], axis=1) for neighb_mat in neighbors]
                    hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
                    neighb_hists += np.vstack(hists)
                    t += [time.time()]

                    # Average timing
                    mean_dt = 0.01 * mean_dt + 0.99 * (np.array(t[1:]) - np.array(t[:-1]))

                    # Console display
                    if (t[-1] - last_display) > 2.0:
                        last_display = t[-1]
                        message = 'Calib Neighbors {:08d} : timings {:4.2f} {:4.2f}'
                        print(message.format(training_step, 1000 * mean_dt[0], 1000 * mean_dt[1]))

                    training_step += 1

                except tf.errors.OutOfRangeError:
                    print('End of train dataset')
                    epoch += 1

            cumsum = np.cumsum(neighb_hists.T, axis=0)
            percentiles = np.sum(cumsum < (keep_ratio * cumsum[hist_n - 1, :]), axis=0)

            self.neighborhood_limits = percentiles
            print('neighborhood_limits : {}'.format(self.neighborhood_limits))

        return


    def init_sampling(self, split):
        ############
        # Parameters
        ############

        # Initiate parameters depending on the chosen split
        if split == 'training':  # First compute the number of point we want to pick in each cloud set - num of samples
            epoch_n = self.config.epoch_steps * self.config.epoch_batch
        elif split == 'validation':
            epoch_n = self.config.validation_steps * self.config.epoch_batch
        elif split == 'test':
            epoch_n = self.config.validation_steps * self.config.epoch_batch
        elif split == 'ERF':
            # First compute the number of point we want to pick in each cloud and for each class
            epoch_n = 1000000
            self.batch_limit = 1  # BxN = 1, single point
            np.random.seed(42)
            split = 'test'
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
        :param split: string in "training", "validation" or "test"
        :param config: configuration file
        :return: gen_func, gen_types, gen_shapes
        """

        config = self.config
        epoch_n = self.init_sampling(split)
        data_split = split
        batch_limit = self.batch_limit
        if split != 'training' and self.batch_limit_val:
            batch_limit = self.batch_limit_val

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

                    # Safe check for very dense areas - align with training setting
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
                    input_labels = self.label_to_idx[input_labels]
                    # input_labels = np.array([self.label_to_idx[l] for l in input_labels])

                # In case batch is full, yield it and reset it
                if batch_n + n > batch_limit and batch_n > 0:
                    yield (np.concatenate(p_list, axis=0),      # [BxN, 3]  - xyz in sample
                           np.concatenate(c_list, axis=0),      # [BxN, 3/1 + 3 (RGB/intensity + global xyz in whole cloud)]
                           np.concatenate(pl_list, axis=0),     # [BxN]     - labels
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
        spatially_regular_gen.types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32)
        spatially_regular_gen.shapes = ([None, 3], [None, 6], [None], [None], [None], [None])

        def _calc_tukeys(pick_point, queried_points, in_radius=config.in_radius): 
            # calc tukeys weights
            dists = np.sum(np.square((queried_points - pick_point).astype(np.float32)), axis=1)
            tukeys = np.square(1 - dists / np.square(in_radius))
            tukeys[dists > np.square(in_radius)] = 0
            return tukeys


        def _batch_block(in_radius, in_radius_super, num_sample):
            batch_n = 0
            cloud_ind = int(np.argmin(self.min_potentials[split]))  # Choose a random cloud (the one that has the point with minimum potentials)
            point_ind = np.argmin(self.potentials[split][cloud_ind])  # Choose point ind as minimum of potentials - center point

            points = np.array(self.input_trees[data_split][cloud_ind].data, copy=False)  # Get points from tree structure
            center_point = points[point_ind, :].reshape(1, -1)  # Center point of the block

            if split != 'ERF':  # Add noise to the center point
                noise = np.random.normal(scale=in_radius_super/10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)
            else:
                pick_point = center_point

            input_inds = self.input_trees[data_split][cloud_ind].query_radius(pick_point, r=in_radius_super)[0]  # Indices of all points for current batch
            input_points = points[input_inds]

            # Split input sphere into samples => potential-based sampling inside sphere
            pick_list = []
            inds_list = []
            # local_tree = KDTree(input_points)
            # local_potentials = np.random.rand(input_points.shape[0]) * 1e-3
            local_potentials = np.array(self.potentials[split][cloud_ind][input_inds], copy=True)  # Local potential for the center of each sample - detached from global potentials
            for i in range(num_sample):
                c = input_points[int(np.argmin(local_potentials)), ...].reshape(1, -1)
                if split != 'ERF':  # add noise
                    c += np.random.normal(scale=in_radius/10, size=c.shape).astype(c.dtype)
                # local_inds = local_tree.query_radius(c, r=in_radius)[0]  # Indices in local input sphere
                pts_inds = self.input_trees[data_split][cloud_ind].query_radius(c, r=in_radius)[0]  # Indices in global area
                joint_inds, _, local_inds = np.intersect1d(pts_inds, input_inds, assume_unique=True, return_indices=True)  # intersection
                pts_n = pts_inds.shape[0]

                if batch_n + pts_n > self.batch_limit and batch_n > 0:
                    break
                pick_list.append(c)
                inds_list.append(pts_inds)
                batch_n += pts_n

                if split != 'ERF':  # Tuckeys weights
                    local_tukeys = _calc_tukeys(c, input_points[local_inds], in_radius)
                    local_potentials[local_inds] += local_tukeys
                    tukeys = _calc_tukeys(c, points[pts_inds], in_radius)
                    self.potentials[split][cloud_ind][pts_inds] += tukeys
                    if pts_n > self.batch_limit:
                        # Safe check for very dense areas - align with training setting
                        pts_inds = np.random.choice(pts_inds, size=int(self.batch_limit), replace=False)
                        pts_n = pts_inds.shape[0]
                # print(len(joint_inds), len(local_inds))
                # local_potentials += 1
                # print(local_potentials)
                # print(self.potentials[split][cloud_ind][input_inds])
                # input()

            self.min_potentials[split][cloud_ind] = float(np.min(self.potentials[split][cloud_ind]))
            return pick_list, inds_list, [cloud_ind] * len(pick_list)

        def _batch_rand(in_radius):
            batch_n = 0
            pick_list = []
            inds_list = []
            ci_list = []

            while True:
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
                input_inds = self.input_trees[data_split][cloud_ind].query_radius(pick_point, r=config.in_radius)[0]
                # Number collected
                n = input_inds.shape[0]

                if split != 'ERF':
                    # Safe check for very dense areas - align with training setting
                    if n > self.batch_limit:
                        input_inds = np.random.choice(input_inds, size=int(self.batch_limit)-1, replace=False)
                        n = input_inds.shape[0]

                if batch_n + n > batch_limit and batch_n > 0:
                    # leave current sample for next batch (not changing potential)
                    break

                batch_n += n
                pick_list += [pick_point]
                inds_list += [input_inds]
                ci_list += [cloud_ind]
                # Update potentials (Tuckey weights)
                if split != 'ERF':
                    dists = np.sum(np.square((points[input_inds] - pick_point).astype(np.float32)), axis=1)
                    tukeys = np.square(1 - dists / np.square(config.in_radius))
                    tukeys[dists > np.square(config.in_radius)] = 0
                    self.potentials[split][cloud_ind][input_inds] += tukeys
                    self.min_potentials[split][cloud_ind] = float(np.min(self.potentials[split][cloud_ind]))

            return pick_list, inds_list, ci_list

        def spatially_regular_block_gen():

            batch_size = config.batch_size  # always use training batch size
            in_bound = config.in_bound if config.in_bound else 1
            in_radius = config.in_radius
            in_radius_super = in_radius * np.cbrt(batch_size * in_bound)  # < 1 to shrink the super shpere, > 1 to enlarge

            block_ratio = float(config.block_ratio) if config.block_ratio else 1  # probs to use block
            epoch_n = int(np.ceil(config.epoch_steps * config.epoch_batch / config.batch_size))  # one-round per-batch
            for i in range(epoch_n):
                use_block = np.random.uniform()
                if block_ratio == 1 or use_block < block_ratio:
                    # print(use_block, block_ratio, 'block', flush=True)
                    pick_list, inds_list, ci_list = _batch_block(in_radius, in_radius_super, batch_size)
                else:
                    # print(use_block, block_ratio, 'rand', flush=True)
                    pick_list, inds_list, ci_list = _batch_rand(in_radius)

                # Collect points and colors
                p_list = []
                c_list = []
                pl_list = []
                for c, inds, cloud_ind in zip(pick_list, inds_list, ci_list):
                    points = np.array(self.input_trees[data_split][cloud_ind].data, copy=False)
                    input_points = (points[inds] - c).astype(np.float32)  # local centering for each sample
                    input_colors = self.input_colors[data_split][cloud_ind][inds]
                    if split in ['test', 'ERF']:
                        input_labels = np.zeros(input_points.shape[0])
                    else:
                        input_labels = self.input_labels[data_split][cloud_ind][inds]
                        input_labels = self.label_to_idx[input_labels]
                        # input_labels = np.array([self.label_to_idx[l] for l in input_labels])

                    # Add data to current batch
                    p_list += [input_points]
                    c_list += [np.hstack((input_colors, points[inds]))]
                    pl_list += [input_labels]

                yield_data = [
                    np.concatenate(p_list, axis=0),
                    np.concatenate(c_list, axis=0),
                    np.concatenate(pl_list, axis=0),
                    np.array([tp.shape[0] for tp in p_list]),
                    np.concatenate(inds_list, axis=0),  # [BxN]
                    np.array(ci_list, dtype=np.int32),
                ]

                # In case batch is full, yield it and reset it
                yield tuple(yield_data)
        spatially_regular_block_gen.kwargs = {}
        spatially_regular_block_gen.types = [tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.int32]
        spatially_regular_block_gen.shapes = [[None, 3], [None, 6], [None], [None], [None], [None]]
        # if config.overlap_gen and config.overlap_gen == 'npgen':
        #     raise ValueError('very slow ops - consider tfmap/tfwhile')
        #     spatially_regular_block_gen.kwargs.update({'k_list': ['overlap_idx', 'overlap_cnt']})  # kwargs for get_tf_mapping
        #     spatially_regular_block_gen.types += [(tf.int32, tf.int32)]
        #     spatially_regular_block_gen.shapes += [([None], [None])]

        def spatially_regular_mp_gen():
            # init batch data
            p_list = []
            c_list = []
            pl_list = []
            pi_list = []
            ci_list = []
            batch_n = 0

            queue_list = self.data_queues[split]
            cloud_samples = [q.get() for q in queue_list]
            for _ in range(epoch_n):

                # Find sample of minimal potentials
                cloud_ind = np.argmin([s[-1] for s in cloud_samples])
                input_points, input_colors, input_labels, input_inds, potentials = cloud_samples[cloud_ind]
                self.min_potentials[split][cloud_ind] = potentials  # record the min-potentials of each cloud
                n = input_points.shape[0]

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
                p_list += [input_points]
                c_list += [input_colors]
                pl_list += [input_labels]
                pi_list += [input_inds]
                ci_list += [cloud_ind]
                # Update batch size
                batch_n += n

                # Get next sample from the cloud - potentially blocking
                cloud_samples[cloud_ind] = queue_list[cloud_ind].get()

            if batch_n > 0:
                yield (np.concatenate(p_list, axis=0),
                       np.concatenate(c_list, axis=0),
                       np.concatenate(pl_list, axis=0),
                       np.array([tp.shape[0] for tp in p_list]),
                       np.concatenate(pi_list, axis=0),
                       np.array(ci_list, dtype=np.int32))
        spatially_regular_mp_gen.init = self.init_mp_sampling
        spatially_regular_mp_gen.types = spatially_regular_gen.types
        spatially_regular_mp_gen.shapes = spatially_regular_gen.shapes

        def spatially_regular_thread_gen():
            spatially_regular_mp_gen()
        spatially_regular_thread_gen.init = lambda *args: self.init_mp_sampling(*args, mp_type='thread')
        spatially_regular_thread_gen.types = spatially_regular_mp_gen.types
        spatially_regular_thread_gen.shapes = spatially_regular_mp_gen.shapes

        # Define the generator that should be used for this split
        gen_func = config.data_gen if config.data_gen else 'spatially_regular_gen'
        gen_func = {
            'spatially_regular_gen': spatially_regular_gen,
            'spatially_regular_block_gen': spatially_regular_block_gen,
            'spatially_regular_mp_gen': spatially_regular_mp_gen,
            'spatially_regular_thread_gen': spatially_regular_thread_gen,
        }[gen_func]
        gen_types = tuple(gen_func.types)
        gen_shapes = tuple(gen_func.shapes)
        if hasattr(gen_func, 'init'):  # extra init
            gen_func.init(split)

        return gen_func, gen_types, gen_shapes

    def get_batch_gen_fixed_size(self, split):

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
                dists, input_inds = self.input_trees[split][cloud_ind].query(pick_point, k=k)
                input_inds = input_inds[0]

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
                # TODO: using dist from tree query ???
                # assert np.all(np.abs(dists ** 2 - np.sum(np.square((points[input_inds] - pick_point).astype(np.float32)), axis=1)) < 1e-9)
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


    def tf_augment_input(self, stacked_points, batch_inds):
        """
        Augment inputs with rotation, scale and noise
        Args:
            batch_inds : [BxN] - batch idx for each point
        """
        # Parameter
        config = self.config
        num_batches = batch_inds[-1] + 1

        ##########
        # Rotation
        ##########
        if config.augment_rotation == 'none':
            R = tf.eye(3, batch_shape=(num_batches,))  # [BxN, 3, 3]
        elif config.augment_rotation == 'vertical':  # -- used in default cfgs
            # Choose a random angle for each element
            theta = tf.random.uniform((num_batches,), minval=0, maxval=2 * np.pi)
            # Rotation matrices
            c, s = tf.cos(theta), tf.sin(theta)
            cs0 = tf.zeros_like(c)
            cs1 = tf.ones_like(c)
            R = tf.stack([c, -s, cs0, s, c, cs0, cs0, cs0, cs1], axis=1)
            R = tf.reshape(R, (-1, 3, 3))  # [B, 3, 3]
            # Create N x 3 x 3 rotation matrices to multiply with stacked_points
            stacked_rots = tf.gather(R, batch_inds)  # [BxN, 3, 3]
            # Apply rotations
            if len(stacked_rots.shape) == len(stacked_points.shape):
                stacked_rots = tf.expand_dims(stacked_rots, axis=-3)  # [BxN, 1, 3, 3] to match [B, N, 1, 3]
            stacked_points = tf.reshape(tf.matmul(tf.expand_dims(stacked_points, axis=-2),  # to row vec: [BxN, 3] -> [BxN, 1, 3]
                                                  stacked_rots),
                                        tf.shape(stacked_points))
        elif config.augment_rotation == 'arbitrarily':
            cs0 = tf.zeros((num_batches,))
            cs1 = tf.ones((num_batches,))
            # x rotation
            thetax = tf.random.uniform((num_batches,), minval=0, maxval=2 * np.pi)
            cx, sx = tf.cos(thetax), tf.sin(thetax)
            Rx = tf.stack([cs1, cs0, cs0, cs0, cx, -sx, cs0, sx, cx], axis=1)
            Rx = tf.reshape(Rx, (-1, 3, 3))
            # y rotation
            thetay = tf.random.uniform((num_batches,), minval=0, maxval=2 * np.pi)
            cy, sy = tf.cos(thetay), tf.sin(thetay)
            Ry = tf.stack([cy, cs0, -sy, cs0, cs1, cs0, sy, cs0, cy], axis=1)
            Ry = tf.reshape(Ry, (-1, 3, 3))
            # z rotation
            thetaz = tf.random.uniform((num_batches,), minval=0, maxval=2 * np.pi)
            cz, sz = tf.cos(thetaz), tf.sin(thetaz)
            Rz = tf.stack([cz, -sz, cs0, sz, cz, cs0, cs0, cs0, cs1], axis=1)
            Rz = tf.reshape(Rz, (-1, 3, 3))
            # whole rotation
            Rxy = tf.matmul(Rx, Ry)
            R = tf.matmul(Rxy, Rz)
            # Create N x 3 x 3 rotation matrices to multiply with stacked_points
            stacked_rots = tf.gather(R, batch_inds)
            # Apply rotations
            if len(stacked_rots.shape) < len(stacked_points.shape):
                stacked_rots = tf.expand_dims(stacked_rots, axis=-3)  # [B, 1, 3, 3] to match [B, N, 1, 3]
            stacked_points = tf.reshape(tf.matmul(tf.expand_dims(stacked_points, axis=-2), stacked_rots), tf.shape(stacked_points))
        else:
            raise ValueError('Unknown rotation augmentation : ' + self.augment_rotation)

        #######
        # Scale
        #######
        # Choose random scales for each example
        min_s = config.augment_scale_min
        max_s = config.augment_scale_max
        if config.augment_scale_anisotropic:  # each batch a scale - [B, 3/1]
            s = tf.random.uniform((num_batches, 3), minval=min_s, maxval=max_s)  # xyz diff scale 
        else:
            s = tf.random.uniform((num_batches, 1), minval=min_s, maxval=max_s)  # xyz same scale
        symmetries = []
        for i in range(3):
            if config.augment_symmetries[i]:  # could flip (multiply by 1/-1)
                symmetries.append(tf.round(tf.random.uniform((num_batches, 1))) * 2 - 1)
            else:
                symmetries.append(tf.ones([num_batches, 1], dtype=tf.float32))
        s *= tf.concat(symmetries, 1)  # [B, 3]
        # Create N x 3 vector of scales to multiply with stacked_points
        stacked_scales = tf.gather(s, batch_inds)  # [BxN, 3]
        # Apply scales
        if len(stacked_scales.shape) < len(stacked_points.shape):
            stacked_scales = tf.expand_dims(stacked_scales, axis=-2)  # [B, 1, 3] to match [B, N, 3]
        stacked_points = stacked_points * stacked_scales

        #######
        # Noise
        #######
        noise = tf.random_normal(tf.shape(stacked_points), stddev=config.augment_noise)  # per-point noise
        stacked_points = stacked_points + noise
        return stacked_points, s, R

    def tf_get_batch_inds(self, stacks_len):
        """
        Method computing the batch indices of all points, given the batch element sizes (stack lengths). Example:
        From [3, 2, 5], it would return [0, 0, 0, 1, 1, 2, 2, 2, 2, 2]
        """

        # Initiate batch inds tensor
        num_batches = tf.shape(stacks_len)[0]
        num_points = tf.reduce_sum(stacks_len)
        batch_inds_0 = tf.zeros((num_points,), dtype=tf.int32)

        # Define body of the while loop
        def body(batch_i, point_i, b_inds):
            num_in = stacks_len[batch_i]
            num_before = tf.cond(tf.less(batch_i, 1),
                                 lambda: tf.zeros((), dtype=tf.int32),
                                 lambda: tf.reduce_sum(stacks_len[:batch_i]))
            num_after = tf.cond(tf.less(batch_i, num_batches - 1),
                                lambda: tf.reduce_sum(stacks_len[batch_i + 1:]),
                                lambda: tf.zeros((), dtype=tf.int32))

            # Update current element indices
            inds_before = tf.zeros((num_before,), dtype=tf.int32)
            inds_in = tf.fill((num_in,), batch_i)
            inds_after = tf.zeros((num_after,), dtype=tf.int32)
            n_inds = tf.concat([inds_before, inds_in, inds_after], axis=0)

            b_inds += n_inds

            # Update indices
            point_i += stacks_len[batch_i]
            batch_i += 1

            return batch_i, point_i, b_inds

        def cond(batch_i, point_i, b_inds):
            return tf.less(batch_i, tf.shape(stacks_len)[0])

        _, _, batch_inds = tf.while_loop(cond,
                                         body,
                                         loop_vars=[0, 0, batch_inds_0],
                                         shape_invariants=[tf.TensorShape([]), tf.TensorShape([]),
                                                           tf.TensorShape([None])])

        return batch_inds

    def tf_stack_batch_inds(self, stacks_len, tight=False):
        impl = self.config.tf_stack_batch_inds
        impl = impl if impl else 'while'
        impl = impl.replace('tf', '')
        return getattr(self, f'tf_stack_batch_inds_{impl}')(stacks_len, tight)

    def tf_stack_batch_inds_while(self, stacks_len, tight=False):
        """
        Stack the flat point idx, given the batch element sizes (stacks_len)
            E.g. stacks_len = [3, 2, 5]; n = sum(stacks_len) = 10
            => return: [[0, 1, 2, n, n, n], 
                        [3, 4, n, n, n, n],
                        [5, 6, 7, 8, 9, n]]
        """
        # Initiate batch inds tensor
        num_points = tf.reduce_sum(stacks_len)
        max_points = tf.reduce_max(stacks_len)
        batch_inds_0 = tf.zeros((0, max_points), dtype=tf.int32)

        # Define body of the while loop
        def body(batch_i, point_i, b_inds):
            # Create this element indices
            element_inds = tf.expand_dims(tf.range(point_i, point_i + stacks_len[batch_i]), axis=0)
            # Pad to right size
            padded_inds = tf.pad(element_inds,
                                 [[0, 0], [0, max_points - stacks_len[batch_i]]],
                                 "CONSTANT",
                                 constant_values=num_points)
            # Concatenate batch indices
            b_inds = tf.concat((b_inds, padded_inds), axis=0)
            # Update indices
            point_i += stacks_len[batch_i]
            batch_i += 1
            return batch_i, point_i, b_inds

        def cond(batch_i, point_i, b_inds):
            return tf.less(batch_i, tf.shape(stacks_len)[0])

        fixed_shapes = [tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([None, None])]
        _, _, batch_inds = tf.while_loop(cond,
                                         body,
                                         loop_vars=[0, 0, batch_inds_0],
                                         shape_invariants=fixed_shapes)

        # Add a last column with shadow neighbor if there is not
        def f1(): return tf.pad(batch_inds, [[0, 0], [0, 1]], "CONSTANT", constant_values=num_points)
        def f2(): return batch_inds
        if not tight:
            batch_inds = tf.cond(tf.equal(num_points, max_points * tf.shape(stacks_len)[0]), true_fn=f1, false_fn=f2)

        return batch_inds

    def tf_stack_batch_inds_map(self, stacks_len, tight=False):
        # Initiate batch inds tensor
        B_inds = tf.range(tf.shape(stacks_len)[0])  # [B]
        num_points = tf.reduce_sum(stacks_len)
        max_points = tf.reduce_max(stacks_len)
        if not tight:
            max_points += 1
        def flatten_idx(batch_i):
            cur_len = stacks_len[batch_i]
            start_i = tf.reduce_sum(stacks_len[:batch_i])
            element_inds = tf.range(start_i, start_i + cur_len)  # Create this element indices (starting at 0)
            padded_inds = tf.pad(element_inds, [[0, max_points - cur_len]], "CONSTANT", constant_values=num_points)  # [max_points] Pad to right size
            return padded_inds
        batch_inds = tf.map_fn(flatten_idx, B_inds, dtype=tf.int32)
        return batch_inds

    def big_neighborhood_filter(self, neighbors, layer):
        """
        Filter neighborhoods with max number of neighbors. Limit is set to keep XX% of the neighborhoods untouched.
        Limit is computed at initialization
        """
        # crop neighbors matrix
        neighbors = neighbors[:, :self.neighborhood_limits[layer]]
        # neighbors = tf.reshape(neighbors, [-1, self.neighborhood_limits[layer]])
        return neighbors


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
        for dt in range(0, downsample_times):  # downsample times
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
            'batches_len': tuple(input_batches_len),
            'features': stacked_features,
            'batch_weights': stacked_weights,
            'in_batches': stacked_batch_inds_0,
            'out_batches': stacked_batch_inds_1,
            'point_labels': point_labels,
        }

        return input_dict

    def tf_segmentation_inputs_fixed_size(self, points, features, point_labels):  # [B, N, 3], [B, N, d], [B, N]

        config = self.config
        assert config.sample in ['random', 'farthest'], f'not supported fixed-size sampling {self.config.sample}'
        assert config.search in ['knn'], f'not supported fixed-size neighbor searching {self.config.search}'
        sample_func = get_tf_func(config.sample, verbose=self.verbose)
        search_func = get_tf_func(config.search, verbose=self.verbose)

        num_layers = config.num_layers
        downsample_times = num_layers - 1

        # Lists of config
        k_search = config.kr_search if isinstance(config.kr_search, list) else [int(config.kr_search)] * num_layers  # k-nn for at each layer (stage)
        k_sample = config.kr_sample if isinstance(config.kr_sample, list) else [int(config.kr_sample)] * downsample_times  # k-nn for subsampling
        k_sample_up = config.kr_sample_up if isinstance(config.kr_sample_up, list) else [int(config.kr_sample_up)] * downsample_times  # k-nn for upsampling
        r_sample = config.r_sample if isinstance(config.r_sample, list) else [int(config.r_sample)] * downsample_times  # ratio for subsampling

        # Lists of inputs
        input_points = [None] * num_layers
        input_neighbors = [None] * num_layers
        input_pools = [None] * num_layers
        input_upsamples = [None] * num_layers
        input_batches_len = [None] * num_layers

        n_points = self.config.in_points  # N at each layer (stage)
        input_upsamples[0] = tf.zeros((0, 1), dtype=tf.int32)  # no upsample for input pt
        for dt in range(0, downsample_times):
            neighbors_inds = search_func(points, points, k_search[dt])
            pool_points = sample_func(points, n_points // r_sample[dt])
            # pool_points = tf.gather(points, down_inds, batch_dims=1)
            pool_inds = search_func(pool_points, points, k_sample[dt])
            up_inds = search_func(points, pool_points, k_sample_up[dt])

            input_points[dt] = points
            input_neighbors[dt] = neighbors_inds
            input_pools[dt] = pool_inds
            input_upsamples[dt + 1] = up_inds
            points = pool_points
            n_points = int(pool_points.shape[-2]) if isinstance(pool_points.shape[-2].value, int) else tf.shape(pool_points)[-2]

        # last (downsampled) layer points
        neighbors_inds = search_func(points, points, k_search[downsample_times])
        input_points[downsample_times] = points
        input_neighbors[downsample_times] = neighbors_inds
        input_pools[downsample_times] = tf.zeros((0, 1), dtype=tf.int32)

        # # Batch unstacking (with first layer indices for optional classif loss) - in_batches
        # stacked_batch_inds_0 = self.tf_stack_batch_inds(input_batches_len[0])
        # # Batch unstacking (with last layer indices for optional classif loss) - out_batches
        # stacked_batch_inds_1 = self.tf_stack_batch_inds(input_batches_len[-1])

        # list of network inputs
        input_dict = {
            'points': tuple(input_points),
            'neighbors': tuple(input_neighbors),
            'pools': tuple(input_pools),
            'upsamples': tuple(input_upsamples),
            # 'batches_len': tuple(input_batches_len),
            'features': features,
            # 'batch_weights': stacked_weights,
            # 'in_batches': stacked_batch_inds_0,
            # 'out_batches': stacked_batch_inds_1,
            'point_labels': point_labels,
        }

        return input_dict


    def get_class_cnt(self, split='train-val'):
        if hasattr(self, 'class_cnt'):
            return self.class_cnt
        class_cnt = np.zeros(self.num_classes)
        for s in split.split('-'):
            s = self.valid_split(s)
            for labels in self.input_labels[s]:
                idx, cnt = np.unique(labels, return_counts=True)
                idx = self.label_to_idx[idx].astype(int)
                mask = np.where(idx >= 0)
                idx = idx[mask]
                cnt = cnt[mask]
                class_cnt[idx] += cnt
        self.class_cnt = class_cnt
        return self.class_cnt
