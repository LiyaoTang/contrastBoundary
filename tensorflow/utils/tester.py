# Basic libs
import os, gc, re, sys, time, json
ROOT_DIR = os.path.abspath(os.path.join(__file__, '../', '../'))
sys.path.insert(0, ROOT_DIR)

import numpy as np
import tensorflow as tf
if tf.__version__.split('.')[0] == '2':
    tf = tf.compat.v1
    tf.disable_v2_behavior()

from sklearn.neighbors import KDTree
from collections import defaultdict

# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
from utils.logger import log_percentage, print_dict, print_mem
from utils.metrics import AverageMeter, metrics_from_confusions, metrics_from_result
from sklearn.metrics import confusion_matrix


class ModelTester:

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, config, verbose=True):
        self.config = config
        self.verbose = verbose

        self.save_extra = {}  # for saving with extra ops

        if config.dataset in ['S3DIS', 'ScanNet', 'Semantic3D', 'NPM3D']:
            self.val_running_vote = self.val_running_vote_seg
            self.val_vote = self.val_vote_seg
            self.test_vote = self.test_vote_seg
        elif config.dataset in ['ModelNet']:
            self.val_running_vote = self.val_running_vote_cls
            self.val_vote = self.val_vote_cls
            self.test_vote = self.test_vote_cls
        else:
            raise NotImplementedError(f'not supported dataset: {config.dataset}')

    def init_pointcloud_log(self, dataset, split, d, dtype=np.float32, init_fn=np.zeros):
        shape = lambda l: [l, d] if d else [l]  # d - size of last dimension => each point d-dim [N, d] (d = None to have [N])
        log = [init_fn(shape=shape(t.data.shape[0]), dtype=dtype) for t in dataset.input_trees[split]]
        return log

    def initialize(self, ops, dataset, model, split):
        # initialize cum_dict & ops
        config = self.config
        ncls = config.num_classes

        run_ops = {k: ops['result_dict'][k] for k in ['inputs', 'seg']}
        cum_dict = {
            'prob': self.init_pointcloud_log(dataset, split, ncls)
        }

        extra_ops = [k for k in config.extra_ops.split('-') if k]
        extra_ops_solved = extra_ops.copy()
        for k in extra_ops:
            if k in ['neighbor', 'boundary', 'prob', 'stat', 'conf']:
                continue
            elif k in ['center', 'edge']:  # ops on each input point sample
                extra_ops_solved.remove(k)
                if k in ops['result_dict']:
                    run_ops[k] = ops['result_dict'][k]
                cum_dict[k] = {}
            elif k == 'feature':  # main feature - head_dict['seg']['latent']
                dims = model.head_dict['result']['seg']['latent'].shape[-1]
                cum_dict[k] = self.init_pointcloud_log(dataset, split, d=dims)
            elif re.match('f[UD][a1-9]', k):  # feature upsampled from each stage - stage_list[stage][i]['latent'] ('f_out' if no latent)
                extra_ops_solved.remove(k)  # replace with concret feature
                assert config.gpu_num <= 1, f'gpu_num should <= 1, but is {config.gpu_num} - ops cannot created outside graph builder'
                from models.head import get_sample_idx
                from models.utils import parse_stage
                from models.basic_operators import tf_gather
                for n, i in parse_stage(k[1:], self.config.num_layers):
                    if f'{n}{i}' == 'up0': continue  # should specify 'feature'
                    k = f'feature-{n}-{i}'
                    f = model.stage_list[n][i]
                    # upsample to output stage to allow voting
                    ftype = 'latent' if 'latent' in f else f['f_out']
                    upsample_idx = get_sample_idx(model.inputs, f'{n}{i}', f'U0', ftype, config, kr=1)
                    f = tf_gather(f[ftype], upsample_idx, get_mask=False) if upsample_idx is not None else f[ftype]
                    run_ops[k] = [f]  # add to val_ops & cum_dict
                    cum_dict[k] = self.init_pointcloud_log(dataset, split, d=f.shape[-1])
                    extra_ops_solved.append(k)
            elif 'novote' in extra_ops:
                cum_dict['prob_novote'] = self.init_pointcloud_log(dataset, split, ncls)
            else:
                raise ValueError(f'not supported extra ops k = {k} from {config.extra_ops}')

        return run_ops, cum_dict, extra_ops_solved

    # Val methods
    # ------------------------------------------------------------------------------------------------------------------

    def val_running_vote_seg(self, sess, ops, dataset, model, validation_probs, epoch=1):
        """
        One epoch validating - running voting used during training, main task results only
        """

        val_smooth = 0.95  # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)

        result_dict = {k: ops['result_dict'][k] for k in ['inputs', 'seg']}  # result dict for seg
        val_ops = {'loss_dict': ops['loss_dict'], 'result_dict': result_dict}
        feed_dict = {ops['is_training']: False}

        # Initialise iterator
        sess.run(ops['val_init_op'])

        ep = 0
        loss_meter = {k: AverageMeter() for k in val_ops['loss_dict']} if 'loss_dict' in val_ops else{}
        cum_dict = {
            'conf': 0,  # conf from current validation
            'prob': validation_probs,  # accumulating probs
        }
        while ep < epoch:
            try:
                rst = sess.run(val_ops, feed_dict=feed_dict)

                loss_dict = rst['loss_dict'] if 'loss_dict' in rst else {}
                cur_rst = rst['result_dict']  # per-gpu result

                for k, v in loss_dict.items():
                    loss_meter[k].update(v)

                # Stack all validation predictions for each class separately - iterate over each gpu & cloud
                self.cumulate_probs(dataset, model, cur_rst, cum_dict, task='seg', smooth=val_smooth)

            except tf.errors.OutOfRangeError:
                ep += 1
                pass

        if loss_meter:
            print(f'val loss avg:', ' '.join([f'{loss_n} = {meter.avg:.3f}' for loss_n, meter in loss_meter.items()]))

        label_to_idx = dataset.label_to_idx
        proportions = dataset.val_proportions
        cur_m = metrics_from_confusions(cum_dict['conf'], proportions=proportions)  # use sampled pred-label of current epoch
        vote_m = metrics_from_result(validation_probs, dataset.input_labels['validation'], dataset.num_classes, label_to_idx=label_to_idx, proportions=proportions)  # use the accumulated per-point voting

        print(f'metrics - current     {cur_m}\n'
              f'        - accumulated {vote_m}', flush=True)
        return cur_m


    def val_vote_seg(self, sess, ops, dataset, model, num_votes=20):
        """
        Voting validating
        """

        feed_dict = {ops['is_training']: False}

        # Smoothing parameter for votes
        val_smooth = 0.95

        # Initialise iterator with val data
        sess.run(ops['val_init_op'])

        # Initiate global prediction over val clouds
        label_to_idx = dataset.label_to_idx
        proportions = dataset.val_proportions
        val_ops, cum_dict, extra_ops = self.initialize(ops, dataset, model, 'validation')
        val_probs = cum_dict['prob']

        vote_ind = 0
        last_min = -0.5
        if self.config.debug:
            print_dict(val_ops, head='val_vote_seg - val_ops')
        while last_min < num_votes:
            try:
                cur_rst = sess.run(val_ops, feed_dict=feed_dict)
                # Stack all validation predictions for each class separately - iterate over each gpu & cloud
                self.cumulate_probs(dataset, model, cur_rst, cum_dict, task='seg', smooth=val_smooth)

            except tf.errors.OutOfRangeError:
                new_min = np.min(dataset.min_potentials['validation'])
                if self.verbose:
                    print(f'Step {vote_ind:3d}, end. Min potential = {new_min:.1f}', flush=True)
                if last_min + 1 < new_min:
                    # Update last_min
                    last_min += 1

                    if self.verbose > 1:
                        # Show vote results on subcloud (match original label to valid) => not the good values here
                        vote_m = metrics_from_result(val_probs, dataset.input_labels['validation'], dataset.num_classes, label_to_idx=label_to_idx, proportions=proportions)
                        print('==> Confusion on sub clouds: ', vote_m.scalar_str)

                    if self.verbose > 1 and int(np.ceil(new_min)) % 2 == 0:
                        # Project predictions
                        proj_probs = [cur_prob[cur_proj,:] for cur_prob, cur_proj in zip(val_probs, dataset.validation_proj)]
                        vote_m = metrics_from_result(proj_probs, dataset.validation_labels, dataset.num_classes, label_to_idx=label_to_idx, proportions=None)
                        print('==> Confusion on full clouds:', vote_m)

                sess.run(ops['val_init_op'])
                vote_ind += 1

        if extra_ops:
            self.solve_extra_ops(sess, 'validation', dataset, val_probs, extra_ops=extra_ops, cum_dict=cum_dict)

        vote_m = metrics_from_result(val_probs, dataset.input_labels['validation'], dataset.num_classes, label_to_idx=label_to_idx, proportions=proportions)
        print('==> Confusion on sub clouds - final: ', vote_m.scalar_str)

        # Project predictions
        proj_probs = [cur_prob[cur_proj,:] for cur_prob, cur_proj in zip(val_probs, dataset.validation_proj)]
        print('==> Confusion on full clouds - final:')
        vote_m = metrics_from_result(proj_probs, dataset.validation_labels, dataset.num_classes, label_to_idx=label_to_idx, proportions=None)
        vote_m.print()
        print('\nfinished\n', flush=True)

        if self.config.save_val:
            self.save_split('validation', dataset, cum_dict)
        return

    # Test methods
    # ------------------------------------------------------------------------------------------------------------------

    def test_vote_seg(self, sess, ops, dataset, model, num_votes=20, test_path=None, make_zip=True):

        config = self.config
        assert os.path.isdir(config.saving_path), f'not a dir: {config.saving_path}'
        if test_path is None:
            test_path = os.path.join(config.saving_path, 'test')
        os.makedirs(test_path, exist_ok=True)

        options = None  # tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = None  # tf.RunMetadata()
        feed_dict = {ops['is_training']: False}

        # Smoothing parameter for votes
        test_smooth = 0.98

        # Initialise iterator with test data
        sess.run(ops['test_init_op'])

        # Initiate global prediction over val clouds
        test_ops, cum_dict, extra_ops = self.initialize(ops, dataset, model, 'test')
        test_probs = cum_dict['prob']

        vote_ind = 0
        last_min = -0.5 
        if config.num_votes:
            num_votes = config.num_votes
        while last_min < num_votes:
            try:
                cur_rst = sess.run(test_ops, feed_dict=feed_dict, options=options, run_metadata=run_metadata)
                # Stack all test predictions for each class separately - iterate over each gpu & cloud
                self.cumulate_probs(dataset, model, cur_rst, cum_dict, task='seg', smooth=test_smooth)

            except tf.errors.OutOfRangeError:
                # NOTE: need to check
                new_min = np.min(dataset.min_potentials['test'])
                if self.verbose:
                    print(f'Step {vote_ind:3d}, end. Min potential = {new_min:.1f}', flush=True)

                if last_min + 1 < new_min:
                    # Update last_min
                    last_min += 1

                # if int(last_min) > 0 and int(last_min) // 5 == 0:  # periodic test results
                #     self.project_test_predictions(dataset, test_path)

                sess.run(ops['test_init_op'])
                vote_ind += 1

        if self.verbose:
            new_min = np.min(dataset.min_potentials['test'])
            print(f'Step {vote_ind:3d}, end. Min potential = {new_min:.1f}', flush=True)

        self.project_test_predictions(dataset, test_probs, test_path)
        print('\nfinished\n', flush=True)

        if make_zip:
            zip_name = test_path.split(os.sep)  # cfg name / Log_* / test_*
            zip_name = '_'.join([i for i in ['test', *zip_name[-3:-1], zip_name[-1][len('test'):].strip('_')] if i])
            # include only test_* dir (except Semantic3D)
            j = 'j' if config.dataset == 'Semantic3D' else ''
            os.system(f'cd {os.path.dirname(test_path)}; zip -rmTq{j} {zip_name}.zip {test_path.split(os.sep)[-1]}/*')  # -m to move, -j junk file, -T test integrity, -q quiet
        return

    def project_test_predictions(self, dataset, test_probs, test_path):

        # Project predictions
        t1 = time.time()
        files = dataset.test_files
        config = self.config
        if config.save_test:
            pred_path = os.sep.join([*test_path.split(os.sep)[:-1], test_path.split(os.sep)[-1].replace('test', 'predictions')])  # model pred
            os.makedirs(pred_path, exist_ok=True)

        for i_test, file_path in enumerate(files):

            # Get file
            points = dataset.load_evaluation_points(file_path)

            # Reproject probs
            probs = test_probs[i_test][dataset.test_proj[i_test], :]

            # Get the predicted labels
            preds = dataset.idx_to_label[np.argmax(probs, axis=-1)]

            # Project potentials on original points
            pots = dataset.potentials['test'][i_test][dataset.test_proj[i_test]]

            # Save plys - predictions & probs
            cloud_name = file_path.split('/')[-1]
            if config.save_test:
                test_name = os.path.join(pred_path, cloud_name)
                prob_names = ['_'.join(dataset.label_to_names[label].split()) for label in dataset.label_values if label not in dataset.ignored_labels]
                write_ply(test_name,
                        [points, preds, pots, probs],
                        ['x', 'y', 'z', 'preds', 'pots'] + prob_names)

            # Save ascii preds - submission files
            if config.dataset == 'Semantic3D':
                ascii_name = os.path.join(test_path, dataset.ascii_files[cloud_name])
            else:
                ascii_name = os.path.join(test_path, cloud_name[:-4] + '.txt')
            np.savetxt(ascii_name, preds, fmt='%d')

        t2 = time.time()
        if self.verbose:
            print('\nReproject Vote in {:.1f}s\n'.format(t2-t1))

    def test_cloud_segmentation_on_val(self, model, dataset, num_votes=100):

        ##########
        # Initiate
        ##########

        # Smoothing parameter for votes
        test_smooth = 0.95

        # Initialise iterator with train data
        self.sess.run(dataset.val_init_op)

        # Initiate global prediction over test clouds
        nc_model = config.num_classes
        self.test_probs = [np.zeros((l.shape[0], nc_model), dtype=np.float32) for l in dataset.input_labels['validation']]

        # Number of points per class in validation set
        val_proportions = np.zeros(nc_model, dtype=np.float32)
        i = 0
        for label_value in dataset.label_values:
            if label_value not in dataset.ignored_labels:
                val_proportions[i] = np.sum([np.sum(labels == label_value)
                                             for labels in dataset.validation_labels])
                i += 1

        # Test saving path
        if config.save_test:
            test_path = join(model.saving_path, 'test')
            if not exists(test_path):
                makedirs(test_path)
            if not exists(join(test_path, 'val_predictions')):
                makedirs(join(test_path, 'val_predictions'))
            if not exists(join(test_path, 'val_probs')):
                makedirs(join(test_path, 'val_probs'))
        else:
            test_path = None

        #####################
        # Network predictions
        #####################

        i0 = 0
        epoch_ind = 0
        last_min = -0.5
        mean_dt = np.zeros(2)
        last_display = time.time()
        _t = time.time()
        if config.num_votes:
            num_votes = config.num_votes
        while last_min < num_votes:

            try:
                # Run one step of the model.
                t = [time.time()]
                ops = (self.prob_logits,
                       model.labels,
                       model.inputs['in_batches'],
                       model.inputs['point_inds'],
                       model.inputs['cloud_inds'])
                stacked_probs, labels, batches, point_inds, cloud_inds = self.sess.run(ops, {model.dropout_prob: 1.0})
                t += [time.time()]

                # Get predictions and labels per instance
                # ***************************************

                # Stack all validation predictions for each class separately
                max_ind = np.max(batches)
                for b_i, b in enumerate(batches):
                    # Eliminate shadow indices
                    b = b[b < max_ind - 0.5]

                    # Get prediction (only for the concerned parts)
                    probs = stacked_probs[b]
                    inds = point_inds[b]
                    c_i = cloud_inds[b_i]

                    # Update current probs in whole cloud
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1-test_smooth) * probs

                # Average timing
                t += [time.time()]
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Display
                if (t[-1] - last_display) > self.gap_display:
                    last_display = t[-1]
                    message = 'Epoch {:3d}, step {:3d} (timings : {:4.2f} {:4.2f}). min potential = {:.1f}'
                    print(message.format(epoch_ind,
                                         i0,
                                         1000 * (mean_dt[0]),
                                         1000 * (mean_dt[1]),
                                         np.min(dataset.min_potentials['validation'])))

                i0 += 1


            except tf.errors.OutOfRangeError:

                # Save predicted cloud
                new_min = np.min(dataset.min_potentials['validation'])
                print('Epoch {:3d}, end. Min potential = {:.1f} (last_min = {:.1f})'.format(epoch_ind, new_min, last_min))

                if last_min + 1 < new_min:

                    # Update last_min
                    last_min += 1

                    # Show vote results (On subcloud so it is not the good values here)
                    print('\nConfusion on sub clouds ---')
                    Confs = []
                    for i_test in range(dataset.num_validation):

                        # Insert false columns for ignored labels
                        probs = self.test_probs[i_test]
                        for l_ind, label_value in enumerate(dataset.label_values):
                            if label_value in dataset.ignored_labels:
                                probs = np.insert(probs, l_ind, 0, axis=1)

                        # Predicted labels
                        preds = dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)

                        # Targets
                        targets = dataset.input_labels['validation'][i_test]

                        # Confs
                        Confs += [confusion_matrix(targets, preds, dataset.label_values)]

                    # Regroup confusions
                    C = np.sum(np.stack(Confs), axis=0).astype(np.float32)

                    # Remove ignored labels from confusions
                    for l_ind, label_value in reversed(list(enumerate(dataset.label_values))):
                        if label_value in dataset.ignored_labels:
                            C = np.delete(C, l_ind, axis=0)
                            C = np.delete(C, l_ind, axis=1)

                    # Rescale with the right number of point per class
                    C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

                    # Compute IoUs
                    IoUs = IoU_from_confusions(C)
                    mIoU = np.mean(IoUs)
                    s = '{:5.2f} | '.format(100 * mIoU)
                    for IoU in IoUs:
                        s += '{:5.2f} '.format(100 * IoU)
                    print(s + '\n', flush=True)

                    if int(np.ceil(new_min)) % 5 == 0:

                        self.solve_extra_ops('validation', model, dataset)

                        # Project predictions
                        print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))), end='')
                        t1 = time.time()
                        files = dataset.train_files
                        i_val = 0
                        proj_probs = []
                        for i, file_path in enumerate(files):
                            if dataset.all_splits[i] == dataset.validation_split:

                                # Reproject probs on the evaluations points
                                probs = self.test_probs[i_val][dataset.validation_proj[i_val], :]
                                proj_probs += [probs]
                                i_val += 1

                        t2 = time.time()
                        print(' - {:.1f} s\n'.format(t2 - t1))

                        # Show vote results
                        print('Confusion on full clouds', end='')
                        t1 = time.time()
                        Confs = []
                        for i_test in range(dataset.num_validation):

                            # Insert false columns for ignored labels
                            for l_ind, label_value in enumerate(dataset.label_values):
                                if label_value in dataset.ignored_labels:
                                    proj_probs[i_test] = np.insert(proj_probs[i_test], l_ind, 0, axis=1)

                            # Get the predicted labels
                            preds = dataset.label_values[np.argmax(proj_probs[i_test], axis=1)].astype(np.int32)

                            # Confusion
                            targets = dataset.validation_labels[i_test]
                            Confs += [confusion_matrix(targets, preds, dataset.label_values)]

                        t2 = time.time()
                        print(' - {:.1f} s\n'.format(t2 - t1))

                        # Regroup confusions
                        C = np.sum(np.stack(Confs), axis=0)

                        # Remove ignored labels from confusions
                        for l_ind, label_value in reversed(list(enumerate(dataset.label_values))):
                            if label_value in dataset.ignored_labels:
                                C = np.delete(C, l_ind, axis=0)
                                C = np.delete(C, l_ind, axis=1)

                        IoUs, OA, mACC = IoU_from_confusions(C, get_acc=True)
                        mIoU = np.mean(IoUs)
                        s = '{:5.2f} | '.format(100 * mIoU)
                        for IoU in IoUs:
                            s += '{:5.2f} '.format(100 * IoU)
                        print('-' * len(s))
                        print(s)
                        print('-' * len(s) + '\n')
                        print(f'OA = {OA:.3f}')
                        print(f'mACC = {mACC:.3f}')

                        # Save predictions
                        if config.save_test:
                            print('Saving clouds')
                            t1 = time.time()
                            files = dataset.train_files
                            i_test = 0
                            for i, file_path in enumerate(files):
                                if dataset.all_splits[i] == dataset.validation_split:

                                    # Get points
                                    points = dataset.load_evaluation_points(file_path)

                                    # Get the predicted labels
                                    preds = dataset.label_values[np.argmax(proj_probs[i_test], axis=1)].astype(np.int32)

                                    # Project potentials on original points
                                    pots = dataset.potentials['validation'][i_test][dataset.validation_proj[i_test]]

                                    # Save plys
                                    cloud_name = file_path.split('/')[-1]
                                    test_name = join(test_path, 'val_predictions', cloud_name)
                                    write_ply(test_name,
                                            [points, preds, pots, dataset.validation_labels[i_test]],
                                            ['x', 'y', 'z', 'preds', 'pots', 'gt'])
                                    test_name2 = join(test_path, 'val_probs', cloud_name)
                                    prob_names = ['_'.join(dataset.label_to_names[label].split())
                                                for label in dataset.label_values]
                                    write_ply(test_name2,
                                            [points, proj_probs[i_test]],
                                            ['x', 'y', 'z'] + prob_names)
                                    i_test += 1
                            t2 = time.time()
                            print('Done in {:.1f} s\n'.format(t2 - t1), flush=True)

                self.sess.run(dataset.val_init_op)
                epoch_ind += 1
                i0 = 0
                continue
        print(f'finished - {time.time() - _t} s', flush=True)
        return

    def cumulate_probs(self, dataset, model, rst, cum_dict, task, smooth):
        # cum_dict - {cum_dict name : {args : rst_dict}}

        # iterate over gpu
        split = 'validation'
        for gpu_i, cloud_inds in enumerate(rst['inputs']['cloud_inds']):
            point_inds = rst['inputs']['point_inds'][gpu_i]

            b_start = 0
            # iterate over clouds
            for b_i, c_i in enumerate(cloud_inds):  # [B]
                if 'batches_len' in rst['inputs']:  # [BxN] - stacked
                    b_len = rst['inputs']['batches_len'][gpu_i][0][b_i]  # npoints in cloud
                    b_i = np.arange(b_start, b_start + b_len)
                    b_start += b_len
                else:  # [B, N] - batched
                    pass
                inds = point_inds[b_i]  # input point inds

                probs = rst[task]['probs'][gpu_i][b_i]
                labels = rst[task]['labels'][gpu_i][b_i]
                if set(labels) == {-1}:
                    continue
                if 'conf' in cum_dict:
                    cur_conf = confusion_matrix(labels, np.argmax(probs, axis=-1).astype(np.int), labels=np.arange(dataset.num_classes))
                    cum_dict['conf'] += cur_conf
                if 'prob' in cum_dict:
                    cum_dict['prob'][c_i][inds] = smooth * cum_dict['prob'][c_i][inds] + (1 - smooth) * probs
                if 'feature' in cum_dict:
                    cum_dict['feature'][c_i][inds] = smooth * cum_dict['feature'][c_i][inds] + (1 - smooth) * rst[task]['latent'][gpu_i][b_i]

                # extra ops
                if 'center' in cum_dict:  # retain center, exclude sample margin
                    probs = rst['seg']['probs'][gpu_i][b_i]
                    labels = rst['seg']['labels'][gpu_i][b_i]
                    thr_list = [0.7, 0.8, 0.9]  # probs vote
                    if not cum_dict['center']:  # init
                        cum_dict['center'] = dict([(thr, self.init_pointcloud_log(dataset, split, dataset.num_classes)) for thr in thr_list])
                        cum_dict['margin'] = dict([(thr, self.init_pointcloud_log(dataset, split, dataset.num_classes)) for thr in thr_list])
                        cum_dict['center']['conf'], cum_dict['margin']['conf'] = defaultdict(lambda: 0), defaultdict(lambda: 0)  # conf
                    for thr in thr_list:
                        pts = np.array(dataset.input_trees[split][c_i].data, copy=False)[inds]  # [N, 3]
                        dist = ((pts - pts.mean(axis=-1, keepdims=True))** 2).sum(axis=-1)  # [N]
                        center_mask = dist < np.percentile(dist, thr)
                        margin_mask = 1 - center_mask
                        for mask_n, mask in zip(['center', 'margin'], [center_mask, margin_mask]):
                            cum_dict[mask_n][thr][c_i][inds[mask]] = smooth * cum_dict[mask_n][thr][c_i][inds[mask]] + (1 - smooth) * probs[mask]
                            cum_dict[mask_n]['conf'][thr] += confusion_matrix(labels[mask], probs[mask].argmax(axis=-1).astype(int), labels=np.arange(dataset.num_classes))

                if 'edge' in cum_dict:
                    # consider sample conf only - evaluating over cloud
                    edge_classes = model.head_dict['edge']['config'].num_classes
                    if not cum_dict['edge']:  # init
                        cum_dict['edge']['conf'] = defaultdict(lambda: 0)
                        # cum_dict['edge']['prob'] = defaultdict(lambda: self.init_pointcloud_log(dataset, 'validation', edge_classes))
                    for k, probs in rst['edge']['probs'].items():  # edge at diff stage
                        labels = rst['edge']['labels'][k]
                        cum_dict['edge']['conf'][k] += confusion_matrix(labels, probs.argmax(axis=-1), labels=np.arange(edge_classes))

                if 'prob_novote' in cum_dict:
                    cum_dict['prob_novote'][c_i][inds] = probs

                k_list = [k for k in cum_dict if re.match(f'feature-(up|down)-\d+', k)]
                for k in k_list:
                    f = rst[k]
                    cum_dict[k][c_i][inds] = smooth * cum_dict[k][c_i][inds] + (1 - smooth) * f[gpu_i][b_i]


    def solve_extra_ops(self, sess, split, dataset, probs, extra_ops, cum_dict):
        """ assume: probs align with dataset split
        """
        if not extra_ops and not cum_dict:
            return
        
        verbose = self.verbose

        config = self.config
        extra_ops = extra_ops.split('-') if isinstance(extra_ops, str) else extra_ops
        idx_list = [i for i in extra_ops if i.isdigit()]
        label_to_idx = dataset.label_to_idx
        proportions = dataset.val_proportions

        # container for all possible extra ops
        conf_total = 0
        if 'neighbor' not in cum_dict:
            kr = config.kr_search[0]
            neighbor_dict = dict([(i, {}) for i in [kr]])  # {k_r : {cloud_idx: neighbor idx}}
        else:
            neighbor_dict = cum_dict['neighbor']
            kr = list(neighbor_dict.keys())[0]

        boundary_dict = dict([(i, defaultdict(lambda: 0, {'mask_label': [], 'mask_pred': []})) for i in neighbor_dict.keys()])
        feature_dict = defaultdict(lambda: defaultdict(lambda: 0, {'boundf': {}, 'var-std': []}))  # { k : {dist sum/cnt: float/int, boundf: arr, var-std: arr} }
        prob_dict = defaultdict(lambda: 0, {'boundp': {}})

        feature_klist = [k for k in extra_ops if re.match('feature-(up|down)-\d+', k)]  # key in feature_dict - specifying stage (n, i)
        feature_klist += ['feature'] if 'feature' in extra_ops else []

        # accumulate over all pred-label
        for cloud_idx, (prob, label) in enumerate(zip(probs, dataset.input_labels[split])):
            if idx_list and cloud_idx not in idx_list:
                continue
            # predicted labels
            pred = np.argmax(prob, axis=-1)
            label = label_to_idx[label].astype(int)  # match to the preds - -1 for invalid
            valid_mask = label >= 0 if len(self.config.ignored_labels) > 0 else None

            if 'stat' in extra_ops:
                conf_total += confusion_matrix(label, pred, labels=np.arange(config.num_classes))  # specifying labels to exclude invalid -1

            if 'boundary' in extra_ops:
                from .tf_utils import get_boundary_mask
                class_cnt = dataset.get_class_cnt()
                # calc acc on boundary
                for k in neighbor_dict.keys():
                    neighbor_idx = self._search_func(k, cloud_idx, split, dataset, neighbor_dict, verbose=verbose)
                    for mask_n, mask_label in zip(['label', 'pred'], [label, pred]):
                        mask_bound, mask_plain, posneg = get_boundary_mask(mask_label, neighbor_idx=neighbor_idx, valid_mask=valid_mask, use_np=True, iterative=k>kr, posneg=True)
                        boundary_dict[k][f'mask_{mask_n}'] = mask_bound
                        boundary_dict[k][f'plain_{mask_n}'] = mask_plain
                        boundary_dict[k][f'posneg_{mask_n}'] = posneg
                        boundary_dict[k][f'conf_bound_{mask_n}'] += confusion_matrix(label[mask_bound], pred[mask_bound], labels=np.arange(config.num_classes))  # label-pred on boundary
                        boundary_dict[k][f'conf_plain_{mask_n}'] += confusion_matrix(label[mask_plain], pred[mask_plain], labels=np.arange(config.num_classes))  # label-pred on plain

                        pred_ideal = pred.copy()
                        pred_ideal[mask_bound] = label[mask_bound]
                        boundary_dict[k][f'conf_ideal_{mask_n}'] += confusion_matrix(label, pred_ideal, labels=np.arange(config.num_classes))
                        if 'boundary' in self.config.save_val:
                            boundary_dict[k][f'{cloud_idx}/mask_{mask_n}'] = mask_bound
                    # iou of label-pred boundary
                    mask_label, mask_pred = boundary_dict[k]['mask_label'], boundary_dict[k]['mask_pred']
                    bound_I = np.logical_and(mask_label, mask_pred)
                    bound_U = np.logical_or(mask_label, mask_pred)
                    boundary_dict[k]['mask_I'] += bound_I.sum()
                    boundary_dict[k]['mask_U'] += bound_U.sum()

                if verbose:
                    print_mem(f'boundary - {cloud_idx} - done', check_time=True, check_sys=True, flush=True)

            if 'boundary' in extra_ops and 'prob' in extra_ops:
                _, d = self._get_boundary_diff(prob, cloud_idx, split, dataset, neighbor_dict, boundary_dict, prob_dict, 'kl', valid_mask, verbose=verbose)
                if 'boundp' in self.config.save_val:
                    prob_dict['boundp'][cloud_idx] = d  # d[kr][dist_n][mean/max] - [BxN] - mean/max dist to neighborhood

            for k in feature_klist:
                f_dict = feature_dict[k]
                features = cum_dict[k][cloud_idx]
                norm = np.sqrt((features ** 2).sum(axis=-1)).sum(axis=0)
                mean = features.mean(axis=0)
                var = 0
                for f in features:
                    var += (f - mean) ** 2
                var /= len(features) - 1
                std = np.sqrt(var)
                f_dict['norm'] += norm
                f_dict['var-std'] += [(var, std)]
                f_dict['sum'] += features.sum(axis=0)
                f_dict['cnt'] += len(features)

                if 'boundary' in extra_ops:
                    _, d = self._get_boundary_diff(features, cloud_idx, split, dataset, neighbor_dict, boundary_dict, f_dict, 'l2-cos-norml2', valid_mask, verbose=verbose)
                    # if config.debug:
                    #     print(np.array(dataset.input_trees['validation'][cloud_idx].data, copy=False).shape, 'features - ', features.shape, config.debug)
                    #     print_dict(d, fn=np.shape)
                    if 'boundf' in self.config.save_val:
                        f_dict['boundf'][cloud_idx] = d  # d[kr][dist_n][mean/max] - [BxN] - mean/max dist to neighborhood

        print('extra ops ----')
        if 'boundary' in extra_ops:
            print('\n-- boundary --')
            for mask_n in ['label', 'pred']:
                for k in boundary_dict:
                    for conf_n in ['bound', 'plain', 'ideal']:
                        conf = boundary_dict[k][f'conf_{conf_n}_{mask_n}']
                        m = metrics_from_confusions(conf)
                        # if config.debug:
                        #     print(conf)
                        print(f'\t{k}_{mask_n} \t - {conf_n} : {m}')
                    print()
            print('\t -- bound IoU --')
            for k in boundary_dict:
                print(f'\t{k}\t -', boundary_dict[k]['mask_I'] / boundary_dict[k]['mask_U'])
                # for w in [w.replace('mask_Iw_', '', 1) for w in boundary_dict[k] if w.startswith('mask_Iw_')]:
                #     sc = boundary_dict[k][f'mask_Iw_{w}'] / boundary_dict[k][f'mask_Uw_{w}']
                #     if hasattr(sc, '__len__'):
                #         sc = [sc.mean(), *sc]
                #     with np.printoptions(linewidth=sys.maxsize, precision=3):
                #         print(f'\t{k}\t {w} \t -', sc)

            print('-- --')

        for k in feature_klist:
            print(f'\n-- {k} --')
            f_dict = feature_dict[k]
            cnt = f_dict['cnt']
            print('\t mean norm', f_dict['norm'] / cnt)
            print('\t mean features', f_dict['sum'] / cnt)
            for n, lst in zip('var-std'.split('-'), zip(*f_dict['var-std'])):
                m = sum(lst) / len(lst)
                print(f'\t mean {n} over clouds', m, m.mean())

            if 'boundary' in extra_ops:
                for k in boundary_dict:
                    for dist_n, dist_d in f_dict[k].items():
                        print(f'\t k = {k}\t{dist_n}')
                        for area_n in ['bound', 'plain', 'overall', 'pos', 'neg']:
                            print(f'\t\t {area_n}\t', dist_d[area_n] / dist_d[f'{area_n}_cnt'])
            print('-- --')

        if 'prob' in extra_ops and 'boundary' in extra_ops:
            print('\n-- prob --')
            for k in boundary_dict:
                for dist_n, dist_d in prob_dict[k].items():
                    print(f'\t k = {k}\t{dist_n}')
                    for area_n in ['bound', 'plain', 'overall', 'pos', 'neg']:
                        print(f'\t\t {area_n}\t', dist_d[area_n] / dist_d[f'{area_n}_cnt'])
            print('-- --')

        if 'stat' in extra_ops:
            print('\n-- stat --')
            conf_dict = {'total': conf_total}
            for mask_n in ['label', 'pred']:
                for k in boundary_dict:
                    for conf_n in ['bound', 'plain']:
                        conf_dict[f'{k}-{mask_n}-{conf_n}'] = boundary_dict[k][f'conf_{conf_n}_{mask_n}']
            for conf_n in list(conf_dict.keys()):
                conf = conf_dict[conf_n]
                conf_dict[f'{conf_n}_TP'] = np.diagonal(conf, axis1=-2, axis2=-1)
                conf_dict[f'{conf_n}_FN'] = np.sum(conf, axis=-1) - conf_dict[f'{conf_n}_TP']
                conf_dict[f'{conf_n}_FP'] = np.sum(conf, axis=-2) - conf_dict[f'{conf_n}_TP']

            err_total = conf_total.sum() - np.diagonal(conf_total, axis1=-2, axis2=-1).sum()
            for mask_n in ['label', 'pred']:
                print(f'\t -- % of error on bound - {mask_n}')
                for k in boundary_dict:
                    conf = conf_dict[f'{k}-{mask_n}-bound']
                    conf_plain = conf_dict[f'{k}-{mask_n}-plain']
                    err_bound = conf.sum() - np.diagonal(conf, axis1=-2, axis2=-1).sum()
                    err_plain = conf_plain.sum() - np.diagonal(conf_plain, axis1=-2, axis2=-1).sum()
                    print(f'\t\t k = {k}: {(err_bound/err_total)*100:5.1f} - bound {err_bound} / plain {err_plain} - total {err_total}')

            for err_t in ['FP', 'FN']:
                print(f'\t -- {err_t}')
                for mask_n in ['label', 'pred']:  # fp/fn on bound/plain
                    for k in boundary_dict:
                        bound_err = conf_dict[f'{k}-{mask_n}-bound_{err_t}']
                        plain_err = conf_dict[f'{k}-{mask_n}-plain_{err_t}']
                        print(f'\t {k:<2} {mask_n} - bound / plain', log_percentage([bound_err, plain_err]), '|', log_percentage([bound_err.sum(), plain_err.sum()]),
                                '| total', conf_dict[f'total_{err_t}'].sum(), '=', bound_err.sum(), '+', plain_err.sum())
            print('-- --')
        
        if 'center' in cum_dict:
            print('\n-- center-margin --')
            thr_list = [thr for thr in cum_dict['center'] if isinstance(thr, float)]
            for thr in thr_list:
                print(f'\n\t -- thr = {thr}')
                for mask_n in ['center', 'margin']:
                    m_sample = metrics_from_confusions(cum_dict[mask_n]['conf'][thr])
                    m_vote = metrics_from_result(cum_dict[mask_n][thr], dataset.input_labels[split], dataset.num_classes, label_to_idx=label_to_idx)
                    print(f'\t{mask_n} - sample', m_sample)
                    print(f'\t{mask_n} - vote: ', m_vote)

            print('-- --')

        if 'edge' in cum_dict:
            print('\n-- edge --')
            for k, conf in cum_dict['edge']['conf'].items():  # edge at diff stage
                m_sample = metrics_from_confusions(conf)
                print(f'\t{k}\t', m_sample)
            print('-- --')

        if 'prob_novote' in cum_dict:
            assert split == 'validation', f'extra_ops: prob_novote - split should be validation but get {split}'
            print('\n-- probs novote --')
            vote_m = metrics_from_result(cum_dict['prob_novote'], dataset.input_labels[split], dataset.num_classes, label_to_idx=label_to_idx, proportions=proportions)
            print('\t sub clouds - final: ', vote_m.scalar_str)

            # Project predictions
            proj_probs = [cur_prob[cur_proj,:] for cur_prob, cur_proj in zip(cum_dict['prob_novote'], dataset.validation_proj)]
            vote_m = metrics_from_result(proj_probs, dataset.validation_labels, dataset.num_classes, label_to_idx=label_to_idx, proportions=None)
            print(f'\t full clouds - final: {vote_m}')

        for k in self.config.save_val.split('-'):
            if k == 'neighbor':
                self.save_extra['neighbor'] = neighbor_dict
            elif k == 'boundary':
                self.save_extra['boundary'] = boundary_dict
            elif k == 'boundf':
                self.save_extra['boundf'] = {k.replace('feature', 'boundf'): f_dict['boundf'] for k, f_dict in feature_dict.items()}  # {k = 'boundf-{n}-{i}': {cloud_idx: ...}}
            elif k == 'boundp':
                self.save_extra['boundp'] = {'boundp': prob_dict['boundp']}
            elif k in ['feature', 'prob']:
                pass
            elif k.isdigit():
                pass
            elif k:
                raise ValueError(f'not supported save val k = {k} from {self.config.save_val}')

        if 'conf' in extra_ops:
            assert split == 'validation', f'extra_ops: conf - split should be validation but get {split}'
            proj = getattr(dataset, f'{split}_proj')
            full_label = getattr(dataset, f'{split}_labels')
            proj_probs = [cur_prob[cur_proj,:] for cur_prob, cur_proj in zip(probs, proj)]
            print('\n -- conf --')
            print('\t==> Confusion on full clouds - final:')
            vote_m = metrics_from_result(proj_probs, dataset.validation_labels, dataset.num_classes, label_to_idx=label_to_idx, proportions=None)
            print('\t', vote_m.final_str.replace('\n', '\n\t'))
            print('-- --')

    def _search_func(self, k_r, cloud_idx, split, dataset, neighbor_dict, verbose=True):  # create tf_ops of generating neighbor_idx & get result
        if cloud_idx in neighbor_dict[k_r]:
            return neighbor_dict[k_r][cloud_idx]

        config = self.config
        points = np.array(dataset.input_trees[split][cloud_idx].data, copy=False)  # [N, 3]

        from ops import get_tf_func
        func = get_tf_func(config.search, verbose=verbose)

        if config.search in ['knn']:
            tf_ops = tf.squeeze(func(points[None, ...], points[None, ...], k_r), axis=0)
        elif config.search in ['radius']:
            tf_ops = func(points, points, [len(points)], [len(points)], k_r)
            # if hasattr(dataset, 'neighborhood_limits'):
            #     print('neighborhood_limits', dataset.neighborhood_limits[0])
            #     tf_ops = tf_ops[..., :dataset.neighborhood_limits[0]]
        else:
            raise

        if verbose:
            print_mem(f'k = {k_r} - start', check_time=True, check_sys=True, flush=True)
        with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}, allow_soft_placement=True)) as s:
            neighbor_idx = s.run(tf_ops)
        if verbose:
            print_mem(f'neighbor_idx {neighbor_idx.shape}', check_time=True, check_sys=True, flush=True)

        neighbor_dict[k_r][cloud_idx] = neighbor_idx  # neighbor idx - np arr
        return neighbor_idx


    def _get_boundary_diff(self, features, cloud_idx, split, dataset, neighbor_dict, boundary_dict, feature_dict, dist_name, valid_mask, verbose=True):
        features = np.concatenate([features, np.zeros(features[:1].shape, features.dtype)])  # [N+1, d]
        dist_d = {}
        for k in boundary_dict:
            if k not in feature_dict: feature_dict[k] = {}

            neighbor_idx = self._search_func(k, cloud_idx, split, dataset, neighbor_dict, verbose=verbose)
            neighbor_idx = neighbor_idx[..., 1:]  # exclude self-loop

            mask_n = neighbor_idx < len(neighbor_idx) if neighbor_idx.max() >= len(neighbor_idx) else None
            if mask_n is not None and valid_mask is not None:
                mask = np.logical_and(np.expand_dims(valid_mask, axis=-1), mask_n)
            else:
                mask = valid_mask if mask_n is None else mask_n

            mask_bound = boundary_dict[k][f'mask_label']
            mask_plain = boundary_dict[k][f'plain_label']
            pos, neg = boundary_dict[k][f'posneg_label']
            pos = pos[..., 1:]
            neg = neg[..., 1:]

            dist_d[k] = {}
            for dist_n in dist_name.split('-'):
                if dist_n not in feature_dict[k]: feature_dict[k][dist_n] = defaultdict(lambda: 0)
                dist_d[k][dist_n] = {}

                dist = np.zeros(neighbor_idx.shape)  # [N, k]
                for i, idx in enumerate(neighbor_idx):
                    fc = np.expand_dims(features[i], axis=-2)  # [1, d]
                    fn = features[idx]  # [k, d]

                    if dist_n in ['cos', 'norml2']:
                        fc /= np.sqrt((fc ** 2).sum(axis=-1, keepdims=True) + 1e-12)
                        fn /= np.sqrt((fn ** 2).sum(axis=-1, keepdims=True) + 1e-12)

                    if dist_n in ['l2', 'norml2']:
                        dist[i] = ((fc - fn) ** 2).sum(axis=-1)  # [k]
                    elif dist_n == 'cos':
                        cos = (fc * fn).sum(axis=-1)
                        dist[i] = cos
                        # dist = (1-dist) / 2
                    elif dist_n == 'kl':
                        dist[i] = (fc * np.log((fc / (fn + 1e-12)) + 1e-12)).sum(axis=-1)
                    else:
                        raise
                    if (np.isnan(dist[i])).any():
                        print(fc)
                        print(fn)
                        raise ValueError('nan in distance')

                dist_sum = dist[np.where(mask)].sum() if mask is not None else dist.sum()
                dist_cnt = mask.sum() if mask is not None else np.prod(dist.shape)

                dist_pos = dist[np.where(pos)].sum()
                dist_neg = dist[np.where(neg)].sum()

                # get distance with neighbor (mean)
                if mask is not None:
                    mask_n = mask.astype(float)  # [N, k]
                    mask_c = mask_n.sum(axis=-1)  # [N]
                    dist_max = (dist*mask_n).max(axis=-1)
                    dist = (dist*mask_n).sum(axis=-1) / (mask_c + 1e-12)
                else:
                    dist_max = dist.max(axis=-1)
                    dist = dist.mean(axis=-1)

                feature_dict[k][dist_n]['overall'] += dist_sum
                feature_dict[k][dist_n]['overall_cnt'] += dist_cnt
                feature_dict[k][dist_n]['pos'] += dist_pos
                feature_dict[k][dist_n]['pos_cnt'] += pos.sum()
                feature_dict[k][dist_n]['neg'] += dist_neg
                feature_dict[k][dist_n]['neg_cnt'] += neg.sum()

                feature_dict[k][dist_n]['plain'] += (dist * mask_plain).sum()
                feature_dict[k][dist_n]['plain_cnt'] += mask_plain.sum()
                feature_dict[k][dist_n]['bound'] += (dist * mask_bound).sum()
                feature_dict[k][dist_n]['bound_cnt'] += mask_bound.sum()

                feature_dict[k][dist_n]['plainmax'] += (dist_max * mask_plain).sum()
                feature_dict[k][dist_n]['plainmax_cnt'] += mask_plain.sum()
                feature_dict[k][dist_n]['boundmax'] += (dist_max * mask_bound).sum()
                feature_dict[k][dist_n]['boundmax_cnt'] += mask_bound.sum()

                dist_d[k][dist_n]['mean'] = dist
                dist_d[k][dist_n]['max'] = dist_max

        return feature_dict, dist_d

    def save_split(self, split, dataset, cum_dict):
        extra_dict = self.save_extra
        step = self.config.model_path.split('-')[-1]
        h5f_path = f'{self.config.saving_path}/{split}_{step}.h5'
        print(f'saving into {h5f_path}')
        if self.config.debug:
            print_dict(cum_dict, fn=np.shape, head='cum_dict')
            print_dict(extra_dict, fn=np.shape, head='extra_dict')

        # specifying stage-used
        feature_klist = [k for k in cum_dict if re.match('feature-(up|down)-\d+', k)]
        feature_klist += ['feature'] if 'feature' in cum_dict else []

        idx_list = [i for i in self.config.extra_ops.split('-') if i.isdigit()]
        idx_list += [int(i) for i in self.config.save_val.split('-') if i.isdigit()]

        import h5py
        h5f = h5py.File(h5f_path, 'a')
        for cloud_idx, label in enumerate(dataset.input_labels[split]):
            if idx_list and cloud_idx not in idx_list:
                continue
            # f_name = dataset.input_files[split][cloud_idx]
            if f'{cloud_idx}' not in h5f:
                h5f.create_group(f'{cloud_idx}')
            h5g = h5f[f'{cloud_idx}']

            if 'prob' in h5g: del h5g['prob']
            h5g.create_dataset('prob', data=cum_dict['prob'][cloud_idx])

            if self.config.save_val_full:  # re-project to full pt & match original label
                cur_proj = getattr(dataset, f'{split}_proj')[cloud_idx]
                pred_full = dataset.idx_to_label[np.argmax(prob, axis=-1)]
                pred_full = pred_full[cur_proj]
                h5g.create_dataset('full_pred', data=pred_full)  # need to read ply for points & labels

            if 'feature' in self.config.save_val:
                for k in feature_klist:
                    if k in h5g: del h5g[k]
                    h5g.create_dataset(k, data=cum_dict[k][cloud_idx])

            # extra ops
            if 'boundary' in self.config.save_val:
                if 'boundary' in h5g: del h5g['boundary']
                for kr in extra_dict['boundary']:
                    for n in extra_dict['boundary']:  # may have pred-{n}-{i}
                        h5g.create_dataset(f'boundary/{kr}/{n}', data=extra_dict['boundary'][kr][f'{cloud_idx}/mask_{n}'])
            if 'neighbor' in self.config.save_val:
                if 'neighbor' in h5g: del h5g['neighbor']
                for kr in extra_dict['neighbor']:
                    h5g.create_dataset(f'neighbor/{kr}', data=extra_dict['neighbor'][kr][cloud_idx])

            for n in ['boundp', 'boundf']:
                if n not in self.config.save_val:
                    continue
                for n, dist_d in extra_dict[n].items():
                    if n in h5g: del h5g[n]
                    dist_d = dist_d[cloud_idx]
                    for kr in dist_d:  # 0.1
                        for dist_n in dist_d[kr]:  # l2/cos/kl
                            for agg_n in dist_d[kr][dist_n]:  # mean/max
                                h5g.create_dataset(f'{n}/{kr}/{dist_n}/{agg_n}', data=dist_d[kr][dist_n][agg_n])
        attrs = {
            'dataset': self.config.dataset,
            'split': split
        }
        for k, v in attrs.items():
            h5f.attrs[k] = v
        h5f.close()


    def solve_extra_ops_from_file(self, path, extra_ops):
        import re, glob, h5py
        config = self.config
        print(f'solving extra ops = {config.extra_sop} from\n {path}')

        h5f = h5py.File(path, 'r')
        split = h5f.attrs['split']
        dataset_name = h5f.attrs['dataset']
        assert dataset_name == config.dataset

        pt_list = []
        label_list = []
        cum_dict = defaultdict(lambda: [])
        neighbor_dict = defaultdict(lambda: {})  # {kr: {cloud idx: [BxN, k]}}
        for cloud_idx in sorted([int(i) for i in h5f.keys()]):
            h5g = h5f[f'{cloud_idx}']
            print(f'reading cloud_idx = {cloud_idx}')
            for k in h5g:
                if k == 'prob':
                    cum_dict[k].append(np.array(h5g[k], dtype=np.float32))
                elif k == 'feature':
                    cum_dict[k].append(np.array(h5g[k], dtype=np.float32))
                elif k == 'label':
                    label_list.append(np.array(h5g[k], dtype=np.int32))
                elif k == 'point':
                    pt_list.append(np.array(h5g[k], dtype=np.float32))
                elif k == 'neighbor':
                    for kr in h5g['neighbor']:
                        neighbor_dict[int(kr)][int(cloud_idx)] = np.array(h5g[k][kr], dtype=np.int32)
                else:
                    print(f'k = {k} - ignored')
        h5f.close()

        import datasets
        dataset = getattr(datasets, f'{dataset_name}Dataset')  # datasets.[name]Dataset
        dataset = dataset(config)
        if label_list:  # if labels stored
            dataset.input_labels[split] = label_list
        if pt_list:  # if pts stored
            dataset.input_trees[split] = pt_list
        if neighbor_dict:
            cum_dict['neighbor'] = neighbor_dict

        with tf.Graph().as_default():
            sess = tf.Session()
            self.solve_extra_ops(sess, split, dataset, probs=cum_dict['prob'], extra_ops=extra_ops, cum_dict=cum_dict)
            sess.close()
        return

# Helper methods
# ------------------------------------------------------------------------------------------------------------------

def gen_scope(func):
    """ decorator: determine if using, or execute through the generator """
    def scopped_func(*args, yield_cloud=False, **kwargs):
        gen = func(*args, **kwargs)
        if yield_cloud:  # return the generator
            return gen
        else:  # exhaust the generator (execute)
            for _ in gen:
                pass
            return
    return scopped_func

@gen_scope
def cumulate_probs(cur_rst, cum_probs, smooth):
    # iterate over gpu
    for gpu_i, (stacked_probs, labels) in enumerate(zip(cur_rst['seg'])):
        point_inds = cur_rst['inputs']['point_inds'][gpu_i]
        cloud_inds, cur_rst['inputs']['cloud_inds'][gpu_i]
        b_start = 0
        for b_i, c_i in enumerate(cloud_inds):  # [B]

            if 'batches_len' in cur_rst['inputs']:  # [BxN] - stacked
                b_len = cur_rst['inputs']['batches_len'][gpu_i][0][b_i]  # npoints in cloud
                b_i = np.arange(b_start, b_start + b_len)
                b_start += b_len

            else:  # [B, N] - batched
                pass

            inds = point_inds[b_i]
            probs = stacked_probs[b_i]
            cum_probs[c_i][inds] = smooth * cum_probs[c_i][inds] + (1 - smooth) * probs

            yield c_i, inds, probs, labels

