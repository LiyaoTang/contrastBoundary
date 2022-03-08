import numpy as np
import os, re, gc, sys, time, glob, shutil, pickle, argparse

from config.utils import log_config, load_config, is_val_success, _read_train

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('boolean value expected, given ', type(v))

def solve_envs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='1', help='the number/ID of GPUs to use [default: 1]')
    parser.add_argument('--data_path', type=str, default='Data', help='full path = data_path/dataset_name')
    parser.add_argument('--model_path', type=str, default='results', help='root dir to search models')
    parser.add_argument('--exclude', type=str, default='', help='regular expression to exclude dirs')
    parser.add_argument('--include', type=str, default='.*', help='regular expression to include dirs')
    parser.add_argument('--skip_exist', type=str2bool, default=True, help='if skip dir with existing validation result')
    parser.add_argument('--list_only', type=str2bool, default=True, help='only list out the target, no action taken')
    parser.add_argument('--save', type=str2bool, default=False, help='save the predicted point labels')
    parser.add_argument('--stop_on_err', type=str2bool, default=True, help='if stop running on error')
    parser.add_argument('--extra_ops', type=str, default='', help='extra ops to preform')
    parser.add_argument('--mode', type=str, default='val', help='choose one from val/test')
    parser.add_argument('--num_votes', type=float, default=None, help='stop criteria for voting')
    parser.add_argument('--batch_size', type=int, help='batch size to use')
    parser.add_argument('--step', type=str, default='1', help='num of saved steps to use')
    parser.add_argument('--snap_step', type=int, default=None, help='snapshot to be restored')
    parser.add_argument('--include_training', action='store_true', help='if include train log with no "finish" line')
    parser.add_argument('--train_thr', type=float, default=0, help='minimial training score to be considered for test/val')
    parser.add_argument('--eval_gap', type=int, default=10, help='the evaluation gap')
    parser.add_argument('--eval_sample', type=int, default=0, help='if evaluate sampled batch')
    parser.add_argument('--files', type=str, help='ckpt from files')
    parser.add_argument('--debug', action='store_true', help='if interactively debug result')
    parser.add_argument('--verbose', type=int, default=None, help='verbose level for printing info')
    parser.add_argument('--set', type=str, help='expect a dict to set the config')
    parser.add_argument('--check', type=str, default='validation_split,version', help='specific setting to check the config in train log')
    parser.add_argument('--order', type=str, default='cfg', help='order to evaluate')

    FLAGS = parser.parse_args()
    # sys.argv = sys.argv[:1]  # clean extra argv

    # solve args
    assert not FLAGS.num_votes or FLAGS.num_votes > 0
    assert FLAGS.data_path is not None
    assert os.path.isdir(FLAGS.model_path) and 'results' in FLAGS.model_path
    assert FLAGS.mode in ['val', 'test']
    FLAGS.stop_on_err = FLAGS.stop_on_err or FLAGS.debug
    FLAGS.verbose = FLAGS.verbose if FLAGS.verbose is not None else 2 if FLAGS.debug else 1  # 0=False, 1=True, 2=debug (print all)

    from config.base import Config
    FLAGS.gpu_devices = Config(vars(FLAGS), parse=False).gpu_devices
    return FLAGS

def clean(dir_p):
    assert os.path.isdir(dir_p)
    for d in ['validation']:
        p = os.path.join(dir_p, d)
        if os.path.isdir(p):
            shutil.rmtree(p)
    # for f in os.listdir(dir_p):
    #     f = os.path.join(dir_p, f)
    #     if os.path.isfile(f) and 'validation' in f:
    #         os.remove(f)

def is_test_success(f, step):
    if not os.path.isfile(f) or not re.search(f'.*log_test.*{step}', f):
        return False
    lines = open(f, 'r').read().strip('\n').split('\n')
    if not any([l.startswith('Done') for l in lines]):
        return False
    return True

def is_rst_success(f, step, split):
    if 'val' in split:
        return is_val_success(f, step)
    elif 'test' in split:
        return is_test_success(f, step)
    else:
        raise

def scan_ckpt(FLAGS):
    # scanning cfg & ckpt
    ckpt_list = []
    cfg_list = []
    for dir_p, _, files in os.walk(FLAGS.model_path):  # start at rst/dataset_name/cfg_name/Log_*/snapshots
        if not dir_p.endswith('snapshots') or re.fullmatch(FLAGS.exclude, dir_p) or not re.fullmatch(FLAGS.include, dir_p):  # select only trained file
            continue
        train_sc = _read_train(os.sep.join(dir_p.split(os.sep)[:-1]))[0]
        if train_sc and train_sc < FLAGS.train_thr:
            continue

        step_list = list(set([f.split('.')[0].split('-')[-1] for f in files if f.startswith('snap')]))
        step_other = [s for s in step_list if not s.isdigit()]
        step_list = sorted([int(s) for s in step_list if s.isdigit()]) + step_other
        if ',' in FLAGS.step:
            # '0,1,best' to select 3 snap
            select = [i.strip() for i in FLAGS.step.split(',') if i.strip()]
            select_i = [step_list[int(i)] for i in select if i.strip('-').isdigit() and int(i) < 99]  # select by idx - max idx = 99
            select_s = [i for i in select if not (i.strip('-').isdigit() and int(i) < 99)]
            select_s = sum([[s for s in step_list if i in str(s) and s not in select_i] for i in select_s], [])  # select by string
            step_list = select_i + select_s
        else:
            step_list = step_list[min([int(FLAGS.step), -int(FLAGS.step)]):]

        # print(step_list)
        for step in step_list:
            # glob to preserve full path
            existing_rst = [f for f in glob.glob(os.path.join(os.path.dirname(dir_p), '*')) if is_rst_success(f, step, FLAGS.mode)]
        
            if FLAGS.skip_exist and len(existing_rst) > 0:
                continue
            train_f = '/'.join(dir_p.split('/')[:-1] + ['log_train.txt'])
            if os.path.exists(train_f):    
                if not FLAGS.include_training and not any([l.startswith('finish') for l in open(train_f).read().split('\n')]):  # still in training
                    continue

            if FLAGS.snap_step == None or FLAGS.snap_step == step:
                ckpt_list.append(os.path.join(dir_p, f'snap-{step}'))
                cfg_path = dir_p.split('results')[-1].split('Log')[0].strip('/')
                cfg_list.append(cfg_path)

    key_fn = {
        'cfg': lambda t: t[1],
        'step': lambda t: -int(t[0].split('-')[-1]) if t[0].split('-')[-1].isdigit() else -float('inf'),
    }[FLAGS.order]
    rst_list = sorted(list(zip(ckpt_list, cfg_list)), key=key_fn)
    ckpt_list, cfg_list = zip(*rst_list)

    return ckpt_list, cfg_list

def scan_files(FLAGS):
    lines = open(FLAGS.files, 'r').read() if os.path.isfile(FLAGS.files) else FLAGS.files
    ckpt_list = [l.split()[0].strip(os.sep) for l in lines.split('\n') if l]  # should be results/datasets/cfg_name/Log*/snapshots/snap-###
    split = {'val': 'validation', 'test': 'test'}[FLAGS.mode]

    if FLAGS.skip_exist:
        f_list = [os.sep.join(ckpt.split(os.sep)[:-2] + [f'log_{split}.txt_' + ckpt.split('-')[-1].split('.')[0]]) for ckpt in ckpt_list]
        ckpt_list = [c for c, f in zip(ckpt_list, f_list) if not is_rst_success(f, '', split)]

    cfg_list = [os.sep.join(ckpt.split(os.sep)[1:3]) for ckpt in ckpt_list]
    return ckpt_list, cfg_list

def tf_eval(ckpt_path, cfg_path, FLAGS):
    """
    evaluate selected cfg & ckpt
    """

    # setup cfg from path
    dataset_name, cfg_name = cfg_path.split('/')
    cfg = load_config(dataset_name=dataset_name, cfg_name=cfg_name)
    cfg.saving_path = ckpt_path.split(cfg.snap_dir)[0].rstrip('/')
    cfg.model_path = ckpt_path
    cfg.mode = FLAGS.mode
    if FLAGS.batch_size:  # per-gpu batch
        cfg.batch_size = FLAGS.batch_size
    if FLAGS.num_votes:
        cfg.num_votes = FLAGS.num_votes
    cfg.extra_ops = FLAGS.extra_ops
    cfg.gpus = FLAGS.gpus
    cfg.debug = FLAGS.debug
    cfg = cfg.freeze()

    if FLAGS.set:
        for arg in FLAGS.set.split(';'):
            cfg.update(arg)

    if FLAGS.check:
        info = [f'checking - {FLAGS.check}']
        _, _, d = _read_train(cfg.saving_path, read_cfg=True)
        for arg in FLAGS.check.split(','):
            k = arg.strip()
            if d is not None and k in d:
                try:
                    v = eval(d[k])
                except:
                    v = d[k]
                setattr(cfg, k, v)
                info += [f'\t - {k} = {v}, {type(v)}']
        if FLAGS.verbose:
            print('\n'.join(info) + '\n')
        del info

    import tensorflow as tf
    from utils.logger import print_mem, redirect_io
    from utils.tester import ModelTester
    from utils.tf_graph_builder import GraphBuilder

    kwargs = {}
    chosen_step = ckpt_path.split('-')[-1]
    tester = ModelTester(cfg, verbose=FLAGS.verbose)
    if 'val' in cfg.mode:
        tester_func = tester.val_vote
        log_file = os.path.join(cfg.saving_path, f'log_validation.txt_{chosen_step}')
    elif 'test' in cfg.mode:
        tester_func = tester.test_vote
        log_file = os.path.join(cfg.saving_path, f'log_test.txt_{chosen_step}')
        kwargs = {
            'make_zip': True,
            'test_path': os.path.join(cfg.saving_path, f'test_{chosen_step}')}

    # eval with new graph
    with redirect_io(log_file, debug=FLAGS.debug):
        with tf.Graph().as_default():
            g = GraphBuilder(cfg, verbose=FLAGS.verbose > 1)
            g.restore(g.sess, cfg.model_path, select_list=['model/.*'], verbose=FLAGS.verbose > 1)

            def func():
                tester_func(g.sess, g.ops, g.dataset, g.model, num_votes=cfg.num_votes, **kwargs)

            log_config(cfg)

            eval_succ = 'success'
            if FLAGS.stop_on_err:
                func()
            else:
                try:
                    func()
                except:
                    eval_succ = 'failed'
                    pass
            g.sess.close()
        print(flush=True)

    print_mem(f'>>> finished {cfg.mode} -- {eval_succ} (gc: {gc.collect()})', check_time=True)
    
    if eval_succ != 'success':
        failed_tuple = (cfg.name, ckpt_path)
    else:
        failed_tuple = None
    return failed_tuple

def evaluate_path(FLAGS, verbose=True):
    ckpt_list, cfg_list = scan_files(FLAGS) if FLAGS.files else scan_ckpt(FLAGS)

    if verbose:
        print(f'In total, {len(ckpt_list)} targets:')
        max_len = max([len(i.split('/')[-1]) for i in cfg_list]) + 2
        for ckpt_path, cfg_path in zip(ckpt_list, cfg_list):
            cfg = cfg_path.split('/')[-1]
            fill = ' ' * (max_len-len(cfg))
            print(cfg + fill  + '/'.join(ckpt_path.split('/')[:-2]) + fill + ckpt_path.split('/')[-1], flush=True)
            # log_dir = '/'.join(ckpt_path.split('/')[:-2])
            # os.system(f'ls {log_dir}')
            # print(flush=True)

    if FLAGS.list_only:
        return

    # setup env
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_devices
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    failed_list = []

    start = time.time()
    for i, (ckpt_path, cfg_path) in enumerate(zip(ckpt_list, cfg_list)):
        if verbose:
            print(f'\n=> evaluating - {cfg_path}\t{ckpt_path}\n')
        failed_tuple = tf_eval(ckpt_path, cfg_path, FLAGS)
        if failed_tuple:
            failed_list.append(failed_tuple)
        if verbose:
            print(f'{i + 1}/{len(ckpt_list)} - {len(ckpt_list) - 1 - i} left, {len(failed_list)} failed')

    print(flush=True)
    if verbose:
        print('finish evaluation -- %d ckpts in %d min' % (len(ckpt_list), int((time.time()-start)/60)))
        for n, p in failed_list:
            print(f'failed: \n\tcfg.name = {n}\n\trestore_snap = {p}')

if __name__ == '__main__':
    FLAGS = solve_envs()
    evaluate_path(FLAGS)
