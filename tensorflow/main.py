# Common libs
import numpy as np
import multiprocessing as mp
import os, sys, time, glob, pickle, psutil, argparse, importlib
sys.path.insert(0, f'{os.getcwd()}')

# Custom libs
from config import load_config, log_config
from utils.logger import print_mem, redirect_io
from config.utils import get_best_val_snap, get_snap

def get_last_train(cfg):
    saving_path = sorted(glob.glob(f'results/{cfg.dataset.lower()}/{cfg.name}/*'))
    return saving_path[-1] if saving_path else None

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cfg_path', type=str, help='config path')
parser.add_argument('--gpus', type=str, default=None, help='the number/ID of GPU(s) to use [default: 1], 0 to use cpu only')
parser.add_argument('--mode', type=str, default=None, help='options: train, val, test')
parser.add_argument('--data_path', type=str, default=None, help='path to dataset dir = data_path/dataset_name')
parser.add_argument('--model_path', type=str, default=None, help='pretrained model path')
parser.add_argument('--saving_path', type=str, default=None, help='specified saving path')
parser.add_argument('--num_votes', type=int, default=None, help='least num of votes of each point (default to 30)')
parser.add_argument('--num_threads', type=lambda n: mp.cpu_count() if n == 'a' else int(n) if n else None, default=None, help='the number of cpu to use for data loading')
parser.add_argument('--set', type=str, help='external source to set the config - str of dict / yaml file')
parser.add_argument('--debug', action='store_true', help='debug mode')
FLAGS = parser.parse_args()
# sys.argv = sys.argv[:1]  # clean extra argv

# ---------------------------------------------------------------------------- #
# solve env & cfg
# ---------------------------------------------------------------------------- #
assert FLAGS.cfg_path is not None

# load config - config path: config(dir).dataset_name(py).config_name(py_class)
cfg = load_config(cfg_path=FLAGS.cfg_path)

# update config
for arg in ['data_path', 'model_path', 'saving_path', 'mode', 'gpus', 'num_threads', 'num_votes', 'debug']:
    if getattr(FLAGS, arg) is not None:
        setattr(cfg, arg, getattr(FLAGS, arg))
if FLAGS.set:
    for arg in FLAGS.set.split(';'):
        cfg.update(arg)

# env setting: visible gpu, tf warnings (level = '0'/'3')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_devices
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import tensorflow as tf
if tf.__version__.split('.')[0] == '2':
    tf = tf.compat.v1
    tf.disable_v2_behavior()
import models, datasets
from utils.tester import ModelTester
from utils.trainer import ModelTrainer
from utils.tf_graph_builder import GraphBuilder

# solve config
if cfg.dataset in ['S3DIS']:
    cfg.mode = cfg.mode.replace('test', 'validation')
if cfg.model_path == 'auto':
    # latest train dir => then restored its latest snapshot
    cfg.model_path = get_last_train(cfg)
if cfg.model_path == 'best':
    # best val of current cfg
    cfg.model_path = get_best_val_snap(cfg)
if cfg.model_path and os.path.isdir(cfg.model_path):
    cfg.model_path = get_snap(cfg.model_path, step='last')
if cfg.save_memory:  # use gradient-checkpointing to save memory
    import utils.memory_saving_gradients
    tf.__dict__['gradients'] = utils.memory_saving_gradients.gradients_memory  # one from the: gradients_speed, gradients_memory, gradients_collection
if isinstance(cfg.rand_seed, int):  # manual set seed
    tf.set_random_seed(cfg.rand_seed)
    np.random.seed(cfg.rand_seed)

if cfg.debug:  # debug mode
    cfg.saving_path = 'test'
    cfg.log_file = sys.stdout

# ---------------------------------------------------------------------------- #
# training
# ---------------------------------------------------------------------------- #
if 'train' in cfg.mode:
    # result dir: results/dataset_name/config_name/Log_time/...
    if not cfg.saving_path:
        time.sleep(np.random.randint(1, 10))  # random sleep (avoid same log dir)
        # dataset_name = '_'.join([i for i in [cfg.dataset.lower(), cfg.version, cfg.validation_split] if i])  # default version / validation_split specified in dataset class
        cfg.saving_path = time.strftime(f'results/{cfg.dataset.lower()}/{cfg.name}/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
    os.makedirs(cfg.saving_path, exist_ok=True)
    if not cfg.log_file:
        cfg.log_file = os.path.join(cfg.saving_path, 'log_train.txt')
    if isinstance(cfg.log_file, str):
        cfg.log_file = open(cfg.log_file, 'w')
    log_config(cfg)
    log_config(cfg, f_out=cfg.log_file)

    # actual training
    print_mem('>>> start training', check_time=True)
    with redirect_io(cfg.log_file, cfg.debug):
        trainer = ModelTrainer(cfg)
        trainer.train()
        print(flush=True)
    print_mem('>>> finished training', check_time=True)


if cfg.gpu_num > 1:
    cfg.gpus = 1

if 'test' in cfg.mode or 'val' in cfg.mode:
    # find chosen snap (and saving_path if not specified)
    log_config(cfg)
    if cfg.model_path:  # specified for val/test
        chosen_snap = cfg.model_path
        cfg.saving_path = os.path.dirname(chosen_snap).split('snapshots')[0].rstrip('/')  # ensure at least is a dir
    elif cfg.saving_path:
        chosen_snap = get_snap(cfg.saving_path, 'best')
        if chosen_snap is None:
            chosen_snap = get_snap(cfg.saving_path, 'last')
    else:
        raise ValueError('provide either cfg.model_path (snap) or cfg.saving_path (dir)')
    assert len(glob.glob(chosen_snap + '*')) > 0 and os.path.isdir(cfg.saving_path), f'err path: chosen_snap = {chosen_snap}, saving_path = {cfg.saving_path}'
    chosen_step = chosen_snap.split('snap-')[-1].split('.')[0]

    # using the saved model
    print('using restored model, chosen_snap =', chosen_snap, flush=True)
    with tf.Graph().as_default():
        g = GraphBuilder(cfg)  # build fresh compute graph
        g.restore(restore_snap=chosen_snap, select_list=['model/.*'])
        tester = ModelTester(cfg)

        if 'val' in cfg.mode:
            log_file = os.path.join(cfg.saving_path, f'log_validation.txt_{chosen_step}')
            with redirect_io(log_file, cfg.debug):
                log_config(cfg)
                tester.val_vote(g.sess, g.ops, g.dataset, g.model, num_votes=cfg.num_votes)  # fresh voting
                print(flush=True)
            print_mem('>>> finished val', check_time=True)

        if 'test' in cfg.mode:
            log_file = os.path.join(cfg.saving_path, f'log_test.txt_{chosen_step}')
            test_path = os.path.join(cfg.saving_path, f'test_{chosen_step}')
            with redirect_io(log_file, cfg.debug):
                log_config(cfg)
                tester.test_vote(g.sess, g.ops, g.dataset, g.model, num_votes=cfg.num_votes, test_path=test_path)
                print(flush=True)
            print_mem('>>> finished test', check_time=True)
