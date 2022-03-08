import os, re, sys, glob, types, itertools
import numpy as np
from collections import defaultdict

def _xor(a, b):
    return not ((a and b) or (not a and not b))
def _is_property(obj, name):
    return isinstance(getattr(type(obj), name, None), property)
def _is_method(x):
    return type(x) in [types.MethodType, types.FunctionType]
def _is_float(x):
    try:
        float(x)
        return True
    except:
        pass
    return False
def _raise(ex):
    raise ex

def gen_config(cfg_gen, store_dict, sep='-'):
    """
    add instance of config generator class to config file 
    Args:
        cfg_gen     : py class  - the generator class, whose `_attr_dict` contains config attributes and the possible values
        store_dict  : dict      - the place to put the generated config instance, may use `globals()`
        sep         : str       - seprator in composite attributes, default to `-`
    """
    if isinstance(cfg_gen, (list, tuple)):
        for cg in cfg_gen:
            gen_config(cg, store_dict, sep)
        return

    assert hasattr(cfg_gen, 'idx_name_pre'), f'no idx_name_pre provided in {cfg_gen}'
    attr_dict = cfg_gen._attr_dict.copy()  # not altering the class variable

    for k, v in attr_dict.items():  # scan for composite config
        assert len(v) > 0, f'found empty list of config options: _attr_dict[{k}]'
        if type(v[0]) == list:
            attr_dict[k] = [sep.join([str(i) for i in v_list if str(i)]).strip(sep) for v_list in itertools.product(*v)]

    attr_k = attr_dict.keys()
    attr_v = [attr_dict[k] for k in attr_k]
    attr_i = [list(range(len(v))) for v in attr_v]

    for idx in itertools.product(*attr_i):
        cfg = cfg_gen(parse=False)
        for i, k, v in zip(idx, attr_k, attr_v):
            setattr(cfg, k, v[i])
        cfg_var_name = cfg.idx_name_pre + '_' + ''.join([str(i) for i in idx])  # use index to attribute value as name postfix
        setattr(cfg, 'idx_name', cfg_var_name)
        cfg.parse()  # static parse after setting attrs
        store_dict[cfg_var_name] = cfg

def is_config(cfg, base=None, mod=None):
    if mod != None and type(cfg) == str:
        if cfg.startswith('_'):
            return False
        cfg = getattr(mod, cfg)
    if base == None:
        assert mod != None, 'must provide either `base` (class Base) or `mod` (python module)'
        base = mod.Base
    return isinstance(cfg, base) or isinstance(type(cfg), base)  # config can be either class or instance

def log_config(config, title='', f_out=None, prefix='', base=None):
    if f_out is None:
        f_out = sys.stdout
    if base is None:
        root = os.path.join(os.getcwd(), os.path.dirname(__file__), '../')
        sys.path += [] if root in sys.path or os.path.realpath(root) in sys.path else [root]
        from config.base import Base as base

    print(f'\n{prefix}<<< ======= {config._cls} ======= {title if title else config.name}', file=f_out)
    max_len = max([len(k) for k in dir(config) if not k.startswith('_')] + [0])
    for k in dir(config):
        if k.startswith('_') or _is_method(getattr(config, k)):
            continue
        cur_attr = getattr(config, k)
        if isinstance(cur_attr, list) and len(str(cur_attr)) > 200:  # overlong list
            cur_attr = '[' + f'\n{prefix}\t\t'.join([''] + [str(s) for s in cur_attr]) + f'\n{prefix}\t]'

        print('\t%s%s\t= %s' % (prefix + k, ' ' * (max_len-len(k)), str(cur_attr)), file=f_out)
        if is_config(cur_attr, base=base):
            log_config(cur_attr, f_out=f_out, prefix=prefix+'\t', base=base)
    print('\n', file=f_out, flush=True)

def load_config(cfg_path=None, dataset_name=None, cfg_name=None, cfg_group=None, reload=True):
    import importlib
    
    # cfg from path
    if cfg_path is not None:
        update = None
        if os.path.isfile(cfg_path):
            # update on the default cfg
            from config.base import Base, Config
            update = Base(cfg_path)
            cfg_path = [update.dataset.lower(), 'default']
        else:
            # directly specified cfg
            cfg_path = cfg_path.replace('/', '.').split('.')
        cfg_path = cfg_path if cfg_path[0] == 'config' else ['config'] + cfg_path
        cfg_module = '.'.join(cfg_path[:2])
        cfg_class = '.'.join(cfg_path[2:])
        mod = importlib.import_module(cfg_module)
        if hasattr(mod, cfg_class):
            cfg = getattr(mod, cfg_class)
        else:
            cfg = load_config(dataset_name=cfg_path[1], cfg_name=cfg_class, reload=reload)

        if update is not None:
            cfg = Config(cfg)  # avoid overriding
            cfg.update(update, exclude=[])  # full override with no exclude
        return cfg

    # setup dict
    cfg_name_dict   = load_config.cfg_name_dict    # dataset_name -> {cfg.name -> cfg.idx_name}
    cfg_module_dict = load_config.cfg_module_dict  # dataset_name -> cfg_module

    if dataset_name is not None and dataset_name not in cfg_module_dict or reload:
        mod = importlib.import_module('config.' + dataset_name)
        cfg_module_dict[dataset_name] = mod
        cfg_name_dict[dataset_name] = {}
        for i in dir(mod):
            if not is_config(i, mod=mod):  # use the 'base' class imported in 'mod'
                continue
            cfg = getattr(mod, i)
            if cfg.name:
                cfg_name_dict[dataset_name][cfg.name] = cfg.idx_name

    # module/cfg from dataset/cfg name
    mod = cfg_module_dict[dataset_name]
    if cfg_name is not None:
        if cfg_name not in cfg_name_dict[dataset_name]:
            raise KeyError(f'no cfg_name={cfg_name} in module {dataset_name}')
        idx_name = cfg_name_dict[dataset_name][cfg_name]
        return getattr(mod, idx_name)
    elif cfg_group is not None:
        if not hasattr(mod, cfg_group):
            raise KeyError(f'no cfg_group={cfg_group} in module {dataset_name}')
        cfg_g = getattr(mod, cfg_group)
        if not isinstance(cfg_g, (tuple, list, dict, set)):
            raise ValueError(f'cfg_group={cfg_group} appears to be {cfg_g}, not of type (tuple, list, dict, set)')
        return cfg_g
    return mod
load_config.cfg_module_dict = {}
load_config.cfg_name_dict = {}

def is_train_success(train_dir):
    # train_dir = results/dataset_name/config_name/Log_*
    # if not 'snapshots' in os.listdir(train_dir) or len(os.listdir(os.path.join(train_dir, 'snapshots'))) == 0:  # snapshots
    #     return False
    log_train = os.path.join(train_dir, 'log_train.txt')  # train_log
    if not os.path.isfile(log_train):
        return False
    # avoid in-training log
    lines = [l for l in open(log_train, 'r').read().strip('\n').split('\n') if l]
    if any([l.startswith('finish') for l in lines]):  # train_log 'finish' print
        return True
    elif any(['Traceback (most recent call last):' in l for l in lines]):  # error occured (ign error after train finish)
        return False
    elif len(lines) == 0:
        return False
    return True  # in-training

def _read_cfg(lines):
    idx_list = []
    for i, l in enumerate(lines):
        if l.startswith('<<<') or 'EPOCH' in l: idx_list.append(i)
        if len(idx_list) == 2: break
    if len(idx_list) == 0:
        return None

    idx_list = idx_list if len(idx_list) == 2 else [*idx_list, None]
    lines = [l for l in lines[idx_list[0] + 1:idx_list[1]] if '= ' in l and l.count('=') == 1]

    num_blanks = [re.search('^[ \t]+', l) for l in lines]
    num_blanks = [len(n.group()) for n in num_blanks if n]
    num_outter = min(num_blanks)
    outter = [l.split('= ') for n, l in zip(num_blanks, lines) if n == num_outter]  # consider only out-most config i.e. the `config`
    cfg = defaultdict(lambda: '', {k.strip():v.strip() for k, v in outter})
    return cfg

def _read_train(train_dir, read_cfg=False):
    log_train = os.path.join(train_dir, 'log_train.txt')  # train_log
    if not os.path.isfile(log_train):
        return (None, None, None) if read_cfg else (None, None)
    lines = open(log_train, 'r').read().strip('\n').split('\n')
    sc = [float(l.split('|')[0].split('=')[1].split()[0]) for l in lines if '|' in l and 'current' in l]
    sc = max(sc) if sc else None
    task_loss = np.array([[float(i.split('=')[-1]) for i in l.split() if '=' in i] for l in lines[-100:] if l.startswith('Step ')])
    task_loss = None if not len(task_loss) else task_loss[:, 1].mean() if not np.isnan(task_loss).any() else float('nan')

    if read_cfg:
        cfg = _read_cfg(lines)
        return sc, task_loss, cfg
    return sc, task_loss

def is_val_success(f, step=None):
    step = str(step) if step != None else ''
    n = f.split('/')[-1]
    if not os.path.isfile(f) or not n.startswith('log_val') or not n.endswith(step):
        return False
    f = open(f, 'r').read().split('\n')
    if any([l.startswith('finish') for l in f]):
        return True
    return False

def _read_val(f, full=True, get_dict=False, get_conf=False):
    if not is_val_success(f):
        return None
    lines = open(f, 'r').read().split('\n')
    kv_dict = {
        'OA': None,
        'mACC': None, 'ACCs': None,
        'mIoU': None, 'IoUs': None,
        'conf': None,
    }

    # print(f)
    if full:  # get result on full cloud
        f = [i for i, l in enumerate(lines) if set(lines[i]) == set('-')]
        if len(f) < 2:
            return None
        kv_dict = {}
        for l in lines[f[-2] + 1:f[-1]]:
            l = l.split('|')
            h = l[0]
            for k in ['OA', 'mACC', 'mIoU']:
                if k not in h: continue
                v = h.split(k + '=')[-1].split()[0]
                kv_dict[k] = float(v)
            if len(l) > 1:
                kv_dict[k.lstrip('m') + 's'] = [float(i) for i in (l[1]).split() if i]

        if get_conf:
            idx = [i for i, l in enumerate(lines) if 'final' in l and 'Confusion' in l]
            idx = idx[-1] if idx else f[-2]
            conf = []
            for l in lines[idx + 1:f[-2]]:
                conf.append([float(i) for i in l.strip().strip('[]').split() if i])
            conf = np.array(conf) if conf else None
            kv_dict['conf'] = conf

    else:  # result on sub cloud
        f = [i for i, l in enumerate(lines) if 'sub clouds - final' in l]
        if not f:
            return None
        idx = f[-1]
        l_m = lines[idx].split('|')[0].split(':')[-1].strip().split()
        keys = [kv.split('=')[0].strip() for kv in l_m]
        v_metrics = [float(kv.split('=')[1].strip()) for kv in l_m]

        mIoU, OA, mACC = None, None, None
        if 'mIoU' in keys:
            kv_dict['mIoU'] = v_metrics[keys.index('mIoU')]
        if 'OA' in keys:
            kv_dict['OA'] = v_metrics[keys.index('OA')]
        if 'mACC' in keys:
            kv_dict['mACC'] = v_metrics[keys.index('mACC')]

        if '|' in lines[idx]:
            kv_dict['IoUs'] = [float(i) for i in lines[idx].split('|')[-1].split()]

        acc = None
        if lines[idx - 2].startswith('ACCs'):
            kv_dict['ACCs'] = [float(i) for i in lines[idx - 2].split('=')[-1].split('|')[-1].split()]

    if get_dict:  # get doct ailgns with `class Metrics`
        return kv_dict

    return kv_dict['mIoU'], kv_dict['OA'], kv_dict['mACC'], kv_dict['IoUs']

def _read_test(f):
    lines = [l for l in open(f, 'r').read().split('\n') if 'Result' in l]
    if not lines:
        return None
    rst = float(lines[-1].split('Result')[-1].split()[0])
    return rst if rst > 1 else rst * 100

def get_best_val_snap(cfg, snap_prefix='snap'):
    # get the best of full validation
    cfg_path = f'results/{cfg.dataset.lower()}/{cfg.name}'
    vals = glob.glob(f'{cfg_path}/*/log_val*')
    if not vals:
        return None
    vals = list(zip(vals, [_read_val(f) for f in vals]))  # [(.../log_v, mIoU), ...]
    vals = [(vf, v[0]) for vf, v in vals if v is not None]
    vals = sorted(vals, key= lambda t: t[1] if t[1] else -1)[-1]
    if not vals[1]:
        return None
    snap = int(vals[0].split('_')[-1])
    log_path = os.path.dirname(vals[0])
    snap_path = f'{log_path}/{cfg.snap_dir}/{snap_prefix}-{snap}'
    return snap_path

def get_snap(saving_path, step='last', snap_prefix='snap'):
    # get the best of running val (done in training)
    snap_path = os.path.join(saving_path, 'snapshots') if not saving_path.endswith('snapshots') else saving_path
    snap_steps = [f[:-5].split('-')[-1] for f in os.listdir(snap_path) if f[-5:] == '.meta']
    if step == 'last':
        snap_steps = [int(s) for s in snap_steps if s.isdigit()]
        chosen_step = np.sort(snap_steps)[-1]  # last saved snap (best val estimation)
        chosen_snap = os.path.join(snap_path, f'snap-{chosen_step}')
    else:
        assert isinstance(step, int) or step.isdigit() or step == 'best', f'not supported step = {step}'
        step = str(step)
        chosen_snap = None
        if step in snap_steps:
            chosen_snap = os.path.join(snap_path, f'snap-{step}')
    return chosen_snap

def to_valid_stage(stage_n, short=False):
    if stage_n in ['D', 'down']:
        stage_n = 'D' if short else 'down'
    elif stage_n in ['U', 'up']:
        stage_n = 'U' if short else 'up'
    else:
        raise ValueError(f'invalid stage_n={stage_n}')
    return stage_n

def parse_stage(stage, num_layers):
    stage = stage.replace('a', ''.join(f'{i}' for i in range(num_layers)))
    stage_list = [i.strip('_') for i in re.split('(\d+)', stage) if i and i.strip('_')]  # e.g. D012_U34
    assert len(stage_list) % 2 == 0, f'invalid stage compound: stage_list={stage_list} from stage={stage}'
    stage_n = [s for i, s in enumerate(stage_list) if i % 2 == 0]
    stage_i = [s for i, s in enumerate(stage_list) if i % 2 == 1]
    stage_list = [[(to_valid_stage(n), int(i)) for i in i_str] for n, i_str in zip(stage_n, stage_i)]
    stage_list = sum(stage_list, [])
    return stage_list


def _str2bool(v, raise_not_support=True):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    elif raise_not_support:
        raise argparse.ArgumentTypeError('boolean value expected, given ', type(v))
    else:
        return None

if __name__ == '__main__':
    import numpy as np
    import pickle, argparse, time, sys, os, re, glob, shutil
    sys.path.insert(0, os.path.join(os.getcwd(), os.path.dirname(__file__), '../'))

    parser = argparse.ArgumentParser()
    parser.add_argument('--list', type=str, help='print the cfg group (file_name, or file_name.dict_name)')
    parser.add_argument('--best', type=str, default=None, help='get config best val')
    FLAGS = parser.parse_args()

    cfg_dir = os.path.join(os.getcwd(), os.path.dirname(__file__)).rstrip('/')
    sys.path.insert(0, os.path.dirname(cfg_dir))

    dir_list = None
    if FLAGS.list:
        _list_config(FLAGS)
    if FLAGS.best:
        print(get_best_val_snap(load_config(FLAGS.best)))



