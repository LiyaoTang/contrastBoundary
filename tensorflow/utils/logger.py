# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import functools
import logging
import os
import sys
from termcolor import colored


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


# so that calling setup_logger multiple times won't add many handlers
@functools.lru_cache()
def setup_logger(
    output=None, distributed_rank=0, *, color=True, name="", abbrev_name=None
):
    """
    Initialize the detectron2 logger and set its verbosity level to "INFO".

    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger

    Returns:
        logging.Logger: a logger
    """
    logger = logging.getLogger(name)  # a global named logger
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = name

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    # stdout logging: master only
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = filename + f".rank{distributed_rank}"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return open(filename, "a")


# ----------------------------------------------------------------------------------------------------------
# functions for directly printing
# ----------------------------------------------------------------------------------------------------------


import os, sys, time, psutil, traceback
import subprocess as sp
import numpy as np

def print_mem(prefix, gpu=True, check_time=False, check_sys=False, **kwargs):
    sep = '\n\t' if any([gpu, check_time]) else ' '
    lines = [prefix, 'Mem Comsumption: %.2f GB' % (print_mem.process.memory_info()[0] / float(2**30))]
    if check_sys:
        sysmem = psutil.virtual_memory()
        lines += [f'Mem in sys: avail {sysmem.available / 2**30:.2f} / total {sysmem.total / 2**30:.2f}']
    if gpu:
        try:
            gpu_mem = get_gpu_mem()
            lines += [f'Availabel Mem of each GPU: {gpu_mem}']
        except FileNotFoundError:
            pass
        except sp.CalledProcessError:
            pass
    if check_time:
        cur_t = time.time()
        if not hasattr(print_mem, 't_start'):
            print_mem.t_start = cur_t
            print_mem.t = cur_t
        else:
            gap = int(cur_t-print_mem.t)
            cum = int(cur_t-print_mem.t_start)
            lines += [f'time used [gap/cum] : {gap // 60}min {gap % 60}s / {cum // 60}min {cum % 60}s']
            print_mem.t = cur_t
    print(sep.join(lines), **kwargs)
print_mem.process = psutil.Process(os.getpid())


def get_gpu_mem():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

def print_dict(d, prefix='', except_k=[], fn=None, head=None):
    if head is not None:
        d = {head: d}
    for k, v in d.items():
        if k in except_k:
            continue
        if isinstance(d[k], dict):
            print(f'{prefix}{str(k)}:')
            print_dict(d[k], prefix=f'{prefix}\t', except_k=except_k, fn=fn)
        else:
            if fn:
                rst = None
                try:
                    if isinstance(v, (list, tuple)):
                        rst = v.__class__([fn(vv) for vv in v])
                    else:
                        rst = fn(v)
                except:
                    pass
                v = rst if rst else v
            line = f'{prefix}{str(k)}\t{str(v)}'
            if isinstance(v, (list, tuple)) and len(str(line)) > 100:  # overlong
                line_pre = f'{prefix}{str(k)}\t' + ('[' if isinstance(v, list) else '(')
                line_post = f'\n{prefix}\t' + (']' if isinstance(v, list) else ')')
                if set([type(s) for s in v]) == set([dict]):  # all dict in list
                    print(line_pre)
                    for s in v[:-1]:
                        print_dict(s, prefix=f'{prefix}\t\t')
                        print(f'{prefix}\t\t,')
                    print_dict(v[-1], prefix=f'{prefix}\t\t')
                    line = line_post
                else:
                    line =  line_pre + f'\n{prefix}\t\t'.join([''] + [str(s) for s in v]) + line_post

            print(line)

def print_table(t, prefix='', sep='  '):  # assume a 2D-list
    max_len = np.array([[len(str(ii)) for ii in l] for l in t], dtype=int).max(axis=0)
    for line in t:
        print(prefix + sep.join([str(ii) + ' ' * (max_len[i] - len(str(ii))) for i, ii in enumerate(line)]))

def log_percentage(arr, precision=0):
    length = precision + 3 if precision else 2
    if len(arr) == 0:
        return ''
    if type(arr) == list:
        arr = np.array(arr)
    arr = arr / arr.sum(axis=0) * 100  # vertical sum
    if len(arr.shape) == 1:
        arr = np.expand_dims(arr, axis=0)
    arr = arr.T
    str_list = []
    for row in arr:
        num_list = [f'%{length}.{precision}f' % i for i in row]
        str_list.append('/'.join(num_list))
    return ' '.join(str_list)

class redirect_io(object):
    def __init__(self, log_file, debug):
        self.log_file = log_file
        self.debug = debug
    def __enter__(self):
        if self.debug:
            return
        self.log_file = open(self.log_file, 'w') if isinstance(self.log_file, str) else self.log_file
        self.stdout, self.stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = self.log_file
  
    def __exit__(self, exc_type, exc_value, tb):
        if self.debug:
            return
        traceback.print_exc()
        self.log_file.close()
        sys.stdout, sys.stderr = self.stdout, self.stderr
