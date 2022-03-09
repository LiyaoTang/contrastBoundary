import time, os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf_batch_subsampling_module = tf.load_op_library(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tf_batch_subsampling.so'))
sess = tf.Session()

batch_size = 1
num_points = 100

pc = np.random.uniform(-1, 1, size=(batch_size, num_points, 3)).astype(np.float32).reshape(-1, 3)
batches_len = np.array([num_points] * batch_size)
sampleDl = 1

# batch grid sampling
pool_points, pool_stacks_lengths = tf_batch_subsampling_module.batch_grid_subsampling(tf.constant(pc, dtype=tf.float32), tf.constant(batches_len, dtype=tf.int32), sampleDl)
print(pool_points)
print(pool_stacks_lengths)

start = time.time()
rst = sess.run([pool_points, pool_stacks_lengths])
print(time.time() - start)

print('pc.shape =', pc.shape, 'batches_len =', batches_len, 'sampleDl =', sampleDl)
print(pc)
print(pc.min(axis=0), pc.max(axis=0))
print('after sampling')

print(rst[0].shape, rst[1].shape)
print(rst[0])
print(rst[1])
