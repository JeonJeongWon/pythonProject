import numpy as np
import tensorflow as tf
from variables import *

x = tf.random.normal(shape=(sys.batch_size,sys.time_step,sys.input_dim))

# data_map = np.array([-3-3j,-3-1j,-3+1j,-3+3j,
#                      -1-3j,-1-1j,-1+1j,-1+3j,
#                      +1-3j,+1-1j,+1+1j,+1+3j,
#                      +3-3j,+3-1j,+3+1j,+3+3j])
