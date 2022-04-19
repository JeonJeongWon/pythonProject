import numpy as np
import tensorflow as tf
from variables import *


x = tf.random.normal(shape=(nn.batch_size,sys.time_step,sys.data_dim))

data = np.random.choice(sys.cons_map,nn.batch_size*sys.time_step,replace=True)
data_real = np.reshape(data.real,(nn.batch_size*sys.time_step,-1));data_img = np.reshape(data.imag,(nn.batch_size*sys.time_step,-1))
data_ = np.reshape(np.hstack((data_real,data_img)),(nn.batch_size,sys.time_step,sys.data_dim))

model_com = tf.keras.models.Sequential()
model_com.add(model_pre)
model_com.add(tf.keras.layers.Reshape((time_step,input_dim)))
model_com.add(model_tx)
model_com.summary()