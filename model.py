import tensorflow as tf
from variables import *

class model(tf.keras.Model):
    def __init__(self,**kwargs):
        super().__init__(self,**kwargs)
        self.__dict__.update(kwargs)
        self.pre_model = tf.keras.Sequential(self.pre_struc)
        self.tx_model = tf.keras.Sequential(self.tx_struc)

    def pre_cal(self,pre_in):
        return self.pre_model(pre_in)

    def tx_cal(self,tx_in):
        return self.tx_model(tx_in)

    def tx_loss(self,x):
        return tf.math.sqrt(tf.math.reduce_sum(tf.math.square(x-self.tx_cal(x)),axis=1))

    def gradient_tx(self,x):
        with tf.GradientTape() as tape:
            tx_loss = self.tx_loss(x)
        cg = tape.gradient(tx_loss,self.tx_model.trainable_variables)
        return cg,tx_loss


    def train_tx(self,x):
        cg, tx_loss = self.tx_loss(x)
        self.optimizer.apply_gradients(zip(cg, self.tx_model.trainable_variables))
        return tx_loss

    pre_struc = [
     tf.keras.layers.LSTM(nn.num_cells,return_sequences=True,input_shape=(sys.time_step,sys.data_dim)),
     tf.keras.layers.LSTM(nn.num_cells,return_sequences=True),
     tf.keras.layers.LSTM(nn.num_cells),
     tf.keras.layers.Dense(nn.num_cells*sys.data_dim)]

    tx_struc = [
     tf.keras.layers.LSTM(nn.num_cells,return_sequences=True,input_shape=(sys.time_step,sys.data_dim)),
     tf.keras.layers.LSTM(nn.num_cells,return_sequences=True),
     tf.keras.layers.LSTM(nn.num_cells),
     tf.keras.layers.Dense(nn.num_cells*sys.data_dim)]
