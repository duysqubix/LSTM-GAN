import tensorflow.keras.backend as K
import tensorflow as tf


class Reshape(tf.keras.layers.Reshape):
    def call(self, inputs):
        return K.reshape(inputs, self.target_shape)


if __name__ == '__main__':
    import numpy as np
    x = np.random.randn(840, 1)
    reshape = Reshape(target_shape=(28, 30, 1))
    print(reshape.__call__(x).shape)
