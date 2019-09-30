import tensorflow as tf
import tensorflow.keras.backend as K


class LSTM(tf.keras.layers.LSTM):
    
    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None

        a = super(LSTM, self).call(inputs,
                                      mask=mask,
                                      training=training,
                                      initial_state=initial_state)
        lstm_outputs_2d = K.reshape(a, [-1, a.shape[-1]])
        return lstm_outputs_2d
    
    def get_output_shape(self, input_shape):
        return input_shape[0]*input_shape[1], input_shape[2]


if __name__ == '__main__':
    import numpy as np
    print(tf.__version__)
    x = np.random.randn(28, 30, 5).astype(np.float32)
    lstm = LSTM(100, input_shape=(30, 5), return_sequences=True)

    #output shape: [28*30=840, 100]
    print(lstm(x).shape)
