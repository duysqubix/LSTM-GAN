
# FIND WHY INPUT OF GAN MODEL EXPECTS 30, 5 WHEN YOU DYNAMICALLY CHANGE INPUT SHAPE SIZE


import numpy as np
import tensorflow as tf

from layers.lstm import LSTM
from layers.reshape import Reshape

class DataGenerator:
    def __init__(self, datasource, data_length=500):
        self.data_length = 500
        self.datasource = datasource
    
    def create_sine_wave(self, n, time_steps):
        samples = []
        ix = np.arange(n) + 1
        f = 100 #np.random.uniform(low=5, high=0)
        A = np.random.uniform(low=0.9, high=0.8999, size=n)
        offset = 0# np.random.uniform(low=-np.pi, high=np.pi, size=n)
        sine_wave = A*np.sin(2*np.pi*f*ix/float(n) + offset)

        sample_size = n // time_steps

        for i in range(0, n, time_steps):
            samples.append(sine_wave[i:i+time_steps])
        
        samples = np.array(samples)
        return samples.reshape(sample_size, time_steps, 1)

    def sine_wave(self, seq_length=30, num_samples=28*5*100, num_signals=1):
        ix = np.arange(seq_length) + 1
        samples = []
        for i in range(num_samples):
            signals = []
            for i in range(num_signals):
                f = np.random.uniform(low=1, high=5)
                A = np.random.uniform(low=0.1, high=0.9)
                offset = np.random.uniform(low=-np.pi, high=np.pi)
                signals.append(A*np.sin(2*np.pi*f*ix/np.float32(seq_length)+offset))
            samples.append(np.array(signals).T)
        samples = np.array(samples)
        return samples




class Generator:
    def __init__(self, latent_dim=5, seq_length=30, batch_size=28, hidden_size=100, num_generated_features=1):
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_generated_features = num_generated_features

        # self.model = tf.keras.models.Sequential([
        #     LSTM(self.hidden_size, input_shape=(self.seq_length, self.latent_dim), return_sequences=True),
        #     tf.keras.layers.Dense(1, input_shape=[None, self.hidden_size]),
        #     tf.keras.layers.Activation('tanh'),
        #     Reshape(target_shape=(self.batch_size, self.seq_length, self.num_generated_features))
        # ])
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(self.hidden_size, input_shape=(self.seq_length, self.latent_dim), return_sequences=True, name='g_lstm1'),
            tf.keras.layers.LSTM(self.hidden_size, return_sequences=True, recurrent_dropout=0.4, name='g_lstm2'),
            tf.keras.layers.LSTM(1, return_sequences=True, name='g_lstm3')
        ], name='generator')

        self.data_gen = DataGenerator(None)
        self.all_real_samples = self.data_gen.sine_wave()

    def real_samples(self):
        #         x =  self.log_generator.create_samples(n=n, steps_in=self.latent_dim)
        # x = self.data_gen.create_sine_wave(n=n, time_steps=time_steps)

        x = self.all_real_samples[np.random.choice(self.all_real_samples.shape[0], self.batch_size, 1), :]
        # y = np.ones((len(x), 1))
        y = np.ones_like(x)
        return x, y

    def fake_samples(self):
        x = self.sample_latent_space()
        x = self.model.predict(x, batch_size=self.batch_size)

        y = np.zeros_like(x)
        return x, y

    def sample_latent_space(self, n=None):
        #         x = np.random.randn(self.latent_dim * n)
        #         return x.reshape(n, self.latent_dim)

        return np.random.randn(self.batch_size, self.seq_length, self.latent_dim)


class Discriminator:
    def __init__(self, input_shape, hidden_size=100):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(hidden_size, input_shape=input_shape, return_sequences=True, name='d_lstm'),
            tf.keras.layers.LSTM(hidden_size, return_sequences=True, name='d_lstm2', recurrent_dropout=0.4),
            tf.keras.layers.Dense(1, activation='linear', name='d_output')
        ], name='discriminator')


        self.model.compile(loss=self.d_loss, optimizer=tf.keras.optimizers.SGD(lr=0.1), metrics=['acc'])

    def d_loss(self, y_true, y_pred):
        loss = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=True)
        return loss

class GAN:
    real_loss = []
    fake_loss = []
    def __init__(self, *args, **kwargs):

        self.generator = Generator(*args, **kwargs)
        gen_output = (self.generator.seq_length, self.generator.num_generated_features)
        self.discriminator = Discriminator(input_shape=gen_output)
        self.discriminator.model.trainable = False

        self.batch_size = self.generator.batch_size
        self.seq_length = self.generator.seq_length


        self.model = tf.keras.models.Sequential([
            self.generator.model,
            self.discriminator.model
        ], name='gan')

        self.model.compile(loss=self.gan_loss, optimizer=tf.keras.optimizers.SGD(lr=0.1), metrics=['acc'])

    def gan_loss(self, y_true, y_pred):
        loss = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=True)
        return loss

    def train(self, epochs, n_eval, d_train_steps=5):
        steps_over_data = len(self.generator.all_real_samples)//self.generator.batch_size
        print("Begin Training with {} steps per epoch for every {} trains on discriminator".format(steps_over_data, d_train_steps))
        for epoch in range(epochs+1):
 
            for step in range(steps_over_data):
                tmp_r, tmp_f = [], []
                for _ in range(d_train_steps):

                    x_r, y_r = self.generator.real_samples()
                    x_f, y_f = self.generator.fake_samples()

                    real = self.discriminator.model.fit(x_r, y_r, epochs=1, batch_size=self.batch_size, verbose=0, shuffle=True).history
                    fake = self.discriminator.model.fit(x_f, y_f, epochs=1, batch_size=self.batch_size, verbose=0, shuffle=True).history

                    tmp_r.append(real['loss'])
                    tmp_f.append(fake['loss'])

            self.real_loss.append(np.mean(tmp_r))
            self.fake_loss.append(np.mean(tmp_f))

            x_gan = self.generator.sample_latent_space()
            y_gan = np.ones((self.batch_size, self.seq_length, self.generator.num_generated_features)).astype(np.float32)

            self.model.fit(x_gan, y_gan, batch_size=self.batch_size, epochs=1, verbose=0)

            if epoch % n_eval == 0:

                print("Epoch: {}/{} Real Loss: {}\tFake Loss: {}".format(epoch, epochs, self.real_loss[-1], self.fake_loss[-1]))




if __name__ == '__main__':
    gan = GAN(latent_dim=5, seq_length=30, batch_size=28)
    gan.model.summary()

    gan.train(epochs=5, n_eval=1)

    # batch_size = 3000 
    # seq_length = 30
    # latent_dim = 5
    # gen = Generator(latent_dim=latent_dim, seq_length=seq_length, batch_size=batch_size, hidden_size=100, num_generated_features=1)
    # # tf.keras.utils.plot_model(gen.model, show_shapes=True)
    # # print(gen.model.predict(tf.random.normal((3000, 30, 5))).shape)
    # # print(gen.model.layers[0])
    # x = tf.random.normal((batch_size, seq_length, latent_dim))

    # for layer in gen.model.layers:
    #     x = layer(x)
    #     print(x.shape)

    # x = tf.random.normal((batch_size, seq_length, latent_dim))

    # print(gen.model.predict(x, batch_size=batch_size).shape)

    # # dis = Discriminator(input_shape=(30, 100))
    # # dis.model.summary()
    # # print(dis.model.predict(tf.random.normal((10000, 30, 1))).shape)

    # # dg = DataGenerator(None)
    # # samples = dg.create_sine_wave(n=10, time_steps=5)
    # # print(samples.shape)

    # # gan = GAN()
    # # # print(samples)

