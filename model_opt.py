
# FIND WHY INPUT OF GAN MODEL EXPECTS 30, 5 WHEN YOU DYNAMICALLY CHANGE INPUT SHAPE SIZE


import numpy as np
import tensorflow as tf


class DataGenerator:
    @staticmethod
    def sine_wave(seq_length=30, num_samples=28*5*100, num_signals=1):
        ix = np.arange(seq_length) + 1
        samples = []
        for i in range(num_samples):
            signals = []
            for i in range(num_signals):
                f = np.random.uniform(low=1, high=5)
                A = np.random.uniform(low=0.1, high=0.9)
                offset = np.random.uniform(low=-np.pi, high=np.pi)
                signals.append(
                    A*np.sin(2*np.pi*f*ix/np.float32(seq_length)+offset))
                # A*np.sin(2*np.pi*f*np.float32(ix)+offset))
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
            tf.keras.layers.LSTM(self.hidden_size,
                                 input_shape=(self.seq_length,
                                              self.latent_dim),
                                 return_sequences=True,
                                 name='g_lstm1'),

            tf.keras.layers.LSTM(self.hidden_size,
                                 return_sequences=True,
                                 recurrent_dropout=0.4,
                                 name='g_lstm2'),

            tf.keras.layers.LSTM(1, return_sequences=True, name='g_lstm3')
        ], name='generator')

        self.data_gen = DataGenerator
        self.all_real_samples = self.data_gen.sine_wave(seq_length=self.seq_length,
                                                        num_samples=self.batch_size*self.hidden_size*self.latent_dim)

        self.ground_truth_valid = tf.ones((self.batch_size,
                                           self.seq_length,
                                           self.num_generated_features))

        self.ground_truth_fake = tf.zeros_like(self.ground_truth_valid)

        self.real_dataset = tf.data.Dataset.from_tensor_slices(self.all_real_samples).batch(
            self.batch_size, drop_remainder=True).shuffle(buffer_size=len(self.all_real_samples))

        self.fake_dataset = tf.data.Dataset.from_generator(
            self.sample_latent_space, tf.float32).batch(self.batch_size, drop_remainder=True)

    def real_samples(self):
        # depreciated

        return self.all_real_samples[
            np.random.choice(self.all_real_samples.shape[0],
                             self.batch_size, 1),
            :]

    def fake_samples(self):
        # depreciated
        x = self.model.predict(self.sample_latent_space(),
                               batch_size=self.batch_size)
        return x

    def sample_latent_space(self, n=None):
        while 1:
            yield tf.random.normal((self.seq_length, self.latent_dim))


class Discriminator:
    def __init__(self, input_shape, hidden_size=100):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(hidden_size,
                                 input_shape=input_shape,
                                 return_sequences=True, name='d_lstm'),

            tf.keras.layers.LSTM(hidden_size,
                                 return_sequences=True,
                                 name='d_lstm2',
                                 recurrent_dropout=0.4,
                                 dropout=0.4),

            tf.keras.layers.Dense(1, activation='sigmoid', name='d_output')
        ], name='discriminator')

        self.model.compile(loss='binary_crossentropy',
                           optimizer=tf.keras.optimizers.Adam(
                               learning_rate=0.0002, beta_1=0.5),
                           metrics=['acc'])


class GAN:
    d_loss = list()

    def __init__(self, *args, **kwargs):

        self.generator = Generator(*args, **kwargs)
        gen_output = (self.generator.seq_length,
                      self.generator.num_generated_features)
        self.discriminator = Discriminator(input_shape=gen_output)
        self.discriminator.model.trainable = False

        self.batch_size = self.generator.batch_size
        self.seq_length = self.generator.seq_length

        self.model = tf.keras.models.Sequential([
            self.generator.model,
            self.discriminator.model
        ], name='gan')

        self.model.compile(
            loss=self.gan_loss,
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))

    def gan_loss(self, y_true, y_pred):
        loss = tf.keras.losses.binary_crossentropy(
            y_true, y_pred, from_logits=True)
        return loss

    def load_weights(self, names=None):
        assert isinstance(names, (tuple, list))

        try:
            self.generator.model.load_weights("{}_weights.h5".format(names[0]))
            self.discriminator.model.load_weights(
                "{}_weights.h5".format(names[1]))
        except FileNotFoundError:
            print("Could not find weight files, skipping...")

    def save_weights(self, names=None):
        assert isinstance(names, (tuple, list))
        self.generator.model.save_weights("{}_weights.h5".format(names[0]))
        self.discriminator.model.save_weights("{}_weights.h5".format(names[1]))

    def plot_preds(self, epoch=None, save_plot=False, rows=2, cols=4):
        import matplotlib.pyplot as plt

        fd = self.generator.fake_dataset
        rd = self.generator.real_dataset
        combined = tf.data.Dataset.zip((rd, fd))

        for real_d, fake_d in combined:
            fig, axs = plt.subplots(
                rows, cols, sharey=True, sharex=True, figsize=(15, 10))
            pred_plots = True

            fake_d = self.generator.model.predict(fake_d)

            for i in range(rows):
                if i >= rows/2:
                    pred_plots = False
                for k in range(cols):
                    ptype = "Generated" if pred_plots else "Ground Truth"
                    axs[i, k].set_title("{}".format(ptype))

                    if pred_plots:
                        axs[i, k].plot(
                            fake_d[np.random.randint(0, len(fake_d)), :, 0], c='c')
                    else:
                        axs[i, k].plot(
                            real_d[np.random.randint(0, len(real_d)), :, 0], c='g')

            break

        if save_plot:
            plt.savefig('result_imgs/results_epoch_{}.png'.format(epoch))
        else:
            plt.show()
        plt.clf()
        plt.close()

    def train(self, epochs, n_eval, d_train_steps=5, load_weights=False, load_w_names=("generator", "discriminator"),
              save_w_names=("generator", "discriminator"), clear_session=False):

        import time

        if load_weights:
            self.load_weights(load_w_names)
            print("Loaded previously saved weights")

        steps_over_data = len(
            self.generator.all_real_samples)//self.generator.batch_size
        print("Begin Training with {} steps per epoch for every {} trains on discriminator".format(
            steps_over_data, d_train_steps))

        real_fake_datasets = tf.data.Dataset.zip(
            (self.generator.real_dataset, self.generator.fake_dataset))
        for epoch in range(epochs):
            start = time.time()

            step_start = time.time()
            d_loss_tmp = list()

            for step, _ in self.generator.real_dataset.enumerate():
                d_loss_real_tmp = list()
                d_loss_fake_tmp = list()

                # train discriminator
                d_counter = 0
                for realx, fakex in real_fake_datasets:
                    if d_counter == d_train_steps:
                        break

                    fakex = self.generator.model.predict(fakex)

                    d_loss_real = self.discriminator.model.train_on_batch(realx,
                                                                          self.generator.ground_truth_valid)

                    d_loss_fake = self.discriminator.model.train_on_batch(fakex,
                                                                          self.generator.ground_truth_fake)

                    d_loss_real_tmp.append(d_loss_real)
                    d_loss_fake_tmp.append(d_loss_fake)
                    d_counter += 1

                step_d_loss = tf.add(d_loss_real_tmp, d_loss_fake) * 0.5
                step_d_loss = tf.reduce_mean(step_d_loss, axis=0)

                d_loss_tmp.append(
                    tf.reduce_mean(
                        tf.add(
                            d_loss_real_tmp, d_loss_fake) * 0.5,
                        axis=0)
                )

                if step % int((steps_over_data*.1)) == 0:
                    print("\tTNE: {} ({:.2f}s)".format(
                        steps_over_data-(step), time.time()-step_start))

                d_loss_avg = tf.reduce_mean(d_loss_tmp, axis=0)

                self.d_loss.append(d_loss_avg)

                for noise in self.generator.fake_dataset.take(1):
                    valid = self.generator.ground_truth_valid
                    g_loss = self.model.train_on_batch(noise, valid)

            end = time.time()-start
            if epoch % n_eval == 0:
                status = "[Time: {:.2f}s Epoch: {}/{}] [D Loss: {:.5f} acc: {:.2f}%] [G Loss {:.5f}]".format(end,
                                                                                                             epoch+1,
                                                                                                             epochs,
                                                                                                             self.d_loss[-1][0],
                                                                                                             self.d_loss[-1][1] *
                                                                                                             100.0,
                                                                                                             g_loss)

                print(status)

                self.save_weights(save_w_names)

                if clear_session:
                    tf.keras.backend.clear_session()
                    print("Clearing Session")

                self.plot_preds(rows=4, cols=6, save_plot=True)


if __name__ == '__main__':
    gan = GAN(latent_dim=5, seq_length=128, batch_size=128)
    gan.generator.model.summary()
    gan.discriminator.model.summary()
    gan.train(epochs=97, n_eval=1, d_train_steps=5,
              load_weights=False, save_w_names=("g", "d"), load_w_names=('g', 'd'))
