from model import GAN, Discriminator, Generator, DataGenerator


@profile
def train():
    gan = GAN(latent_dim=5, seq_length=30, batch_size=128)
    gan.discriminator.model.summary()
    gan.load_weights()
    gan.train(epochs=40, n_eval=1, d_train_steps=3,
              load_weights=True, metric='loss')


if __name__ == '__main__':
    train()
