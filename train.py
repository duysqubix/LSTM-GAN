

def train():
    from model import GAN
    gan = GAN(latent_dim=5, seq_length=30, batch_size=128)
    gan.discriminator.model.summary()
    gan.load_weights()
    gan.train(epochs=40, n_eval=1, d_train_steps=3,
              load_weights=True, metric='loss')


def train_new():
    from model_opt import GAN
    gan = GAN(latent_dim=5, seq_length=30, batch_size=128)
    gan.train(epochs=100, n_eval=1, d_train_steps=3, save_w_names=("g2", "d2"))


if __name__ == '__main__':
    train_new()
