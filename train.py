from model import GAN, Discriminator, Generator, DataGenerator
gan = GAN(latent_dim=5, seq_length=30, batch_size=128)
gan.discriminator.model.summary()
gan.train(epochs=50, n_eval=1, d_train_steps=3)
gan.generator.model.save_weights("generator_weights.h5")
gan.discriminator.model.save_weights("discriminator_weights.h5")