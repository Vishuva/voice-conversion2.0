import numpy as np
import librosa
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Function to load and preprocess audio files
def load_audio(file_path, sr=22050):
    audio, _ = librosa.load(file_path, sr=sr)
    return audio

# Function to extract mel-spectrogram
def audio_to_mel_spectrogram(audio, sr=22050, n_fft=1024, hop_length=256, n_mels=80):
    spectrogram = librosa.feature.melspectrogram(audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    spectrogram = librosa.power_to_db(spectrogram)
    return spectrogram

# Load datasets
def load_dataset(directory):
    audios = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.wav'):
            file_path = os.path.join(directory, file_name)
            audio = load_audio(file_path)
            mel_spectrogram = audio_to_mel_spectrogram(audio)
            audios.append(mel_spectrogram)
    return np.array(audios)

# Example paths (use your own dataset paths)
source_voice_path = 'data/source_voice'
target_voice_path = 'data/target_voice'

source_audios = load_dataset(source_voice_path)
target_audios = load_dataset(target_voice_path)

# Define CycleGAN components
def build_generator():
    inputs = layers.Input(shape=(None, 80, 1))
    x = layers.Conv2D(64, (3, 9), padding='same')(inputs)
    x = layers.ReLU()(x)
    
    for _ in range(3):
        x = layers.Conv2D(128, (3, 9), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    
    x = layers.Conv2DTranspose(64, (3, 9), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    outputs = layers.Conv2D(1, (3, 9), padding='same')(x)
    model = Model(inputs, outputs)
    return model

def build_discriminator():
    inputs = layers.Input(shape=(None, 80, 1))
    x = layers.Conv2D(64, (3, 9), padding='same')(inputs)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    for _ in range(3):
        x = layers.Conv2D(128, (3, 9), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
    
    outputs = layers.Conv2D(1, (3, 9), padding='same')(x)
    model = Model(inputs, outputs)
    return model

# Define CycleGAN model
class CycleGAN(Model):
    def __init__(self):
        super(CycleGAN, self).__init__()
        self.generator_g = build_generator()
        self.generator_f = build_generator()
        self.discriminator_x = build_discriminator()
        self.discriminator_y = build_discriminator()

    def compile(self, gen_g_optimizer, gen_f_optimizer, disc_x_optimizer, disc_y_optimizer, gen_loss_fn, disc_loss_fn, cycle_loss_fn, identity_loss_fn):
        super(CycleGAN, self).compile()
        self.gen_g_optimizer = gen_g_optimizer
        self.gen_f_optimizer = gen_f_optimizer
        self.disc_x_optimizer = disc_x_optimizer
        self.disc_y_optimizer = disc_y_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn

    def train_step(self, batch_data):
        real_x, real_y = batch_data

        with tf.GradientTape(persistent=True) as tape:
            fake_y = self.generator_g(real_x, training=True)
            cycled_x = self.generator_f(fake_y, training=True)

            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)

            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)

            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)

            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            gen_g_loss = self.gen_loss_fn(disc_fake_y)
            gen_f_loss = self.gen_loss_fn(disc_fake_x)

            total_cycle_loss = self.cycle_loss_fn(real_x, cycled_x) + self.cycle_loss_fn(real_y, cycled_y)

            total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss_fn(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss_fn(real_x, same_x)

            disc_x_loss = self.disc_loss_fn(disc_real_x, disc_fake_x)
            disc_y_loss = self.disc_loss_fn(disc_real_y, disc_fake_y)

        grads_g = tape.gradient(total_gen_g_loss, self.generator_g.trainable_variables)
        grads_f = tape.gradient(total_gen_f_loss, self.generator_f.trainable_variables)
        disc_x_grads = tape.gradient(disc_x_loss, self.discriminator_x.trainable_variables)
        disc_y_grads = tape.gradient(disc_y_loss, self.discriminator_y.trainable_variables)

        self.gen_g_optimizer.apply_gradients(zip(grads_g, self.generator_g.trainable_variables))
        self.gen_f_optimizer.apply_gradients(zip(grads_f, self.generator_f.trainable_variables))
        self.disc_x_optimizer.apply_gradients(zip(disc_x_grads, self.discriminator_x.trainable_variables))
        self.disc_y_optimizer.apply_gradients(zip(disc_y_grads, self.discriminator_y.trainable_variables))

        return {
            "gen_g_loss": total_gen_g_loss,
            "gen_f_loss": total_gen_f_loss,
            "disc_x_loss": disc_x_loss,
            "disc_y_loss": disc_y_loss
        }

# Define loss functions and optimizers
def generator_loss_fn(disc_generated_output):
    return tf.keras.losses.MeanSquaredError()(tf.ones_like(disc_generated_output), disc_generated_output)

def discriminator_loss_fn(disc_real_output, disc_generated_output):
    real_loss = tf.keras.losses.MeanSquaredError()(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = tf.keras.losses.MeanSquaredError()(tf.zeros_like(disc_generated_output), disc_generated_output)
    return (real_loss + generated_loss) * 0.5

def cycle_loss_fn(real_image, cycled_image, LAMBDA=10):
    loss = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return LAMBDA * loss

def identity_loss_fn(real_image, same_image, LAMBDA=10):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss

gen_g_optimizer = Adam(2e-4, beta_1=0.5)
gen_f_optimizer = Adam(2e-4, beta_1=0.5)
disc_x_optimizer = Adam(2e-4, beta_1=0.5)
disc_y_optimizer = Adam(2e-4, beta_1=0.5)

# Instantiate CycleGAN model
cyclegan_model = CycleGAN()

cyclegan_model.compile(
    gen_g_optimizer=gen_g_optimizer,
    gen_f_optimizer=gen_f_optimizer,
    disc_x_optimizer=disc_x_optimizer,
    disc_y_optimizer=disc_y_optimizer,
    gen_loss_fn=generator_loss_fn,
    disc_loss_fn=discriminator_loss_fn,
    cycle_loss_fn=cycle_loss_fn,
    identity_loss_fn=identity_loss_fn
)

# Train the model
EPOCHS = 50

for epoch in range(EPOCHS):
    for i in range(len(source_audios)):
        batch_data = (source_audios[i], target_audios[i])
        losses = cyclegan_model.train_step(batch_data)
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Losses: {losses}")

# Save the model
cyclegan_model.generator_g.save('voice_conversion_generator_g.h5')
