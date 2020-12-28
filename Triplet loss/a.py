from PIL import Image
import numpy as np
import os
import tensorflow_addons as tfa
import tensorflow as tf
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import imageio 
import glob 
import PIL
import tensorflow as tf
from tensorflow.keras import layers
import time
import zipfile

train_images = np.load('./dataset/dataset/dataset.npy')

BUFFER_SIZE = 1000000
BATCH_SIZE = 256
embedding_size = 32
margin = 0.5


train_rest = np.concatenate((train_images[0][:15000,:,:,:] , train_images[2][:15000,:,:,:],train_images[3][:15000,:,:,:],train_images[4][:15000,:,:,:],train_images[1][:15000,:,:,:],train_images[6][:15000,:,:,:],train_images[7][:15000,:,:,:],train_images[8][:15000,:,:,:]))
print(train_rest.shape)


train_dataset_2 = tf.data.Dataset.from_tensor_slices(train_images[1]).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_dataset_3 = tf.data.Dataset.from_tensor_slices(train_images[2]).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

train_dataset_6 = tf.data.Dataset.from_tensor_slices(train_images[5]).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

train_dataset_rest = tf.data.Dataset.from_tensor_slices(train_rest).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model


def make_discriminator_model():
    # model = tf.keras.Sequential()
    input_layer = layers.Input(shape=[28, 28, 1])
    layer = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(input_layer)
    layer = layers.LeakyReLU()(layer)
    layer = layers.Dropout(0.3)(layer)

    layer = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(layer)
    layer = layers.LeakyReLU()(layer)
    layer = layers.Dropout(0.3)(layer)

    layer = layers.Flatten()(layer)
    embedding_layer = layers.Dense(embedding_size)(layer)
    embedding_layer = layers.ReLU()(embedding_layer)
    # output_layer = layers.Dense(class_count)(embedding_layer)
    layer = layers.Dense(1)(embedding_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=[layer, embedding_layer])
    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# triplet_loss_object = tfa.losses.TripletSemiHardLoss()

def triplet_loss(model_output_anchor, model_output_positive, model_output_negative):
    d_pos = tf.reduce_sum(tf.square(model_output_anchor - model_output_positive), 1)
    d_neg = tf.reduce_sum(tf.square(model_output_anchor - model_output_negative), 1)

    loss = tf.maximum(0., margin + d_pos - d_neg)
    loss = tf.reduce_mean(loss)
    return loss

def discriminator_loss(real_output, fake_output , real_embed , fake_embed , non_rel):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    # t_loss = triplet_loss(fake_embed , real_embed , non_rel)
    total_loss = real_loss + fake_loss #+ t_loss
    return total_loss

def generator_loss(fake_output , fake_embed , real_embed , non_rel):
    cent =  cross_entropy(tf.ones_like(fake_output), fake_output)
    # t_loss = triplet_loss(fake_embed , real_embed , non_rel)
    total_loss = cent # + t_loss
    return total_loss

EPOCHS = 10
noise_dim = 100
num_examples_to_generate = 25

# @tf.function
# def train_step(images):

def generate_and_save_images(model, epoch, test_input,smp):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    plt.imshow(predictions[1, :, :, 0] * 127.5 + 127.5, cmap='gray')

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(5, 5, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('./'+ smp +'/image_at_epoch_{:04d}.png'.format(epoch))
    #   plt.show()

def train_for_a_number(nubmer , epochs , dataset , non_relev_image):

    generator = make_generator_model()
    generator.summary()

    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)

    plt.imshow(generated_image[0, :, :, 0], cmap='gray')

    discriminator = make_discriminator_model()
    discriminator.summary()

    decision = discriminator(generated_image)
    # print (decision)
    # cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint_dir = './training_checkpoints_' + str(nubmer)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)

    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    smp = str(nubmer)
    for epoch in range(epochs):
        start = time.time()
        e_rel = non_relev_image.__iter__()
        for images in dataset:

            noise = tf.random.normal([images.shape[0], noise_dim])

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = generator(noise, training=True)

                real_output , real_embed = discriminator(images, training=True)
                fake_output , fake_embed = discriminator(generated_images, training=True)
                _ , dis_output_non_rel = discriminator(e_rel.next() , training=True)


                gen_loss = generator_loss(fake_output , fake_embed , real_embed , dis_output_non_rel )
                disc_loss = discriminator_loss(real_output, fake_output , real_embed , fake_embed , dis_output_non_rel)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


        generate_and_save_images(generator,epoch + 1,seed,smp)

        # Save the model every 15 epochs
            
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print (smp+':Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))


    generate_and_save_images(generator,epochs,seed,smp)




train_for_a_number(2, EPOCHS , train_dataset_2 , train_dataset_rest)
train_for_a_number(3, EPOCHS , train_dataset_3 , train_dataset_rest)
train_for_a_number(6, EPOCHS , train_dataset_6 , train_dataset_rest)


exit(1)

