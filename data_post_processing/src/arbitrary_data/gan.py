#!/usr/bin/env python

# class for a generative adversarial network. One example at
# https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/GAN/dcgan_mnist.py

'''
DCGAN on MNIST using Keras
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
Dependencies: tensorflow 1.0 and keras 2.0
Usage: python3 dcgan_mnist.py
'''

import numpy as np
import time, glob, sys
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Reshape, concatenate, multiply
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, Input
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
import os
if not 'DISPLAY' in os.environ.keys():
    import matplotlib as mpl
    mpl.use('Agg')
import matplotlib.pyplot as plt

class GAN:
    def __init__(self, img_rows=200, img_cols=200, channel=1):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.E = None   # encoder used by generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model
        self.GM = None

    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        depth = 64
        dropout = 0.4
        # In: 200 x 200 x 2, depth = 1
        # Out: 25 x 25 x 1, depth=64
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D

    def generator(self):
        if self.G:
            return self.G
        self.G = Sequential()
        dropout = 0.4
        depth = 64+64+64+64
        dim = 25
        # In: 100
        # Out: dim x dim x depth
        self.G.add(Conv2D(1,kernel_size=5,strides=2,input_shape=(self.img_rows,self.img_cols,1),data_format='channels_last',padding='same'))
        self.G.add(Conv2D(1,kernel_size=5,strides=2,padding='same'))
        self.G.add(Conv2D(1,kernel_size=5,strides=2,padding='same'))
        self.G.add(Conv2D(1,kernel_size=5,strides=1,padding='same'))
        self.G.add(Flatten())
        self.G.add(Dense(dim*dim*depth, input_dim=625))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim, dim, depth)))
        self.G.add(Dropout(dropout))

        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        # Out: 200 x 200 x 1 grayscale image [0.0,1.0] per pix
        self.G.add(Conv2DTranspose(1, 5, padding='same'))
        self.G.add(Activation('sigmoid'))
        self.G.summary()
        return self.G

    def encoder(self):
        if self.E:
            return self.E
        self.E = Sequential()
        self.E.add(Conv2D(1,kernel_size=5,strides=2,input_shape=(200,200,1),padding='same'))
        self.E.add(Flatten())
        return self.E

    def generator_model(self):
        if self.GM:
            return self.GM
        generator = self.generator()
        # encoder = self.encoder()
        im = Input(shape = (self.img_rows,self.img_cols,1))
        output = generator(im)
        self.GM = Model(im,output)
        # noise = Input(shape=(1,100))
        # img = Input(shape=(self.img_rows,self.img_cols,1))
        # encoded = encoder(img)
        # # model_input = concatenate([encoded,noise])
        # output = generator(model_input)
        # self.GM = Model([img,noise],output)


        return self.GM

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = Adam(lr=0.0002)
        heatmap = Input(shape=(self.img_rows,self.img_cols,1))
        img = Input(shape=(self.img_rows,self.img_cols,1))

        input = multiply([img,heatmap])
        output = self.discriminator()(input)
        self.DM = Model([img,heatmap],output)
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = Adam(lr=0.0001)
        noise = Input(shape=(self.img_rows,self.img_cols,1))
        img = Input(shape=(self.img_rows,self.img_cols,1))
        generator_model = self.generator_model()
        discriminator_model = self.discriminator_model()

        # make discriminator not train when training generator
        discriminator_model.trainable = False
        # heatmap = generator_model([img,noise])
        heatmap = generator_model(img)
        output = discriminator_model([img,heatmap])
        # self.AM = Model([img,noise],output)
        self.AM = Model(img,output)
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.AM

    def train(self, data_path, pattern, epochs,batch_size=128, sample_interval=100, output_path = None):
        seed = int(time.time())
        np.random.seed(seed)

        files = glob.glob(data_path+'**/'+pattern)
        X_maps = []
        X_heatmaps = []
        np.random.shuffle(files)
        training_files = files[:int(len(files)*.9)]
        validation_files = files[int(len(files)*.9):]
        for file in training_files:
            data = np.load(file)
            for point in data:
                X_maps.append((point[:,:,0]/255.0))
                X_heatmaps.append((point[:,:,1]))
        X_maps = np.array(X_maps)
        X_heatmaps = np.array(X_heatmaps)
        X_maps = X_maps.reshape(X_maps.shape[0],X_maps.shape[1],X_maps.shape[2],1)
        X_heatmaps = X_heatmaps.reshape(X_heatmaps.shape[0],X_heatmaps.shape[1],X_heatmaps.shape[2],1)
        shuffler = np.arange(len(X_maps))
        np.random.shuffle(shuffler)
        X_maps = X_maps[shuffler]
        X_heatmaps = X_heatmaps[shuffler]

        valid = np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))

        generator_model = self.generator_model()
        discriminator_model = self.discriminator_model()
        adversarial_model = self.adversarial_model()
        for epoch in range(epochs):
            # Train the discriminator first
            idx = np.random.randint(0,X_maps.shape[0],batch_size)
            imgs, heatmaps = X_maps[idx], X_heatmaps[idx]

            # noise = np.random.normal(0,1, (batch_size,100))

            # gen_heatmaps = generator_model.predict([imgs,noise])
            gen_heatmaps = generator_model.predict(imgs)

            # training step

            d_loss_real = discriminator_model.train_on_batch([imgs,heatmaps],valid)
            d_loss_fake = discriminator_model.train_on_batch([imgs,gen_heatmaps],fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            idx = np.random.randint(0,X_maps.shape[0],batch_size)
            sampled_ims = X_maps[idx]

            # g_loss = adversarial_model.train_on_batch([sampled_ims,noise],valid)
            g_loss = adversarial_model.train_on_batch(sampled_ims,valid)

            # plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))

            if epoch % sample_interval == 0:
                self.sample_images(epoch,X_maps,X_heatmaps,output_path)


    def sample_images(self,epoch,X_maps, X_heatmaps,output_path):
        if not output_path:
            return 0
        r,c = 4,3
        # noise = np.random.normal(0,1, (r, X_maps.shape[1], X_maps.shape[2],1))
        idx = np.random.randint(0,X_maps.shape[0],r)
        sampled_ims = X_maps[idx]
        sampled_heatmaps = X_heatmaps[idx]

        generator_model = self.generator_model()

        # gen_ims = generator_model.predict([sampled_ims,noise])
        gen_ims = generator_model.predict(sampled_ims)

        fig,axs = plt.subplots(r,c)
        cnt = 0

        for i in range(r):
            axs[i,0].imshow(sampled_ims[cnt,:,:,0], cmap='gray')
            axs[i,1].imshow(gen_ims[cnt,:,:,0],cmap='plasma')
            axs[i,2].imshow(sampled_heatmaps[cnt,:,:,0],cmap='plasma')
            axs[i,0].set_title("Map: %d" % cnt)
            axs[i,1].set_title("Generated: %d" % cnt)
            axs[i,2].set_title("Real: %d" % cnt)
            axs[i,0].axis('off')
            axs[i,1].axis('off')
            axs[i,2].axis('off')
            cnt += 1

        fig.savefig(output_path + "%d.png" % epoch)

if __name__ == "__main__":
    g = GAN()
    g.train(sys.argv[1],'relative_training_data.npy', epochs=20000, batch_size = 64, sample_interval=500,output_path = sys.argv[2])
