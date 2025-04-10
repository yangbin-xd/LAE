
# generate fake CSI with received pilots
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from process import CSI, CSI_comp, label_repe, normalize_CSI, add_noise
[N, Nv, Nt, Nc, Nf, Ns] = CSI.shape

cgan_nmse = np.empty([7, int(Nv*N*Nf*0.2)])

for SNR in np.arange(0, 31, 5):
    # channel estimation with pilots
    CSI_noise = add_noise(CSI, SNR)
    CSI_tran = CSI_noise.transpose(1,0,4,3,2,5) # (6, 231, 10, 12, 32, 14)
    CSI_temp = CSI_tran.reshape([Nv*N, Nf, Nc, Nt, Ns]) # (1386, 10, 12, 32, 14)
    CSI_resp = CSI_temp.reshape([Nv*N*Nf, Nc, Nt, Ns]) # (1.386w, 12, 32, 14)
    CSI_norm = normalize_CSI(CSI_resp) # (1.386w, 12, 32, 14)
    Nc_pos = [2,4,8,10]
    CSI_pilot = np.empty([CSI_norm.shape[0], Nt], dtype='complex')
    for i in range(CSI_norm.shape[0]):
        for j in range(len(Nc_pos)):
            for k in range(int(Nt/len(Nc_pos))):
                CSI_pilot[i, 8*j+k] = \
                np.mean(CSI_norm[i, Nc_pos[j]:Nc_pos[j]+2, 8*j+k, 3:7])
    CSI_pilot = np.concatenate([np.real(CSI_pilot), np.imag(CSI_pilot)], axis=-1) # (1.386w, 64)

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(CSI_comp, CSI_pilot,
                                                        test_size=0.2, random_state=1)
    label_train, label_test = train_test_split(label_repe,
                                               test_size=0.2, random_state=1)

    # set hyperparameters
    s = 3
    z_dim = 500
    batch_size = 128
    epoch = 20000
    decay_factor = 0.1
    patience = 2000

    # build generator
    def build_generator():
        noise_input = layers.Input(shape=(z_dim,))
        label_input = layers.Input(shape=(2*Nt,))
        input = layers.Concatenate()([noise_input, label_input])

        x = layers.Dense(2 * 4 * 2 * 256, input_dim=z_dim+2*Nt)(input)
        x = layers.Reshape((2, 4, 2, 256))(x)

        x = layers.UpSampling3D(size=(2, 2, 2))(x) 
        x = layers.Conv3D(128, kernel_size=(s, s, s), strides=(1, 1, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.01)(x)

        x = layers.UpSampling3D(size=(2, 2, 2))(x)
        x = layers.Conv3D(64, kernel_size=(s, s, s), strides=(1, 1, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.01)(x)

        x = layers.UpSampling3D(size=(2, 2, 2))(x)
        x = layers.Conv3D(32, kernel_size=(s, s, s), strides=(1, 1, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.01)(x)

        x = layers.Conv3D(2, kernel_size=(s, s, s), strides=(1, 1, 1), padding='same')(x)
        x = layers.Cropping3D(cropping=((2, 2), (0, 0), (1, 1)))(x)

        output = layers.Activation('tanh')(x)
        return models.Model([noise_input, label_input], output)

    # build discriminator
    def build_discriminator():
        input = layers.Input(shape=(Nc, Nt, Ns, 2))

        x = layers.Conv3D(32, kernel_size=(s, s, s), strides=(2, 2, 2),
                           padding='same')(input)
        x = layers.LeakyReLU(alpha=0.01)(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Conv3D(64, kernel_size=(s, s, s), strides=(2, 2, 2),
                           padding='same')(x)
        x = layers.LeakyReLU(alpha=0.01)(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Conv3D(128, kernel_size=(s, s, s), strides=(2, 2, 2),
                           padding='same')(x)
        x = layers.LeakyReLU(alpha=0.01)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Flatten()(x)

        output = layers.Dense(1)(x)
        return models.Model(input, output)

    # build model
    generator = build_generator()
    discriminator = build_discriminator()

    # train
    g_losses, d_losses, e_losses, dgan_losses, ggan_losses, gp_losses=[[] for _ in range(6)]
    def train(lr = 0.0001):
        g_optimizer = optimizers.Adam(learning_rate=lr, beta_1=0.1, beta_2=0.999)
        d_optimizer = optimizers.Adam(learning_rate=lr, beta_1=0.1, beta_2=0.999)
        no_improve_steps = 0
        best_dgan_loss = float('inf')
        best_e_loss = float('inf')
        best_i_loss = float('inf')

        # gradient penalty
        def gradient_penalty(discriminator, real_images, fake_images, batch_size):
            alpha = tf.random.uniform([batch_size, 1, 1, 1, 1], 0., 1.)
            interpolated = alpha * real_images + (1 - alpha) * fake_images
            with tf.GradientTape() as tape:
                tape.watch(interpolated)
                pred = discriminator(interpolated)
            grads = tape.gradient(pred, interpolated)
            norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3, 4]))
            gp = tf.reduce_mean((norm - 1.0) ** 2)
            return gp

        for iteration in range(epoch):
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            real_images = x_train[idx]
            real_labels = y_train[idx].reshape(batch_size, 2*Nt)
            z = np.random.normal(0, 1, (batch_size, z_dim))
            fake_images = generator([z, real_labels])

            # train discriminator
            with tf.GradientTape() as d_tape:
                fake_logits = discriminator(fake_images, training=True)
                real_logits = discriminator(real_images, training=True)
                dgan_loss = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
                gp_loss = gradient_penalty(discriminator, real_images,
                                           fake_images, batch_size)
                d_loss = dgan_loss + gp_loss * 10

            d_gradients = d_tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

            # train generator
            with tf.GradientTape() as g_tape:
                z = np.random.normal(0, 1, (batch_size, z_dim))
                idx = np.random.randint(0, x_train.shape[0], batch_size)
                real_images = x_train[idx]
                real_labels = y_train[idx].reshape(batch_size, 2*Nt)
                fake_images = generator([z, real_labels])
                fake_logits = discriminator(fake_images, training=False)
                e_loss = tf.reduce_mean(tf.square(real_images - fake_images))

                ggan_loss = -tf.reduce_mean(fake_logits)
                g_loss = ggan_loss + 1000*e_loss

            g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

            # save loss
            g_losses.append(g_loss.numpy())
            d_losses.append(d_loss.numpy())
            e_losses.append(e_loss.numpy())
            dgan_losses.append(dgan_loss.numpy())
            ggan_losses.append(ggan_loss.numpy())
            gp_losses.append(gp_loss.numpy())

            i_loss = dgan_loss + 1000*e_loss

            if (iteration + 1) >= 5000:
                if np.abs(i_loss) < best_i_loss:
                    best_i_loss = np.abs(i_loss)
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1

                if no_improve_steps >= patience and lr > 5e-7:
                    lr = lr * decay_factor
                    g_optimizer.learning_rate = lr
                    d_optimizer.learning_rate = lr
                    no_improve_steps = 0
                    best_i_loss = np.abs(i_loss)
                    print('learning rate changed to:', lr)

                if e_loss < best_e_loss:
                    best_e_loss = e_loss
                    generator.save(f'model/best_generator{SNR}.h5')

            if (iteration + 1) % 1000 == 0:
                print(f'{iteration + 1} [D loss: {d_loss}] [dgan_loss: {dgan_loss}] [gp_loss: {gp_loss}]')
                print(f'{iteration + 1} [G loss: {g_loss}] [ggan_loss: {ggan_loss}] [e_loss: {e_loss}]')

        generator.save(f'model/generator{SNR}.h5')
        discriminator.save(f'model/discriminator{SNR}.h5')

        np.save(f'loss/g_losses{SNR}.npy', g_losses)
        np.save(f'loss/d_losses{SNR}.npy', d_losses)
        np.save(f'loss/e_losses{SNR}.npy', e_losses)
        np.save(f'loss/dgan_losses{SNR}.npy', dgan_losses)
        np.save(f'loss/ggan_losses{SNR}.npy', ggan_losses)
        np.save(f'loss/gp_losses{SNR}.npy', gp_losses)

    # run 
    train()

    # test
    generator = models.load_model(f'model/generator{SNR}.h5')
    z = np.random.normal(0, 1, (y_test.shape[0], 500))
    x_pred = generator.predict([z, y_test])

    # calculate nmse
    def cal_nmse(CSI, CSI_esti):
        nmse = np.empty(CSI.shape[0])
        for i in range(CSI.shape[0]):
            mse = np.mean(np.abs(CSI[i,:] - CSI_esti[i,:]) ** 2)
            norm_factor = np.mean(np.abs(CSI[i,:]) ** 2)
            nmse[i] = mse / norm_factor
        return nmse
    
    cgan_nmse[int(SNR/5),:] = cal_nmse(x_test, x_pred)

# save nmse and velocity
np.save('result/cgan_nmse.npy', cgan_nmse)
np.save('result/velocity.npy', label_test[:,2])

# print result vs SNR
print('cgan nmse:', np.round(np.mean(cgan_nmse, 1), 4))
