import os, cv2, numpy as np, tensorflow as tf
from tensorflow.keras import layers, models # type: ignore

def LoadImagesFromFolder(folder, size = (256, 256), batch_size = 32):
    dataset = tf.keras.utils.image_dataset_from_directory(
        folder,
        validation_split = 0.2,
        subset = "training",
        seed = 123,
        image_size = size, 
        batch_size = batch_size,
        label_mode = None,
)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        folder,
        validation_split = 0.2,
        subset = "validation",
        seed = 123,
        image_size = size,
        batch_size = batch_size,
        label_mode = None
)

    return dataset, val_ds

def EditImage(image):
    image = tf.cast(image, tf.float32) / 255.0
    hsv_image = tf.image.rgb_to_hsv(image)

    h = hsv_image[:, :, :, 0]
    s = hsv_image[:, :, :, 1]
    v = hsv_image[:, :, :, 2]

    hs = tf.stack([h, s], axis=-1)
    v = tf.expand_dims(v, axis=-1)

    return hs, v

def PreprocessImages(dataset):
    hs_images = dataset.map(lambda image: EditImage(image)[0])
    v_images = dataset.map(lambda image: EditImage(image)[1])

    return tf.data.Dataset.zip((v_images, hs_images))

def LoadOriginalImages(folder, size = (256, 256)):
    images = []
    for file_name in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file_name))
        if img is not None:
            img = cv2.resize(img, size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

    return images

def CombineWithGray(ai_generated, gray_images, num):
    finished = []
    for i in range(num):
        hs = ai_generated[i]
        h = hs[:, :, 0]
        s = hs[:, :, 1]
        v = gray_images[i][:, :, 0]

        h = h * 179
        s = s * 255
        v = v * 255

        photo = np.stack([h, s, v], axis = -1).astype(np.uint8)
        photo = cv2.cvtColor(photo, cv2.COLOR_HSV2RGB)

        finished.append(photo)
    
    return finished

def MakeModel(input_size=(128, 128, 1)):
    size = input_size[0]
    inputs = layers.Input(input_size)
    
    # Encoder
    conv1 = layers.Conv2D(size // 2, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(size // 2, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(size, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(size, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(2 * size, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(2 * size, 3, activation='relu', padding='same')(conv3)

    # Decoder
    up4 = layers.Conv2DTranspose(size, 2, strides=(2, 2), padding='same')(conv3)
    merge4 = layers.concatenate([conv2, up4], axis=3)
    conv4 = layers.Conv2D(size, 3, activation='relu', padding='same')(merge4)
    conv4 = layers.Conv2D(size, 3, activation='relu', padding='same')(conv4)

    up5 = layers.Conv2DTranspose(size // 2, 2, strides=(2, 2), padding='same')(conv4)
    merge5 = layers.concatenate([conv1, up5], axis=3)
    conv5 = layers.Conv2D(size // 2, 3, activation='relu', padding='same')(merge5)
    conv5 = layers.Conv2D(size // 2, 3, activation='relu', padding='same')(conv5)

    # Output layer
    outputs = layers.Conv2D(2, 1, activation='sigmoid')(conv5)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def MakeLargerModel(input_size = (256, 256, 1)):
    # Input
    inputs = layers.Input(input_size)

    # Encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs) # 1 nivo
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1) # slika postaje (128, 128)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1) # 2 nivo
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2) # slika postaje (64, 64)

    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2) # 3 nivo
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3) # slika postaje (32, 32)

    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3) # 4 nivo
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4) # slika postaje (16, 16)

    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4) # 5 nivo
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)

    # Decoder
    up6 = layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv5) # slika postaje (32, 32)
    merge6 = layers.concatenate([conv4, up6], axis=3) # spajaju se konvolucija4 i uvecavanje8 jer su 4 nivo 
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6) # slika postaje (64, 64)
    merge7 = layers.concatenate([conv3, up7], axis=3) # spajaju se konvolucija3 i uvecavanje9 jer su 3 nivo
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7) # slika postaje (128, 128)
    merge8 = layers.concatenate([conv2, up8], axis=3) # spajaju se konvolucija2 i uvecavanje10 jer su 2 nivo
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8) # slika postaje (256, 256)
    merge9 = layers.concatenate([conv1, up9], axis=3) # spajaju se konvolucija1 i uvecavanje11 jer su 1 nivo
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)

    # Output
    outputs = layers.Conv2D(2, 1, activation='sigmoid')(conv9)
    
    # Model
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def GenerateHistogram(photo, bins = 256):
    h = photo[:, :, 0]
    s = photo[:, :, 1]
    range = [0., 1.]

    h_hist = tf.histogram_fixed_width(h, range, nbins = bins)
    s_hist = tf.histogram_fixed_width(s, range, nbins = bins)

    h_hist = h_hist / tf.reduce_sum(h_hist)
    s_hist = s_hist / tf.reduce_sum(s_hist)

    return (h_hist, s_hist)

def Divergence(a, b):
    epsilon = 1e-10
    a = tf.clip_by_value(a, epsilon, 1.0)
    b = tf.clip_by_value(b, epsilon, 1.0)
    return tf.reduce_sum(a * tf.math.log(a/b))

def ColorDistibutionLoss(hist_real, hist_pred):
    h_real, s_real = hist_real
    h_pred, s_pred = hist_pred

    dist_hue = Divergence(h_real, h_pred)
    dist_sat = Divergence(s_real, s_pred)
    loss = dist_hue + dist_sat
    
    return loss

def MseLoss(image_real, image_pred):
    return tf.reduce_mean(tf.square(image_real - image_pred))

def CustomLoss(photo_real, photo_generated, color_lambda = 1., mse_lambda = 1.):

    hist_real = GenerateHistogram(photo_real)
    hist_generated = GenerateHistogram(photo_generated)
    color_distribution_loss = ColorDistibutionLoss(hist_real, hist_generated)
    color_distribution_loss = tf.cast(color_distribution_loss, dtype = tf.float64)

    mse_loss = MseLoss(photo_real, photo_generated)
    mse_loss = tf.cast(mse_loss, dtype = tf.float64)

    total_loss = color_lambda * color_distribution_loss + mse_lambda * mse_loss
    return total_loss