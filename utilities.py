import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
import os, cv2, numpy as np

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
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(4 * size, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(4 * size, 3, activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = layers.Conv2D(8 * size, 3, activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(8 * size, 3, activation='relu', padding='same')(conv5)

    # Decoder
    up6 = layers.Conv2DTranspose(4 * size, 2, strides=(2, 2), padding='same')(conv5)
    merge6 = layers.concatenate([conv4, up6], axis=3)
    conv6 = layers.Conv2D(4 * size, 3, activation='relu', padding='same')(merge6)
    conv6 = layers.Conv2D(4 * size, 3, activation='relu', padding='same')(conv6)

    up7 = layers.Conv2DTranspose(2 * size, 2, strides=(2, 2), padding='same')(conv6)
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(2 * size, 3, activation='relu', padding='same')(merge7)
    conv7 = layers.Conv2D(2 * size, 3, activation='relu', padding='same')(conv7)

    up8 = layers.Conv2DTranspose(size, 2, strides=(2, 2), padding='same')(conv7)
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(size, 3, activation='relu', padding='same')(merge8)
    conv8 = layers.Conv2D(size, 3, activation='relu', padding='same')(conv8)

    up9 = layers.Conv2DTranspose(size // 2, 2, strides=(2, 2), padding='same')(conv8)
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(size // 2, 3, activation='relu', padding='same')(merge9)
    conv9 = layers.Conv2D(size // 2, 3, activation='relu', padding='same')(conv9)

    # Output layer
    outputs = layers.Conv2D(2, 1, activation='sigmoid')(conv9)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def LoadValidation(folder, size = (256, 256), batch_size = 32):
    images = tf.keras.utils.image_dataset_from_directory(
        folder,
        image_size = size, 
        batch_size = batch_size,
        label_mode = None,
)
    return images

def MakeGray(images):
    gray_images = images.map(lambda image: EditImage(image)[1])
    
    return gray_images

def CombineWithGray(after_ai, gray_images):
    images = []
    for i, hs_image in enumerate(after_ai):
        image = np.zeros((hs_image.shape[0], hs_image.shape[1], 3))
        image[:, :, 0] = hs_image[:, :, 0] * 179
        image[:, :, 1] = hs_image[:, :, 1] * 255
        image[:, :, 2] = gray_images[i]
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        images.append(image)
    
    return images

def LoadImages(folder, size = (256, 256)):
    images, gray = [], []
    for image_path in os.listdir(folder):
        image = cv2.resize(cv2.imread(os.path.join(folder, image_path)), size)
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        gray.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    
    return images, gray