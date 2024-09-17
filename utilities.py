import tensorflow as tf
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
