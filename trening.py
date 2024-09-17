from utilities import *
import tensorflow as tf
import matplotlib.pyplot as plt

def main():
    log_dir = 'logs'
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                             profile_batch='1, 2')
    size = (128, 128)

    # Pravljenje modela
    model = MakeModel(input_size = (size[0], size[1], 1))
    model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])

    # Učitavanje fajlova
    folder_path = './mid_smaller_dataset'
    dataset, val = LoadImagesFromFolder(folder_path, size = size, batch_size = 32)
    dataset = PreprocessImages(dataset)
    val = PreprocessImages(val)
    del folder_path

    # Treniranje modela
    history = model.fit(dataset, epochs = 2, validation_data = val, callbacks=[tb_callback])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc = 'upper left')
    plt.show()
    del history

    # Čuvanje istreniranog modela
    model.save('AI.h5')

if __name__ == '__main__':
    main()