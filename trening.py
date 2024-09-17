from utilities import *
import tensorflow as tf
import matplotlib.pyplot as plt

def main():
    # Pravljenje modela
    model = MakeModel(input_size = (256, 256, 1))
    model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])

    # Učitavanje fajlova
    folder_path = './mid_smaller_dataset'
    dataset, val = LoadImagesFromFolder(folder_path, size = (256, 256), batch_size = 16)
    dataset = PreprocessImages(dataset)
    val = PreprocessImages(val)

    # Treniranje modela
    history = model.fit(dataset, epochs = 100, validation_data = val)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc = 'upper left')
    plt.savefig("Model loss")

    # Čuvanje istreniranog modela
    model.save('AI.h5')

if __name__ == '__main__':
    main()