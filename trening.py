from utilities import *
import tensorflow as tf

def main():
    # Pravljenje modela
    model = MakeLargerModel()
    model.compile(optimizer = 'adam', loss = CustomLoss)

    # Učitavanje fajlova
    folder_path = './mid_smaller_dataset'
    color_images, gray_images = LoadImagesFromFolder(folder_path)

    # Treniranje modela
    print("pocinje trening")
    history = model.fit(gray_images, color_images, validation_split=0.2, epochs=2, batch_size=32)

    # Čuvanje istreniranog modela
    model.save('AI.h5')

if __name__ == '__main__':
    main()