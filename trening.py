from utilities import *
from sklearn.model_selection import train_test_split
import tensorflow as tf

def main():
    # Pravljenje modela
    model = MakeModel()
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError()) # treba staviti moju loss funkciju

    # Učitavanje fajlova
    folder_path = './dataset'
    color_images, gray_images = LoadImagesFromFolder(folder_path)

    # Podela slika za trening i test
    train_gray_images, test_gray_images, train_color_images, test_color_images = train_test_split(gray_images, color_images, test_size=0.2, random_state=42)

    # Treniranje modela
    history = model.fit(train_gray_images, train_color_images, validation_split=0.1, epochs=200, batch_size=None)

    # Čuvanje istreniranog modela
    model.save('AI.h5')

if __name__ == '__main__':
    main()