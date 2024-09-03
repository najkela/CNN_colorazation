from utilities import *
import tensorflow as tf
import cProfile

def main():
    # Pravljenje cProfile čitača
    pr = cProfile.Profile()
    pr.enable()

    # Pravljenje modela
    model = MakeLargerModel()
    model.compile(optimizer = 'adam', loss = CustomLoss)

    # Učitavanje fajlova
    folder_path = './all_in_one_mini_dataset'
    color_images, gray_images = LoadImagesFromFolder(folder_path)

    # Treniranje modela
    history = model.fit(gray_images, color_images, validation_split=0.2, epochs=2, batch_size=32)

    # Čuvanje istreniranog modela
    model.save('AI.h5')

    # Analiza pročitanih vrednosti sa cProfilea
    pr.disable()
    pr.print_stats()

if __name__ == '__main__':
    main()