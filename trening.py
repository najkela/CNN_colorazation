from utilities import *
import tensorflow as tf
import cProfile

def main():
    # Pravljenje cProfile čitača
    pr = cProfile.Profile()
    pr.enable()

    # Pravljenje modela
    model = MakeLargerModel()
    model.compile(optimizer = 'adam', loss = 'mse')

    # Učitavanje fajlova
    folder_path = './mid_smaller_dataset'
    dataset = LoadImagesFromFolder(folder_path)
    dataset = dataset.batch(32)

    # Treniranje modela
    model.fit(dataset, epochs=2)

    # Čuvanje istreniranog modela
    model.save('AI.h5')

    # Analiza pročitanih vrednosti sa cProfilea
    pr.disable()
    pr.print_stats()

if __name__ == '__main__':
    main()