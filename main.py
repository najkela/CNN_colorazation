from tensorflow.keras.models import load_model # type: ignore
import tensorflow as tf
from utilities import *
import matplotlib.pyplot as plt

def main():
    # Učitavanje istreniranog modela
    model = load_model('AI_mse_500_ljudi.h5')

    # Učitavanje slika za obradu
    folder_path = './moje slike'
    color_images, gray_images = LoadImagesFromFolder(folder_path)

    # Obrada slika
    num = 2 # broj slika koje ćemo obraditi
    after_ai = model.predict(gray_images[-num:])
    
    # Prikaz obrađenih slika
    fig, ax = plt.subplots(3, num, figsize = (200, 100))
    fig.set_label("Rezultati")
    for i in range(num): 
        ax[0][i].set_axis_off()
        ax[1][i].set_axis_off()
        ax[2][i].set_axis_off()
        ax[0][i].imshow(color_images[-num+i])
        ax[1][i].imshow(gray_images[-num+i], cmap = 'gray')
        ax[2][i].imshow(after_ai[i])
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()