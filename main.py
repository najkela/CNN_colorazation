from tensorflow.keras.models import load_model # type: ignore
import tensorflow as tf
from utilities import *
import matplotlib.pyplot as plt

def main():
    # Učitavanje istreniranog modela
    model = load_model('AI_my_loss_50e.h5', compile = False)

    # Učitavanje slika za obradu
    folder_path = './moje slike'
    _, gray_images = LoadImagesFromFolder(folder_path, (128, 128))
    originals = LoadOriginalImages(folder_path, (128, 128))

    # Obrada slika
    num = 6 # broj slika koje ćemo obraditi
    after_ai = model.predict(gray_images[:num])
    after_ai = CombineWithGray(after_ai, gray_images, num)
    
    # Prikaz obrađenih slika
    fig, ax = plt.subplots(3, num, figsize = (200, 100))
    fig.set_label("Rezultati")
    for i in range(num): 
        ax[0][i].set_axis_off()
        ax[1][i].set_axis_off()
        ax[2][i].set_axis_off()
        ax[0][i].imshow(originals[i])
        ax[1][i].imshow(gray_images[i], cmap = 'gray')
        ax[2][i].imshow(after_ai[i])
    fig.tight_layout()
    plt.show()
    

if __name__ == '__main__':
    main()