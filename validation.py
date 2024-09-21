from utilities import *
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # type: ignore

def main():
    folder = './moje slike'
    size = (256, 256)
    originals = LoadValidation(folder, size)
    gray_images = MakeGray(originals)

    model_name = 'AI_bebic.h5'
    model = load_model(model_name, compile = False)

    after_ai = model.predict(gray_images)

    del model, originals, gray_images

    originals, gray_images = LoadImages(folder, size)
    after_ai = CombineWithGray(after_ai, gray_images)
    num = len(originals)

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