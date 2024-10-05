from utilities import *
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # type: ignore
from skimage.metrics import structural_similarity as SSIM

def main():
    # Učitavanje slika za predikciju modela
    folder = './moje slike'
    size = (256, 256)
    originals = LoadValidation(folder, size)
    gray_images = MakeGray(originals)

    # Učitavanje modela
    model_name = 'AI_bebic.h5'
    model = load_model(model_name, compile = False)

    # Predikcija modela
    after_ai = model.predict(gray_images)

    # Brisanje nepotrebnih podataka
    del model, originals, gray_images

    # Učitavanje slika za prikaz rezultata
    originals, gray_images = LoadImages(folder, size)
    after_ai = CombineWithGray(after_ai, gray_images)
    num = len(originals)

    # Čuvanje valudacije
    SSIMs = []

    # Prikaz prediktovanih slika
    fig, ax = plt.subplots(3, num, figsize = (200, 100))
    fig.set_label("Rezultati")
    for i in range(num): 
        original, predicted = originals[i], after_ai[i]
        ssim = SSIM(original.flatten(), predicted.flatten(), win_size = original.shape[0] - 1)
        SSIMs.append(ssim)

        ax[0][i].set_axis_off()
        ax[1][i].set_axis_off()
        ax[2][i].set_axis_off()
        ax[0][i].imshow(original)
        ax[1][i].imshow(gray_images[i], cmap = 'gray')
        ax[2][i].imshow(predicted)
    fig.tight_layout()
    plt.show()

    print(f"{sum(SSIMs) / num:.2f}")

if __name__ == '__main__':
    main()