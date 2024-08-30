import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.filters import gabor

# Parámetros del filtro de Gabor
lambda_ = 2.0       # Longitud de onda
theta = np.pi / 4   # Ángulo en radianes (45 grados)

# Cargar y procesar la imagen
image_paths = [
    r'C:\Users\Lenovo\OneDrive\Escritorio\Gabor Filter\Dataset\arandano.jpg',
    r'C:\Users\Lenovo\OneDrive\Escritorio\Gabor Filter\Dataset\oso.jpg',
    r'C:\Users\Lenovo\OneDrive\Escritorio\Gabor Filter\Dataset\caballo.jpg',
    r'C:\Users\Lenovo\OneDrive\Escritorio\Gabor Filter\Dataset\estrella.jpg',
    r'C:\Users\Lenovo\OneDrive\Escritorio\Gabor Filter\Dataset\koala.jpg',
    r'C:\Users\Lenovo\OneDrive\Escritorio\Gabor Filter\Dataset\flor.jpg',
    r'C:\Users\Lenovo\OneDrive\Escritorio\Gabor Filter\Dataset\rostro.jpg',
    r'C:\Users\Lenovo\OneDrive\Escritorio\Gabor Filter\Dataset\_lena.png',
    r'C:\Users\Lenovo\OneDrive\Escritorio\Gabor Filter\Dataset\frutas.png',
]

for image_path in image_paths:
    # Cargar imagen y convertir a escala de grises
    image = io.imread(image_path, as_gray=True)

    # Aplicar padding a la imagen
    padding = 1
    image_padded = np.pad(image, pad_width=((padding, padding), (padding, padding)), mode='constant', constant_values=0)

    # Aplicar el filtro de Gabor utilizando la función de scikit-image
    convoluted_image_real, convoluted_image_imag = gabor(image_padded, frequency=1/lambda_, theta=theta)

    # Recortar el padding
    convoluted_image_real = convoluted_image_real[padding:-padding, padding:-padding]
    convoluted_image_imag = convoluted_image_imag[padding:-padding, padding:-padding]

    # Calcular la magnitud de la parte real e imaginaria
    convoluted_image_magnitude = np.sqrt(convoluted_image_real**2 + convoluted_image_imag**2)

    # Graficar el filtro de Gabor y la imagen convolucionada
    fig, ax = plt.subplots(1, 3, figsize=(21, 7))

    # Imagen original
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Imagen Original')
    ax[0].axis('off')

    # Imagen convolucionada (parte real)
    ax[1].imshow(convoluted_image_real, cmap='gray')
    ax[1].set_title('Parte Real de la Convolución')
    ax[1].axis('off')

    # Imagen convolucionada (magnitud)
    ax[2].imshow(convoluted_image_magnitude, cmap='gray')
    ax[2].set_title('Magnitud de la Convolución')
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()
