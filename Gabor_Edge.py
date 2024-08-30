import numpy as np
import cv2
import matplotlib.pyplot as plt

def gabor_kernel(x, y, λ, θ, ϕ, σ, γ):
    x_prime = x * np.cos(θ) + y * np.sin(θ)
    y_prime = -x * np.sin(θ) + y * np.cos(θ)
    
    real_part = np.exp(-(x_prime**2 + γ**2 * y_prime**2) / (2 * σ**2)) * np.cos(2 * np.pi * x_prime / λ + ϕ)
    imag_part = np.exp(-(x_prime**2 + γ**2 * y_prime**2) / (2 * σ**2)) * np.sin(2 * np.pi * x_prime / λ + ϕ)
    
    return real_part, imag_part

def generate_gabor_kernels(λ, θ, ϕ, σ, γ, kernel_size=51):
    half_size = kernel_size // 2
    x, y = np.meshgrid(np.linspace(-half_size, half_size, kernel_size), 
                       np.linspace(-half_size, half_size, kernel_size))
    
    real_part, imag_part = gabor_kernel(x, y, λ, θ, ϕ, σ, γ)
    return [(real_part, imag_part)]

def apply_gabor_filter(image, kernels):
    filtered_images = []
    for real_part, imag_part in kernels:
        real_response = cv2.filter2D(image, cv2.CV_64F, real_part)
        imag_response = cv2.filter2D(image, cv2.CV_64F, imag_part)
        filtered_images.append((real_response, imag_response))
    return filtered_images

def preliminary_edge_detection(filtered_images):
    edge_magnitudes = [np.sqrt(real**2 + imag**2) for real, imag in filtered_images]
    return edge_magnitudes

def apply_adaptive_threshold(image):
    image_uint8 = np.uint8(np.clip(image, 0, 255))
    _, thresholded_image = cv2.threshold(image_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _, thresholded_image

def non_maximum_suppression(image):
    dilated = cv2.dilate(image, None)
    return np.uint8(dilated == image)

def multi_scale_fusion(filtered_images, weights):
    images_normalized = [np.clip(img, 0, 255) for img in filtered_images]
    weighted_images = [img * weight for img, weight in zip(images_normalized, weights)]
    fused_image = np.sum(weighted_images, axis=0)
    fused_image = np.clip(fused_image, 0, 255)
    return np.uint8(fused_image)

def calculate_psnr(original, processed):
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def process_image(image, λ, θ, ϕ, σ, γ):
    kernels = generate_gabor_kernels(λ, θ, ϕ, σ, γ)
    
    filtered_images = apply_gabor_filter(image, kernels)
    edge_images = preliminary_edge_detection(filtered_images)
    
    thresholds_and_images = [apply_adaptive_threshold(img) for img in edge_images]
    thresholded_images = [thresh_img for _, thresh_img in thresholds_and_images]
    thresholds = [thresh for thresh, _ in thresholds_and_images]
    
    nms_images = [non_maximum_suppression(img) for img in thresholded_images]
    # Ponderaciones para la fusión multiescala (puedes ajustar estos valores según tus necesidades)
    weights = [1.0] * len(nms_images)  # Todos los filtros tienen el mismo peso por defecto
    
    fused_image = multi_scale_fusion(nms_images, weights)
    
    return filtered_images, thresholded_images, fused_image, thresholds

def process_dataset(image_paths, λ, θ, ϕ, σ, γ):
    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f'No se pudo abrir la imagen en la ruta: {path}')
            continue
        
        original_image = image.copy()
        
        filtered_images, thresholded_images, final_image, thresholds = process_image(image, λ, θ, ϕ, σ, γ)

        # Calcular el PSNR
        psnr = calculate_psnr(original_image, final_image)

        # Mostrar la imagen original y el resultado final con PSNR
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image, cmap='gray')
        plt.title('Imagen Original')
        
        plt.subplot(1, 2, 2)
        plt.imshow(final_image, cmap='gray')
        plt.title(f'Resultado Final\nPSNR: {psnr:.2f} dB')
        
        plt.show()

        for i, threshold in enumerate(thresholds):
            print(f'Umbral de Otsu para la imagen {i+1}: {threshold:.2f}')

# Parámetros fijos para el filtro de Gabor
λ = 1 / 0.15
θ = np.pi / 4
ϕ = 1 * np.pi / 4
σ = 0.2
γ = 200

# Rutas del dataset
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

# Procesar todas las imágenes en el dataset
process_dataset(image_paths, λ, θ, ϕ, σ, γ)