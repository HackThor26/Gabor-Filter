import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def calculate_std(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f'No se pudo abrir la imagen en la ruta: {image_path}')
    std_dev = np.std(image)
    return std_dev

def calculate_avr(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f'No se pudo abrir la imagen en la ruta: {image_path}')
    average = np.average(image)
    return average

def calculate_eta(λ, σ_x, σ_y):
    return 1 / (2 * np.pi * λ * σ_x * σ_y)

def gabor_kernel(x, y, λ, θ, ϕ, σ_x, σ_y):
    x_prime = x * np.cos(θ) + y * np.sin(θ)
    y_prime = -x * np.sin(θ) + y * np.cos(θ)
    
    η = calculate_eta(λ, σ_x, σ_y)
    
    real_part = η * np.exp(-((x_prime**2) / (2 * σ_x**2) + (y_prime**2) / (2 * σ_y**2))) * np.cos(2 * np.pi * x_prime / λ + ϕ)
    imag_part = η * np.exp(-((x_prime**2) / (2 * σ_x**2) + (y_prime**2) / (2 * σ_y**2))) * np.sin(2 * np.pi * x_prime / λ + ϕ)
    
    return real_part, imag_part

def generate_gabor_kernels(λ_values, θ_values, ϕ, σ_x_values, σ_y_values, kernel_size=3):
    half_size = kernel_size // 2
    x, y = np.meshgrid(np.linspace(-half_size, half_size, kernel_size), 
                       np.linspace(-half_size, half_size, kernel_size))
    
    kernels = []
    for λ in λ_values:
        for θ in θ_values:
            for σ_x in σ_x_values:
                for σ_y in σ_y_values:
                    real_part, imag_part = gabor_kernel(x, y, λ, θ, ϕ, σ_x, σ_y)
                    kernels.append((real_part, imag_part))
    
    return kernels

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

def calculate_ssim(original, processed):
    ssim_value = ssim(original, processed)
    return ssim_value

def process_image(image, λ_values, θ_values, ϕ, σ_x_values, σ_y_values):
    kernels = generate_gabor_kernels(λ_values, θ_values, ϕ, σ_x_values, σ_y_values)
    
    filtered_images = apply_gabor_filter(image, kernels)
    edge_images = preliminary_edge_detection(filtered_images)
    
    thresholds_and_images = [apply_adaptive_threshold(img) for img in edge_images]
    thresholded_images = [thresh_img for _, thresh_img in thresholds_and_images]
    thresholds = [thresh for thresh, _ in thresholds_and_images]
    
    nms_images = [non_maximum_suppression(img) for img in thresholded_images]
    
    weights = [1.0] * len(nms_images)
    
    fused_image = multi_scale_fusion(nms_images, weights)
    
    return filtered_images, thresholded_images, fused_image, thresholds

def process_dataset(image_paths, λ_values, θ_values, ϕ, σ_x_values, σ_y_values):
    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f'No se pudo abrir la imagen en la ruta: {path}')
            continue
        
        original_image = image.copy()
        
        filtered_images, thresholded_images, final_image, thresholds = process_image(image, λ_values, θ_values, ϕ, σ_x_values, σ_y_values)
        
        psnr = calculate_psnr(original_image, final_image)
        ssim_value = calculate_ssim(original_image, final_image)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image, cmap='gray')
        plt.title('Imagen Original')
        
        plt.subplot(1, 2, 2)
        plt.imshow(final_image, cmap='gray')
        plt.title(f'Resultado Final\nPSNR: {psnr:.2f} dB\nSSIM: {ssim_value:.4f}')
        
        plt.show()

        for i, threshold in enumerate(thresholds):
            print(f'Umbral de Otsu para la imagen {i+1}: {threshold:.2f}')
        
        print(f'Valores de σ_x: {σ_x_values}, σ_y: {σ_y_values}')

# Rutas del dataset
image_paths = [
    #r'C:\Users\Lenovo\OneDrive\Escritorio\Gabor Filter\Dataset\arandano.jpg',
    #r'C:\Users\Lenovo\OneDrive\Escritorio\Gabor Filter\Dataset\oso.jpg',
    #r'C:\Users\Lenovo\OneDrive\Escritorio\Gabor Filter\Dataset\caballo.jpg',
    #r'C:\Users\Lenovo\OneDrive\Escritorio\Gabor Filter\Dataset\estrella.jpg',
    #r'C:\Users\Lenovo\OneDrive\Escritorio\Gabor Filter\Dataset\koala.jpg',
    #r'C:\Users\Lenovo\OneDrive\Escritorio\Gabor Filter\Dataset\flor.jpg',
    #r'C:\Users\Lenovo\OneDrive\Escritorio\Gabor Filter\Dataset\rostro.jpg',
    r'C:\Users\Lenovo\OneDrive\Escritorio\Gabor Filter\Dataset\_lena.png',
    #r'C:\Users\Lenovo\OneDrive\Escritorio\Gabor Filter\Dataset\frutas.png',
]

# Parámetros
λ_values = [1/0.15, 1/0.3, 1/0.45]
θ_values = [0, np.pi/4, np.pi/2, 3*np.pi/4]
ϕ = 3*np.pi/4
σ_x_values = [1, 5, 10] #[0.01, 0.2, 0.5]
σ_y_values = [1, 5, 10] #[0.01, 0.2, 0.5]

# Procesar todas las imágenes en el dataset
process_dataset(image_paths, λ_values, θ_values, ϕ, σ_x_values, σ_y_values)
