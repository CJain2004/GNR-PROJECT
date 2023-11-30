import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from matplotlib import pyplot as plt

def process_image_and_show(image_path, cutoff_frequency, sigma_x, sigma_y, order):
    # Load the original image
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Compute the Fourier Transform
    fft_result = np.fft.fft2(original_image)
    fft_result_shifted = np.fft.fftshift(fft_result)
    magnitude_spectrum = np.log(np.abs(fft_result_shifted) + 1).real

    # Find the radius of the low values - max Cutoff Frequency
    max_magnitude = np.max(magnitude_spectrum)
    threshold = 0.5 * max_magnitude
    low_values_indices = np.where(magnitude_spectrum < threshold)
    radius_low_values = np.sqrt((low_values_indices[0] - fft_result_shifted.shape[0] // 2)**2 +
                            (low_values_indices[1] - fft_result_shifted.shape[1] // 2)**2)
    max_radius_low_values = np.max(radius_low_values)
    cut_off_frequency_max = max_radius_low_values

    if cutoff_frequency > cut_off_frequency_max:
        cutoff_frequency = cut_off_frequency_max
        label_cutoff_warning.config(text="Warning: Cutoff frequency is beyond the maximum allowable value.", fg="red")
        return None

    else:
        label_cutoff_warning.config(text="Max Cutoff Frequency: " + str(cut_off_frequency_max), fg="black")  # Clear the warning message
    
    if sigma_x<5 or sigma_y<5:
        label_gaussian_warning.config(text="Warning: Sigma X and Sigma Y should be greater than 5.", fg="red")
        return None
    else:
        label_gaussian_warning.config(text="Min Sigma X and Sigma Y: 5 " , fg="black")  # Clear the warning message
     
    
    # Gaussian Smoothing
    kernel_size_l = fft_result_shifted.shape[0]
    kernel_size_b = fft_result_shifted.shape[1]
    gaussian_filter = gaussian_kernel(kernel_size_l,kernel_size_b, sigma_y,sigma_x )

    blurred_image = cv2.GaussianBlur(original_image, (5, 5), 0)
    convolved = fft_result_shifted * gaussian_filter
    convolved_image = np.fft.ifftshift(convolved)
    image_restored = np.fft.ifft2(convolved_image).real


    # Butterworth Filter
    lowpass_filtered = butterworth_filter(fft_result_shifted.shape, cutoff_frequency, order)
    highpass_filtered = 1 - lowpass_filtered

    # Apply the filters to the Fourier Transform
    lowpass_result = fft_result_shifted * lowpass_filtered
    highpass_result = fft_result_shifted * highpass_filtered

    # Inverse Fourier Transform
    lowpass_image = np.fft.ifft2(np.fft.ifftshift(lowpass_result)).real
    highpass_image = np.fft.ifft2(np.fft.ifftshift(highpass_result)).real


    # Display images
    plt.figure(figsize=(16, 12))

    plt.subplot(2, 4, 1), plt.imshow(original_image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 4, 2), plt.imshow(np.log(np.abs(fft_result) + 1).real, cmap='gray')
    plt.title('FFT Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 4, 3), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Shifted FFT Image'), plt.xticks([]), plt.yticks([])


    plt.subplot(2, 4, 4), plt.imshow(image_restored, cmap='gray')
    plt.title('Gaussian Smoothing'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 4, 5), plt.imshow(np.log(np.abs(lowpass_result) + 1).real, cmap='gray')
    plt.title('Lowpass Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 4, 6), plt.imshow(np.log(np.abs(highpass_result) + 1).real, cmap='gray')
    plt.title('Highpass Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 4, 7), plt.imshow(lowpass_image, cmap='gray')
    plt.title('Lowpass Time Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 4, 8), plt.imshow(highpass_image, cmap='gray')
    plt.title('Highpass Time Image'), plt.xticks([]), plt.yticks([]) 

    plt.tight_layout()
    plt.show()

def upload_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    entry_path.delete(0, tk.END)
    entry_path.insert(0, file_path)

def process_image():
    image_path = entry_path.get()
    cutoff_frequency = float(entry_cutoff.get())
    sigma_x = float(entry_sigma_x.get())
    sigma_y = float(entry_sigma_y.get())
    order = int(entry_order.get())
    process_image_and_show(image_path, cutoff_frequency, sigma_x, sigma_y, order)

# Butterworth Filter Function
def butterworth_filter(shape, cutoff, order=1):
    rows, cols = shape
    u = np.arange(cols)
    v = np.arange(rows)

    idx = u > cols // 2
    u[idx] -= cols

    idy = v > rows // 2
    v[idy] -= rows

    x = np.fft.fftshift(u)
    y = np.fft.fftshift(v)
    xx, yy = np.meshgrid(x, y)

    radius = np.sqrt(xx ** 2 + yy ** 2)

   
    butterworth = 1 / (1 + (radius / cutoff) ** (2 * order))
    return butterworth

def gaussian_kernel(l, b, sigma_x, sigma_y):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma_x*sigma_y)) * np.exp(-((x-(l-1)/2)**2/(2*sigma_x**2) + (y-(b-1)/2)**2/(2*sigma_y**2))), (l, b))
    return kernel / np.sum(kernel)



# def process_image():
#     image_path = entry_path.get()
#     cutoff_frequency = float(entry_cutoff.get())
#     sigma_x = float(entry_sigma_x.get())
#     sigma_y = float(entry_sigma_y.get())
#     order = int(entry_order.get())

#     # Check if cutoff frequency is beyond the maximum allowable value
#     if cutoff_frequency > cut_off_frequency_max:
#         cutoff_frequency = cut_off_frequency_max
#         label_cutoff_warning.config(text="Warning: Cutoff frequency is beyond the maximum allowable value.", fg="red")

#     else:
#         label_cutoff_warning.config(text="")  # Clear the warning message




# GUI Setup
root = tk.Tk()
root.title("Image Processing App")

# File Path
label_path = tk.Label(root, text="Image Path:")
label_path.grid(row=0, column=0)
entry_path = tk.Entry(root, width=40)
entry_path.grid(row=0, column=1)
button_browse = tk.Button(root, text="Browse", command=upload_image)
button_browse.grid(row=0, column=2)

# Parameters
label_cutoff = tk.Label(root, text="Cutoff Frequency:")
label_cutoff.grid(row=1, column=0)
entry_cutoff = tk.Entry(root)
entry_cutoff.grid(row=1, column=1)

label_cutoff_warning = tk.Label(root, text="", fg="red")  # You can adjust the color as needed
label_cutoff_warning.grid(row=2, column=1 )

label_sigma_x = tk.Label(root, text="Sigma X:")
label_sigma_x.grid(row=3, column=0)
entry_sigma_x = tk.Entry(root)
entry_sigma_x.grid(row=3, column=1)

label_sigma_y = tk.Label(root, text="Sigma Y:")
label_sigma_y.grid(row=4, column=0)
entry_sigma_y = tk.Entry(root)
entry_sigma_y.grid(row=4, column=1)


label_gaussian_warning = tk.Label(root, text="", fg="red")  # You can adjust the color as needed
label_gaussian_warning.grid(row=5, column=1 )


label_order = tk.Label(root, text="Order:")
label_order.grid(row=6, column=0)
entry_order = tk.Entry(root)
entry_order.grid(row=6, column=1)

# Process Button
button_process = tk.Button(root, text="Process Image", command=process_image)
button_process.grid(row=7, column=1)

root.mainloop()
