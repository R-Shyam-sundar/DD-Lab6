import cv2
import numpy as np
import matplotlib.pyplot as plt

def hist_match(input_img, ref_img):
    # Create a CLAHE object for histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    # Apply histogram equalization to the reference image
    equalized_ref_img = clahe.apply(ref_img)

    # Get the histogram of the input and reference images
    hist_input = cv2.calcHist([input_img], [0], None, [256], [0,256])
    hist_ref = cv2.calcHist([equalized_ref_img], [0], None, [256], [0,256])

    # Normalize the histograms
    hist_input_norm = hist_input / np.sum(hist_input)
    hist_ref_norm = hist_ref / np.sum(hist_ref)

    # Compute the cumulative distribution functions (CDFs)
    cdf_input = np.cumsum(hist_input_norm)
    cdf_ref = np.cumsum(hist_ref_norm)

    # Compute the mapping function
    mapping = np.zeros((256,), dtype=np.uint8)
    for i in range(256):
        j = 255
        while j >= 0 and cdf_input[i] <= cdf_ref[j]:
            j -= 1
        mapping[i] = j

    # Apply the mapping function to the input image
    matched_img = cv2.LUT(input_img, mapping)

    return matched_img

# Load the input and reference images
img_dark = cv2.imread('pout-dark.jpg', 0)
img_bright = cv2.imread('pout-bright.jpg', 0)

# Apply histogram matching
matched_img = hist_match(img_dark, img_bright)

# Display the input, reference, and matched images
plt.subplot(1,3,1)
plt.imshow(img_dark, cmap='gray')
plt.title('Input Image')
plt.xticks([]), plt.yticks([])

plt.subplot(1,3,2)
plt.imshow(img_bright, cmap='gray')
plt.title('Reference Image')
plt.xticks([]), plt.yticks([])

plt.subplot(1,3,3)
plt.imshow(matched_img, cmap='gray')
plt.title('Matched Image')
plt.xticks([]), plt.yticks([])

plt.show()