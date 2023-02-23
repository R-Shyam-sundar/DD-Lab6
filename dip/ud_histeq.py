import cv2
import matplotlib.pyplot as plt

def hist_eq(img):
    # Create a CLAHE object for histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    # Apply histogram equalization to the input image
    equalized_img = clahe.apply(img)

    return equalized_img

# Load the input image
img = cv2.imread('pout-dark.jpg', 0)

# Perform histogram equalization
equalized_img = hist_eq(img)

# Display the input and equalized images
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title('Input Image')
plt.xticks([]), plt.yticks([])

plt.subplot(1,2,2)
plt.imshow(equalized_img, cmap='gray')
plt.title('Equalized Image')
plt.xticks([]), plt.yticks([])

plt.show()
