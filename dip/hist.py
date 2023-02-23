import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('pout-dark.jpg', 0)

# Apply histogram equalization
equalized_img = cv2.equalizeHist(img)

# Display the original and equalized images
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.xticks([]), plt.yticks([])

plt.subplot(1,2,2)
plt.imshow(equalized_img, cmap='gray')
plt.title('Equalized Image')
plt.xticks([]), plt.yticks([])

plt.show()
