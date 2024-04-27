
import cv2
import numpy as np

# Bild laden
image = cv2.imread('Images/Table_with_objects.jpg')

# Höhe und Breite des Bildes ermitteln
height, width = image.shape[:2]

# Bild in vier Teile zerlegen
subimage1 = image[:height//2, :width//2]
subimage2 = image[:height//2, width//2:]
subimage3 = image[height//2:, :width//2]
subimage4 = image[height//2:, width//2:]

# Unterbilder speichern
cv2.imwrite('Images/Image_Patching/subimage1.jpg', subimage1)
cv2.imwrite('Images/Image_Patching/subimage2.jpg', subimage2)
cv2.imwrite('Images/Image_Patching/subimage3.jpg', subimage3)
cv2.imwrite('Images/Image_Patching/subimage4.jpg', subimage4)