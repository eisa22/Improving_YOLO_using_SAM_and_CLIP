import cv2
import numpy as np

# Bild laden
image = cv2.imread('Images/Table_with_objects.jpg')

# Höhe und Breite des Bildes ermitteln
height, width = image.shape[:2]

# Definieren Sie die Größe des Kernels (Fensters)
kernel_size = 100

# Erstellen Sie eine Liste, um die Subbilder zu speichern
subimages = []

# Verwenden Sie eine verschachtelte Schleife, um über das Bild zu iterieren
for y in range(0, height, kernel_size):
    for x in range(0, width, kernel_size):
        # Extrahieren Sie das Fenster aus dem Bild
        subimage = image[y:y+kernel_size, x:x+kernel_size]
        subimages.append(subimage)

# Speichern Sie jedes Subbild
for i, subimage in enumerate(subimages):
    cv2.imwrite(f'Images/Sliding_Window_Results/subimage{i}.jpg', subimage)