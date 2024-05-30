import cv2
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import os

# Pfad zur COCO-Annotationsdatei und Zielverzeichnisse
annotations_path = "Datasets/Coco/annotations/instances_val2017.json"
link_results_file = "Datasets/Coco/annotations/link_results.txt"

# COCO-Instanz laden
coco = COCO(annotations_path)

# Bild-IDs und URLs extrahieren
image_ids = coco.getImgIds()
image_urls = ["http://images.cocodataset.org/val2017/{:012d}.jpg".format(img_id) for img_id in image_ids]

def show_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    plt.imshow(img)
    plt.axis('off')  # Achsen ausblenden
    plt.show()

def main():
    with open(link_results_file, "a") as f:
        for url in image_urls:
            show_image(url)
            key = input("Drücken Sie 'y', um den Link zu speichern, 'n' um zum nächsten Bild zu wechseln oder 'x' um das Programm zu beenden: ").strip().lower()
            if key == 'y':
                f.write(url + "\n")
                print(f"Link gespeichert: {url}")
            elif key == 'n':
                print("Link nicht gespeichert. Weiter zum nächsten Bild.")
            elif key == 'x':
                print("Programm beendet.")
                break
            else:
                print("Ungültige Eingabe. Weiter zum nächsten Bild.")

if __name__ == "__main__":
    main()
