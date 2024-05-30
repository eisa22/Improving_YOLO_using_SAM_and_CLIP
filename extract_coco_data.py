import json
from pycocotools.coco import COCO

# Pfade zu den COCO-Annotationsdateien und Zielverzeichnissen
annotations_path = "Datasets/Coco/annotations/instances_val2017.json"
link_results_file = "Datasets/Coco/annotations/link_results.txt"
output_annotations_path = "Datasets/Coco/annotations/instances_val2017_subset.json"

# Links aus der Datei lesen
with open(link_results_file, "r") as f:
    image_urls = [line.strip() for line in f]

# Extrahiere Bild-IDs aus den URLs
image_ids = [int(url.split('/')[-1].split('.')[0]) for url in image_urls]

# COCO-Instanz laden
coco = COCO(annotations_path)

# Annotations für die ausgewählten Bilder extrahieren
subset_annotations = {
    "info": coco.dataset["info"],
    "licenses": coco.dataset["licenses"],
    "images": [],
    "annotations": [],
    "categories": coco.dataset["categories"]
}

# Annotations und Bildinformationen kopieren
for img_id in image_ids:
    img_info = coco.loadImgs(img_id)[0]
    subset_annotations["images"].append(img_info)

    # Annotations kopieren
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    subset_annotations["annotations"].extend(anns)

# Subset-Annotations als JSON speichern
with open(output_annotations_path, "w") as f:
    json.dump(subset_annotations, f, indent=4)

print(f"Subset-Annotations wurden gespeichert in {output_annotations_path}")
