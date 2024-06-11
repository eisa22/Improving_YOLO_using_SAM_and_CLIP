import YOLO_V8
import Helper_Functions
import Validation
import OpenAI_Key
from pycocotools.coco import COCO

import requests
from PIL import Image, ImageDraw
from io import BytesIO
import numpy as np


# Control Panel
camera_idx = 0
webcam = False
conf_threshold = 0.5
conf_threshold_crawler = 0.1
sam_checkpoint = 'SegmentAnythingModel/sam_vit_h_4b8939.pth'
openAI_key = OpenAI_Key.openAI_key

debug_mode = True
enable_crawler = True
enable_evaluation_mode = False

# Initialize COCO ground truth
coco_gt = COCO('Datasets/Coco/annotations/instances_val2017_subset.json')

iou_scores = []

def draw_bounding_boxes(bbox_list, image_url):
    # Fetch the image from the URL
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    # Create a drawing context
    draw = ImageDraw.Draw(img)

    # Iterate through the list of bounding boxes and draw each one
    for bbox_data in bbox_list:
        bbox = bbox_data['bbox']
        x, y, width, height = bbox
        draw.rectangle([x, y, x + width, y + height], outline="red", width=2)

    # Display the image with bounding boxes
    img.show()


if __name__ == "__main__":

    yolo = YOLO_V8.YOLO_V8('yolov8n.pt')  # Initialize the YOLO_V8 class with the model path
    image_ids = coco_gt.getImgIds()
    all_ious = []

    with open("Results_IoU.txt", "w") as f:
        for image_id in image_ids:
            img_info = coco_gt.loadImgs(image_id)[0]
            print("Processing image: ", img_info['file_name'])
            f.write(f"Processing image: {img_info['file_name']}\n")
            image_url = img_info['coco_url']
            image, pred_boxes, pred_class_ids = yolo.get_detections(image_url)
            # Get ground truth annotations
            ann_ids = coco_gt.getAnnIds(imgIds=image_id)
            anns = coco_gt.loadAnns(ann_ids)
            gt_boxes = []
            gt_class_ids = []
            for ann in anns:
                bbox = ann['bbox']
                # Convert bbox format from [x, y, width, height] to [xmin, ymin, xmax, ymax]
                gt_boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                gt_class_ids.append(ann['category_id'])
            yolo.plot_detections(image, pred_boxes, pred_class_ids, gt_boxes, gt_class_ids)
            # Calculate IoU for matched bounding boxes by class names
            matched_ious = []
