import os
import requests
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO
import json

# Setting the environment variable to handle the OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

COCO_CATEGORY_ID_LIST = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat',
    10: 'traffic light', 11: 'fire hydrant', 12: 'street sign', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra',
    25: 'giraffe', 26: 'hat', 27: 'backpack', 28: 'umbrella', 29: 'shoe', 30: 'eye glasses', 31: 'handbag',
    32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
    45: 'plate', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
    53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza',
    60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 66: 'mirror', 67: 'dining table',
    68: 'window', 69: 'desk', 70: 'toilet', 71: 'door', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote',
    76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
    83: 'blender', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
    90: 'toothbrush', 91: 'hair brush'
}

YOLO_CATEGORY_ID_LIST = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat',
    9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird',
    15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra',
    23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup',
    42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
    50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch',
    58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
    79: 'toothbrush'
}


class YOLO_V8:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)  # Load the YOLOv8 model

    def get_image_from_url(self, url):
        response = requests.get(url)
        image_array = np.asarray(bytearray(response.content), dtype="uint8")
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image

    def get_detections(self, image_url):
        # Get the image from the URL
        image = self.get_image_from_url(image_url)
        # Run the model on the image
        results = self.model(image)
        # Extract bounding boxes and class IDs
        boxes = []
        class_ids = []
        for result in results:
            for box in result.boxes:
                boxes.append(box.xyxy.tolist())  # xyxy format bounding box
                class_ids.append(int(box.cls.tolist()[0]))
        return image, boxes, class_ids

    def plot_detections_and_save_results(self, image, pred_boxes, pred_class_ids, gt_boxes=None, gt_class_ids=None,
                                         image_id=None, image_url=None):

        # Convert image from BGR (OpenCV format) to RGB (matplotlib format)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Plot the image
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(image_rgb)

        results = {
            "image_id": image_id,
            "url": image_url,
            "predictions": [],
            "ground_truths": []
        }

        # Plot each predicted bounding box
        for i, box in enumerate(pred_boxes):
            xmin, ymin, xmax, ymax = box[0]
            width, height = xmax - xmin, ymax - ymin
            # Create a rectangle patch
            rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            # Add predicted class ID text
            class_name = YOLO_CATEGORY_ID_LIST.get(pred_class_ids[i], "Unknown")
            ax.text(xmin, ymin - 10, class_name, color='red', fontsize=12, weight='bold')

            # Append to results
            results["predictions"].append({
                "bbox": [xmin, ymin, xmax, ymax],
                "category": class_name
            })

        # Plot each ground truth bounding box if provided
        if gt_boxes is not None and gt_class_ids is not None:
            for i, box in enumerate(gt_boxes):
                xmin, ymin, xmax, ymax = box
                width, height = xmax - xmin, ymax - ymin
                # Create a rectangle patch
                rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='g', facecolor='none')
                ax.add_patch(rect)
                # Add ground truth class ID text
                class_name = COCO_CATEGORY_ID_LIST.get(gt_class_ids[i], "Unknown")
                ax.text(xmin, ymin - 10, class_name, color='green', fontsize=12, weight='bold')

                # Append to results
                results["ground_truths"].append({
                    "bbox": [xmin, ymin, xmax, ymax],
                    "category": class_name
                })

        plt.axis('off')  # Hide axes
        plt.show()

        # Return results instead of saving directly to JSON
        return results


# Example usage
if __name__ == "__main__":
    yolo = YOLO_V8('yolov8n.pt')  # Initialize the YOLO_V8 class with the model path
    coco_gt = COCO('path/to/instances_val2017.json')  # Update this path to your COCO annotations

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

