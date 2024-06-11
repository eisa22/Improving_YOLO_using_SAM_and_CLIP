import YOLO_V8
import Helper_Functions
import Validation
import OpenAI_Key
from pycocotools.coco import COCO
import json
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


if __name__ == "__main__":
    yolo = YOLO_V8.YOLO_V8('yolov8n.pt')  # Initialize the YOLO_V8 class with the model path
    image_ids = coco_gt.getImgIds()
    all_ious = []

    all_results = []

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
            gt_urls = []
            gt_ids = []
            gt_boxes = []
            gt_class_ids = []
            for ann in anns:
                bbox = ann['bbox']
                # Convert bbox format from [x, y, width, height] to [xmin, ymin, xmax, ymax]
                gt_boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                gt_class_ids.append(ann['category_id'])
                gt_urls.append(image_url)
                gt_ids.append(image_id)
            result = yolo.plot_detections_and_save_results(image, pred_boxes, pred_class_ids, gt_boxes, gt_class_ids, image_id, image_url)
            all_results.append(result)

    # Save all results to JSON
    with open('results.json', 'w') as f:
        json.dump(all_results, f, indent=4)


    # Calculate IoU ___________________________________________________
    with open('results.json') as f:
        results = json.load(f)

    validation = Validation.Validation()
    results_iou, overall_iou = validation.process_results(results)

    # Save IoU results to results_IoU.txt in the current working directory
    results_file_path = "results_IoU.txt"
    with open(results_file_path, "w") as f:
        for image_id, iou in results_iou.items():
            f.write(f"Image ID: {image_id}, IoU: {iou}\n")
        f.write(f"Overall IoU: {overall_iou}\n")

    print("Overall IoU:", overall_iou)


    # Correct BBox and Labels--------------------------------------------------------------------------------
    bbox_and_labels_results, overall_bbox_and_labels_accuracy = validation.check_bbox_and_labels(results)

    # Save BBox and Labels results to results_bbox_and_labels.txt in the current working directory
    bbox_and_labels_file_path = "results_bbox_and_labels.txt"
    with open(bbox_and_labels_file_path, "w") as f:
        for image_id, accuracy in bbox_and_labels_results.items():
            f.write(f"Image ID: {image_id}, Correct BBox and Labels: {accuracy}\n")
        f.write(f"Overall Accuracy of BBox and Labels: {overall_bbox_and_labels_accuracy}\n")

    print("Overall Accuracy of BBox and Labels:", overall_bbox_and_labels_accuracy)

