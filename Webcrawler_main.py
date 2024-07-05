import YOLO_V8
import Helper_Functions
import Validation
import OpenAI_Key
import OpenAI_API
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
openAI_key = OpenAI_Key.openAI_key

debug_mode = True
enable_crawler = False

if __name__ == "__main__":
    if enable_crawler:
        webcrawler = OpenAI_API.Webcrawler(openAI_key)  # Initialize the Webcrawler class with the API key
        webcrawler.filter_and_send_to_api('results.json', 0.4, 'results_ChatGPT.json')

    validation = Validation.Validation()
    # Calculate IoU and bboxes ___________________________________________________
    with open('results_ChatGPT.json') as f:
        results = json.load(f)

        # Correct BBox and Labels--------------------------------------------------------------------------------
        bbox_and_labels_results, overall_bbox_and_labels_accuracy = validation.calculate_accuracy('results_ChatGPT.json', 'results_GPT.txt')

        # Save BBox and Labels results to results_bbox_and_labels.txt in the current working directory
        bbox_and_labels_file_path = "results_bbox_and_labels.txt"
        with open(bbox_and_labels_file_path, "w") as f:
            for image_id, accuracy in bbox_and_labels_results.items():
                f.write(f"Image ID: {image_id}, Correct BBox and Labels: {accuracy}\n")
            f.write(f"Overall Accuracy of BBox and Labels: {overall_bbox_and_labels_accuracy}\n")

        print("Overall Accuracy of BBox and Labels:", overall_bbox_and_labels_accuracy)

