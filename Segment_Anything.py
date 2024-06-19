import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import json
import requests
from PIL import Image
from io import BytesIO

class SegmentAnything:
    def __init__(self, sam_checkpoint, model_type='vit_h'):
        # Initialize the SAM model
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type
        self._load_sam_model()

    def _load_sam_model(self):
        # Load the SAM model
        try:
            self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
            self.sam.to(device='cuda' if torch.cuda.is_available() else 'cpu')
            self.predictor = SamPredictor(self.sam)
        except Exception as e:
            print(f"Error loading SAM model: {e}")
            raise

    def load_json(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data

    def download_image(self, url):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img

    def get_bounding_boxes(self, image):
        # Convert PIL image to numpy array
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Set the image for the predictor
        self.predictor.set_image(img_array)

        # Get the segmentation mask
        masks, _, _ = self.predictor.predict()

        bounding_boxes = []
        for mask in masks:
            # Find contours and bounding box for each mask
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                bounding_boxes.append((x, y, x + w, y + h))

        return bounding_boxes

    def process_images(self, data):
        results = []
        total_bounding_boxes = 0
        for item in data:
            image_id = item['image_id']
            url = item['url']
            img = self.download_image(url)
            bounding_boxes = self.get_bounding_boxes(img)
            total_bounding_boxes += len(bounding_boxes)
            results.append({
                'image_id': image_id,
                'url': url,
                'bounding_boxes': bounding_boxes
            })
        return results, total_bounding_boxes

    def write_results_to_file(self, results, file_path='Results_SegmentAnything.txt'):
        with open(file_path, 'w') as file:
            for result in results:
                file.write(f"Image ID: {result['image_id']}\n")
                file.write(f"URL: {result['url']}\n")
                file.write("Bounding Boxes:\n")
                for bbox in result['bounding_boxes']:
                    file.write(f"  {bbox}\n")
                file.write("\n")