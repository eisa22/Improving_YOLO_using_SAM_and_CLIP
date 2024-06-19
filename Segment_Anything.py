import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import json


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

    def segment_images(self):
        # Load the data from results.json
        with open('results.json') as f:
            results_data = json.load(f)

        # Initialize the SegmentAnything class
        segmenter = SegmentAnything(sam_checkpoint=self.sam_checkpoint)

        # Process each image in the results data
        for image in results_data:
            # Load the image
            img = cv2.imread(image['image_id'])

            # Use the SegmentAnything class to segment the image
            segments = segmenter.predictor.predict(img)

            # Get the bounding boxes for each segment
            bounding_boxes = []
            for segment in segments:
                x, y, w, h = cv2.boundingRect(segment)
                bounding_boxes.append({'bbox': [x, y, w, h]})

            # Replace the predictions with the new bounding boxes
            image['predictions'] = bounding_boxes

        # Write the results to Results_SegmentAnything.json
        with open('Results_SegmentAnything.json', 'w') as f:
            json.dump(results_data, f, indent=4)






