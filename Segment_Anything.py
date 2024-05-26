import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor


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





