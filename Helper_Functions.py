import cv2
import numpy as np
import json


coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
class Helper_Functions:

    def __init__(self, image_path):
        self.image_path = image_path

    def draw_boxes(self, image_path, bounding_boxes, debug_mode):
        # Load the original image
        image = cv2.imread(image_path)

        # Draw all bounding boxes on the original image
        for boxes in bounding_boxes:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle on image

        if debug_mode:
            # Display the image with bounding boxes
            cv2.imshow('Bounding Boxes', image)
            cv2.waitKey(0)

    def create_binary_mask(self, bounding_boxes, debug_mode):
        # Load the original image
        image = cv2.imread(self.image_path)

        # Create an empty binary image
        binary_mask = np.zeros_like(image)

        # Fill in the regions corresponding to the bounding boxes
        for boxes in bounding_boxes:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
                binary_mask[y1:y2, x1:x2] = 255  # Fill in the region

                # Draw bounding box in green on the binary mask
                cv2.rectangle(binary_mask, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if debug_mode:
            # Display the binary mask
            cv2.imshow('Binary Mask', binary_mask)
            cv2.waitKey(0)

        return binary_mask





    def prepare_YOLO_Data(self, bounding_boxes, confidences, class_ids, threshold):
        yolo_labels_with_boxes = list(zip(bounding_boxes, class_ids, confidences))

        # Apply labels from coco classes list and format bounding boxes correctly
        formatted_yolo_labels_with_bboxes = [
            (bbox[0], coco_classes[int(label[0])], confidences[0])
            for bbox, label, confidences in yolo_labels_with_boxes
            if confidences[0] >= threshold
        ]

        return formatted_yolo_labels_with_bboxes

    @staticmethod
    def draw_bboxes_with_labels(frame, labels_with_boxes, yolo_labels_with_boxes):
        """
        Draws bounding boxes and labels on the image.

        Parameters
        ----------
        frame : ndarray
            The original image frame.
        labels_with_boxes : list
            List of tuples containing bounding box coordinates and labels from the webcrawler.
        yolo_labels_with_boxes : list
            List of tuples containing bounding box coordinates and labels from YOLO.
        """
        for (bbox, label, confidences) in labels_with_boxes + yolo_labels_with_boxes:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidences:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('Image with Labels', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def merge_lists_and_write_to_json(self, list1, list2, image_id=None):
        merged_list = list1 + list2

        # Create a dictionary for each item in the merged list
        formatted_list = [
            {
                'image_id': image_id,
                'category_id': item[0],
                'bbox': item[1],
                'score': item[2]
            }
            for item in merged_list
        ]

        # Read existing data
        try:
            with open('results.json', 'r') as f:
                existing_data = json.load(f)
        except FileNotFoundError:
            existing_data = []

        # Append new data to existing data
        combined_data = existing_data + formatted_list

        # Write combined data to file
        with open('results.json', 'w') as f:
            json.dump(combined_data, f)







