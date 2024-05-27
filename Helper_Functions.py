import cv2
import numpy as np
import os


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


        if debug_mode:
            # Display the binary mask
            cv2.imshow('Binary Mask', binary_mask)
            cv2.waitKey(0)

        return binary_mask


    def draw_boxes_on_black_areas(self, binary_mask, debug_mode=False):

        _, binary_image = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        black_areas = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            black_areas.append((x, y, w, h))
            print(f"Detected black areas: {black_areas}")

        return black_areas

    def mask_undetected_areas(self, original_image_path, binary_mask, debug_mode=False):
        # Load the original image
        original_image = cv2.imread(original_image_path)
        if original_image is None:
            raise ValueError("The original image is invalid or not found.")

        # Load the binary mask
        if isinstance(binary_mask, str):
            binary_mask = cv2.imread(binary_mask, cv2.IMREAD_GRAYSCALE)
        elif isinstance(binary_mask, np.ndarray) and len(binary_mask.shape) == 3:
            binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)

        if binary_mask is None:
            raise ValueError("The binary mask image is invalid or not found.")

        # Ensure the binary mask has binary values
        _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)

        if debug_mode:
            print("Binary Mask after Thresholding:")
            print(np.unique(binary_mask, return_counts=True))  # Print unique values and their counts

        # Resize the binary mask to match the original image size
        binary_mask = cv2.resize(binary_mask, (original_image.shape[1], original_image.shape[0]))

        # Create a white background
        white_background = np.ones_like(original_image) * 255

        # Invert the binary mask to create a mask for areas YOLO didn't detect
        inverted_mask = cv2.bitwise_not(binary_mask)

        # Create a masked version of the original image
        masked_image = np.where(inverted_mask[:, :, np.newaxis] == 255, original_image, white_background)

        if debug_mode:
            cv2.imshow("Masked Image", masked_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        return masked_image

    def prepare_YOLO_Data(self, bounding_boxes, confidences, class_ids):
        yolo_labels_with_boxes = list(zip(bounding_boxes, class_ids))

        # Apply labels from coco classes list and format bounding boxes correctly
        formatted_yolo_labels_with_bboxes = [
            (bbox[0], coco_classes[int(label[0])]) for bbox, label in yolo_labels_with_boxes
        ]

        return formatted_yolo_labels_with_bboxes







