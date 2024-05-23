import cv2
import numpy as np
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




