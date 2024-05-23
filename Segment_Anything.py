import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
import matplotlib.pyplot as plt


class SAMSegmenter:
    def __init__(self, sam_checkpoint_path, model_type="vit_b", grid_size=32):
        self.sam_checkpoint_path = sam_checkpoint_path
        self.model_type = model_type
        self.grid_size = grid_size
        self.sam = self._load_sam_model()
        self.predictor = SamPredictor(self.sam)

    def _load_sam_model(self):
        if self.model_type not in sam_model_registry:
            raise ValueError(
                f"Model type {self.model_type} is not recognized. Use one of {list(sam_model_registry.keys())}")
        return sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint_path)

    def segment_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image at path '{image_path}' not found.")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image)

        height, width, _ = image.shape
        input_points = self._generate_grid_points(height, width)

        input_labels = np.ones(len(input_points))
        masks, _, _ = self.predictor.predict(point_coords=input_points, point_labels=input_labels)

        segmented_images = self._save_masks(masks)
        return segmented_images

    def _generate_grid_points(self, height, width):
        points = []
        for y in range(0, height, self.grid_size):
            for x in range(0, width, self.grid_size):
                points.append([x, y])
        return np.array(points)

    def _save_masks(self, masks):
        segmented_images = []
        for i, mask in enumerate(masks):
            mask_image = (mask * 255).astype(np.uint8)
            mask_image_path = f"mask_{i}.png"
            cv2.imwrite(mask_image_path, mask_image)
            segmented_images.append(mask_image_path)
        return segmented_images

    def overlay_mask(self, image_path, mask_path):
        # Load the image and mask
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image at path '{image_path}' not found.")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask at path '{mask_path}' not found.")

        # Create an RGB version of the mask
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        # Create a color mask
        color_mask = np.zeros_like(mask_rgb)
        color_mask[mask > 0] = [255, 0, 0]  # Red color

        # Overlay the color mask on the image
        overlay = cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)

        # Display the overlay
        plt.imshow(overlay)
        plt.axis('off')
        plt.show()

    def get_bounding_boxes(self, mask_path):
        # Load the mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask at path '{mask_path}' not found.")

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get the bounding boxes
        bounding_boxes = [cv2.boundingRect(contour) for contour in contours]

        return bounding_boxes

    def draw_bounding_boxes(self, image_path, bounding_boxes):
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image at path '{image_path}' not found.")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Draw each bounding box
        for x, y, w, h in bounding_boxes:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green color

        # Display the image
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    def filter_bounding_boxes(self, bounding_boxes, min_area=500, min_width=20, min_height=20,
                              aspect_ratio_range=(0.5, 2.0)):
        filtered_boxes = []
        for x, y, w, h in bounding_boxes:
            area = w * h
            aspect_ratio = w / h
            if area >= min_area and w >= min_width and h >= min_height and aspect_ratio_range[0] <= aspect_ratio <= \
                    aspect_ratio_range[1]:
                filtered_boxes.append((x, y, w, h))
        return filtered_boxes


# Example usage
if __name__ == "__main__":
    sam_checkpoint = "SegmentAnythingModel/sam_vit_h_4b8939.pth"  # Update this path with the actual location of the correct checkpoint
    image_path = "C:/Users/samue/OneDrive/Dokumente/TU-Wien/Robot_Vision_Project/Robot_Vision_YOLO_Object_Detection/Images/Table_with_objects.jpg"  # Update this path with the actual image location

    segmenter = SAMSegmenter(sam_checkpoint_path=sam_checkpoint, model_type="default", grid_size=32)
    segmented_images = segmenter.segment_image(image_path)
    print("Segmented images saved:", segmented_images)

    for mask_path in segmented_images:
        bounding_boxes = segmenter.get_bounding_boxes(mask_path)
        filtered_boxes = segmenter.filter_bounding_boxes(bounding_boxes)
        segmenter.draw_bounding_boxes(image_path, filtered_boxes)  # switch to filtered
        segmenter.overlay_mask(image_path, mask_path)
