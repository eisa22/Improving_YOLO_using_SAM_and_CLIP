from DBSCAN import run_dbscan_segmentation
from Cloud_VisionAPI import annotate_image

import cv2

image_path = "Images/Table_with_objects.jpg"


# Run DBScan code

if __name__ == "__main__":
    # Run DBSCAN
    output_image, best_labels, bounding_boxes = run_dbscan_segmentation(image_path)

    # Label images
    annotate_image(image_path, bounding_boxes)
