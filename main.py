import YOLO
import Helper_Functions
import json
import cv2

# Control Panel
# Choose camera idx: 0 for laptop webcam; idx:1 for external camera
camera_idx = 0
webcam = False
image_path = 'Images/Table_with_objects.jpg' # Images/Sliding_Window_Results/subimage22.jpg
debug_mode = True

if __name__ == "__main__":
    # Call YOLO Detector
    detector = YOLO.ObjectDetection(camera_idx, webcam, image_path, debug_mode)
    detector()

    # Get YOLO Results
    bounding_boxes, confidences, class_ids = detector.get_Yolo_Results()

    # Draw Bounding Boxes
    helper = Helper_Functions.Helper_Functions(image_path)
    helper.draw_boxes(image_path, bounding_boxes, debug_mode)

    # Create Binary Mask
    binary_mask = helper.create_binary_mask(bounding_boxes, debug_mode)

    # Draw Bounding Boxes on Black Areas
    helper.draw_boxes_on_black_areas(binary_mask, debug_mode)








