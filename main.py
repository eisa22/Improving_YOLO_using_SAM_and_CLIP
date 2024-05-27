import YOLO
import Helper_Functions
import json
import cv2
import Segment_Anything
import OpenAI_API
import OpenAI_Key
import matplotlib.pyplot as plt

# Control Panel
# Choose camera idx: 0 for laptop webcam; idx:1 for external camera
camera_idx = 0
webcam = False
conf_threshold = 0.05
conf_threshold_crawler = 0.5
image_path = 'Images/Table_with_objects.jpg' # Images/Sliding_Window_Results/subimage22.jpg
sam_checkpoint = 'SegmentAnythingModel/sam_vit_h_4b8939.pth' # Segment anything checkpoint
openAI_key = OpenAI_Key.openAI_key

debug_mode = True
enable_crawler = True

if __name__ == "__main__":
    # Call YOLO Detector
    detector = YOLO.ObjectDetection(camera_idx, webcam, image_path, debug_mode, conf_threshold)
    detector()

    # Initialize Helper Functions
    helper = Helper_Functions.Helper_Functions(image_path)

    # Get YOLO Results
    bounding_boxes, confidences, class_ids = detector.get_Yolo_Results()
    yolo_labels_with_boxes = helper.prepare_YOLO_Data(bounding_boxes, confidences, class_ids, conf_threshold_crawler)


    if enable_crawler:

        # Load the image frame
        frame = cv2.imread(detector.image_path)

        # Initialize Webcrawler with your API key
        openAI_key = openAI_key # OpenAI API key
        webcrawler = OpenAI_API.Webcrawler(openAI_key)

        # Filter and send to API for relabeling
        labels_with_boxes = webcrawler.filter_and_send_to_api(frame, bounding_boxes, confidences, conf_threshold_crawler)
        print("Formatted labels with boxes: ", yolo_labels_with_boxes)
        print("New labels with boxes: ", labels_with_boxes)

        # Draw bounding boxes with new labels
        helper.draw_bboxes_with_labels(frame, labels_with_boxes, yolo_labels_with_boxes)

        # Merge lists
        helper.merge_lists_and_write_to_json(labels_with_boxes, yolo_labels_with_boxes, image_id=1)



    # Draw Bounding Boxes
    helper.draw_boxes(image_path, bounding_boxes, debug_mode)

    # Create Binary Mask
    binary_mask = helper.create_binary_mask(bounding_boxes, debug_mode)












