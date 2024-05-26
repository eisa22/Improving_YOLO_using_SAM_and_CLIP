import YOLO
import Helper_Functions
import json
import cv2
import Segment_Anything
import OpenAI_API

# Control Panel
# Choose camera idx: 0 for laptop webcam; idx:1 for external camera
camera_idx = 0
webcam = False
conf_threshold = 0.1
conf_threshold_crawler = 0.5
image_path = 'Images/Table_with_objects.jpg' # Images/Sliding_Window_Results/subimage22.jpg
sam_checkpoint = 'SegmentAnythingModel/sam_vit_h_4b8939.pth' # Segment anything checkpoint

debug_mode = True

if __name__ == "__main__":
    # Call YOLO Detector
    detector = YOLO.ObjectDetection(camera_idx, webcam, image_path, debug_mode, conf_threshold)
    detector()

    # Get YOLO Results
    bounding_boxes, confidences, class_ids = detector.get_Yolo_Results()
    # Load the image frame
    frame = cv2.imread(detector.image_path)

    # Initialize Webcrawler with your API key
    openAI_key = 'sk-proj-JnZWLeUpKJvjucLRkGE5T3BlbkFJOz06KwkQ45Mn6CXzVBDq' # OpenAI API key
    webcrawler = OpenAI_API.Webcrawler(openAI_key)

    # Filter and send to API for relabeling
    labels_with_boxes = webcrawler.filter_and_send_to_api(frame, bounding_boxes, confidences, conf_threshold_crawler)

    # Draw bounding boxes with new labels
    webcrawler.draw_bboxes_with_labels(frame, labels_with_boxes)

    # Draw Bounding Boxes
    helper = Helper_Functions.Helper_Functions(image_path)
    helper.draw_boxes(image_path, bounding_boxes, debug_mode)

    # Create Binary Mask
    binary_mask = helper.create_binary_mask(bounding_boxes, debug_mode)











