import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
import json

class ObjectDetection:

    def __init__(self, capture_index, webcam=True, image_path=None):
        self.capture_index = capture_index
        self.webcam = webcam
        self.image_path = image_path

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Usind device: ", self.device)

        self.model = self.load_model()

    def load_model(self):
        model = YOLO('yolov8n')
        model.fuse()
        return model

    def predict(self, frame):
        results = self.model(frame)
        return results

    def plot_bboxes(self, results, frame):
        xyxys = []
        confidences = []
        class_ids = []

        # Extract detections for class
        for result in results:
            boxes = result.boxes.cpu().numpy()

            # Extract xy coordinates, confidences and class ids
            xyxys.append(boxes.xyxy)
            confidences.append(boxes.conf.tolist())  # Convert ndarray to list
            class_ids.append(boxes.cls.tolist())  # Convert ndarray to list

        # Save confidences and class_ids to a JSON file
        data = {
            "confidences": confidences,
            "class_ids": class_ids
        }

        with open('results.json', 'w') as f:
            json.dump(data, f)

        return results[0].plot(), xyxys, confidences, class_ids

    def __call__(self):

        if self.webcam:
            cap = cv2.VideoCapture(self.capture_index)
            assert cap.isOpened(), "Cannot open camera"

            while True:
                # Capture frame-by-frame
                ret, frame = cap.read()
                # if frame is read correctly ret is True
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break

                # Perform object detection
                results = self.predict(frame)

                # Draw bounding boxes on the frame and get the plotted image, bounding boxes, confidences, and class ids
                plotted_image, xyxys, confidences, class_ids = self.plot_bboxes(results, frame)

                # Display the resulting frame
                cv2.imshow('frame', plotted_image)

                # Break the loop if 'q' key is pressed
                if cv2.waitKey(1) == ord('q'):
                    break

            # When everything done, release the capture
            cap.release()
            cv2.destroyAllWindows()
        else:
            assert self.image_path is not None, "Image path must be provided when webcam is False"
            frame = cv2.imread(self.image_path)

            # Perform object detection
            results = self.predict(frame)

            # Draw bounding boxes on the frame and get the plotted image, bounding boxes, confidences, and class ids
            plotted_image, xyxys, confidences, class_ids = self.plot_bboxes(results, frame)

            # Display the resulting frame
            cv2.imshow('frame', plotted_image)
            cv2.waitKey(0)  # Wait for any key to be pressed to close the window
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # Control Panel
    # Choose camera idx: 0 for laptop webcam; idx:1 for external camera
    camera_idx = 1

    # Choose if webcam should be active (True) or image should be used (False)
    webcam = True

    # Image path
    image_path = 'Images/Table_with_objects.jpg'

    detector = ObjectDetection(camera_idx, webcam, image_path)
    detector()

