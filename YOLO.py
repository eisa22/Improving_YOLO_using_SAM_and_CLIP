import torch
import numpy as np
import cv2
from ultralytics import YOLO
import json

class ObjectDetection:
    def __init__(self, capture_index, webcam=True, image_path=None, debug_mode=False, conf_threshold=0.1):
        self.capture_index = capture_index
        self.webcam = webcam
        self.image_path = image_path
        self.debug_mode = debug_mode
        self.conf_threshold = conf_threshold

        # ToDo: Make everything run on cuda
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using device: ", self.device)

        self.model = self.load_model()

    def load_model(self):
        model = YOLO('yolov8n')
        model.fuse()
        return model

    def predict(self, frame):
        results = self.model(frame)
        # Filter results by confidence threshold
        filtered_results = []
        for result in results:
            filtered_boxes = []
            for box in result.boxes:
                if box.conf >= self.conf_threshold:
                    filtered_boxes.append(box)
            result.boxes = filtered_boxes
            filtered_results.append(result)
        return filtered_results

    def plot_bboxes(self, results, frame):
        xyxys = []
        confidences = []
        class_ids = []

        # Extract detections for class
        for result in results:
            boxes = result.boxes
            for box in boxes:
                xyxys.append(box.xyxy.cpu().numpy().tolist())
                confidences.append(box.conf.cpu().numpy().tolist())
                class_ids.append(box.cls.cpu().numpy().tolist())

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

            # Save confidences and class_ids to a JSON file
            data = {
                "confidences": confidences,
                "class_ids": class_ids,
                "bounding_boxes": xyxys
            }

            with open('results.json', 'w') as f:
                json.dump(data, f)

            if self.debug_mode:
                # Display the resulting frame
                cv2.imshow('thisframe', plotted_image)
                cv2.waitKey(0)  # Wait for any key to be pressed to close the window
                cv2.destroyAllWindows()

    def get_Yolo_Results(self):
        """
        Returns the results of the object detection process.

        Returns
        -------
        list, list, list
            the bounding box coordinates, the confidences, and the class ids
        """
        # Read bounding box coordinates
        with open('results.json', 'r') as f:
            data = json.load(f)

        confidences = data['confidences']
        class_ids = data['class_ids']
        bounding_boxes = data['bounding_boxes']
        return bounding_boxes, confidences, class_ids

