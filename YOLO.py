import torch
import numpy as np
import cv2
from ultralytics import YOLO
import json
import requests

class ObjectDetection:
    def __init__(self, capture_index, webcam=True, debug_mode=False, conf_threshold=0.1):
        self.capture_index = capture_index
        self.webcam = webcam
        self.debug_mode = debug_mode
        self.conf_threshold = conf_threshold
        self.results= []

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
            print("Filtered results: ", filtered_results)
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

    def get_YOLO_Results(self, image_url, image_id):
        if self.webcam:
            cap = cv2.VideoCapture(self.camera_idx)
            assert cap.isOpened(), "Cannot open camera"

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break

                results = self.predict(frame)
                plotted_image, xyxys, confidences, class_ids = self.plot_bboxes(results, frame)
                cv2.imshow('frame', plotted_image)

                if cv2.waitKey(1) == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
        else:
            assert image_url is not None, "Image URL must be provided when webcam is False"

            response = requests.get(image_url)
            assert response.status_code == 200, "Failed to fetch image from URL"
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            results = self.predict(frame)
            plotted_image, xyxys, confidences, class_ids = self.plot_bboxes(results, frame)

            data = {
                "image_id": image_id,
                "bbox": xyxys,
                "category_id": class_ids,
                "confidences": confidences,
                "image_url": image_url
            }

            # Convert data to COCO format
            coco_data = []
            for i in range(len(xyxys)):
                coco_data.append({
                    'image_id': image_id,
                    'bbox': xyxys[i][0],  # Assuming xyxys is a nested list
                    'category_id': int(class_ids[i][0]),  # Assuming class_ids is a nested list
                    'id': i,  # Unique ID for each annotation
                    'iscrowd': 0  # Assuming there are no crowd annotations in the YOLO results
                })

            self.results.append(coco_data)
            self.save_results()

            if self.debug_mode:
                cv2.imshow('YOLO_Results', plotted_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return coco_data, confidences, image_url, frame


    def save_results(self):
        with open('Results_YOLO_Detector.json', 'w') as f:
            json.dump(self.results, f, indent=4)

