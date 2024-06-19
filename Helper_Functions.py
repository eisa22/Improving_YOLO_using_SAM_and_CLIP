import cv2
import numpy as np
import json



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

    def __init__(self):
        pass

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





    def prepare_YOLO_Data(self, bounding_boxes, confidences, class_ids, threshold, enable_evaluation_mode):
        yolo_labels_with_boxes = list(zip(bounding_boxes, class_ids, confidences))
        formatted_yolo_labels_string = []
        yolo_labels_class_id = []
        if not enable_evaluation_mode:
            # Apply labels from coco classes list and format bounding boxes correctly
            formatted_yolo_labels_string = [
                (bbox[0], coco_classes[int(label[0])], confidences[0])
                for bbox, label, confidences in yolo_labels_with_boxes
                if confidences[0] >= threshold
            ]

        if enable_evaluation_mode:
            # Format yolo_labels_with_boxes in the same way
            yolo_labels_class_id = [
                (bbox[0], label[0], confidences[0])
                for bbox, label, confidences in yolo_labels_with_boxes
            ]


        return formatted_yolo_labels_string, yolo_labels_class_id

    @staticmethod
    def draw_bboxes_with_labels(frame, labels_with_boxes, yolo_labels_with_boxes):
        """
        Draws bounding boxes and labels on the image.

        Parameters
        ----------
        frame : ndarray
            The original image frame.
        labels_with_boxes : list
            List of tuples containing bounding box coordinates and labels from the webcrawler.
        yolo_labels_with_boxes : list
            List of tuples containing bounding box coordinates and labels from YOLO.
        """
        for (bbox, label, confidences) in labels_with_boxes + yolo_labels_with_boxes:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidences:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('Image with Labels', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def merge_lists_and_write_to_json(self, list1, list2, enable_evaluation_mode=False, image_id=None):
        merged_list = list1 + list2

        # Create a dictionary for each item in the merged list
        formatted_list = [
            {
                'image_id': image_id,
                'category_id': item[1],
                'bbox': item[0],
                'score': item[2]
            }
            for item in merged_list
        ]
        if enable_evaluation_mode:
            # Read existing data
            try:
                with open('results_evaluation.json', 'r') as f:
                    existing_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing_data = []

            # Append new data to existing data
            combined_data = existing_data + formatted_list

            # Write combined data to file
            with open('results_evaluation.json', 'w') as f:
                json.dump(combined_data, f)
        else:
            # Read existing data
            try:
                with open('results_gui.json', 'r') as f:
                    existing_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing_data = []

            # Append new data to existing data
            combined_data = existing_data + formatted_list

            # Write combined data to file
            with open('results_gui.json', 'w') as f:
                json.dump(combined_data, f)

        return formatted_list  # Return the formatted list


    def find_crawler_coco_ids(self, labels_with_boxes):
        if not labels_with_boxes:
            print("No labels found.")
            return []

        labels_with_ids = []
        for bbox, label, confidence in labels_with_boxes:
            try:
                label_id = coco_classes.index(label.lower())
            except ValueError:
                continue  # Skip if label is not found in coco_classes
            labels_with_ids.append((bbox, label_id, confidence))
        return labels_with_ids


    @staticmethod
    def merge_jsons():
        # Load the data from results.json
        with open('results.json') as f:
            results_data = json.load(f)

        # Load the data from results_Chat_GPT.json
        with open('results_ChatGPT.json') as f:
            chat_gpt_data = json.load(f)

        # Create a dictionary for faster lookup of GPT predictions
        gpt_predictions_dict = {image['image_id']: image['GPT_predictions'] for image in chat_gpt_data}

        # Process each image in the results data
        for image in results_data:
            image_id = image['image_id']

            # If this image is in the GPT predictions, overwrite the categories
            if image_id in gpt_predictions_dict:
                gpt_predictions = gpt_predictions_dict[image_id]

                # Create a dictionary for faster lookup of GPT categories
                gpt_categories_dict = {tuple(pred['bbox']): pred['category'] for pred in gpt_predictions}

                # Overwrite the categories in the results data
                for pred in image['predictions']:
                    bbox_tuple = tuple(pred['bbox'])
                    if bbox_tuple in gpt_categories_dict:
                        pred['category'] = gpt_categories_dict[bbox_tuple]

        # Write the merged data to final_result.json
        with open('final_result.json', 'w') as f:
            json.dump(results_data, f, indent=4)











