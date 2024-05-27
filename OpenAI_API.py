from openai import OpenAI
import openai
import cv2
import base64
import os

class Webcrawler:
    def __init__(self, api_key):
        self.api_key = api_key   # Initialize OpenAI client without arguments

    def filter_and_send_to_api(self, frame, bounding_boxes, confidences, confidence_threshold):
        """
        Filters bounding boxes by confidence threshold and sends the corresponding images to the API for relabeling.

        Parameters
        ----------
        frame : ndarray
            The original image frame.
        bounding_boxes : list
            List of bounding boxes.
        confidences : list
            List of confidence scores.
        confidence_threshold : float
            The confidence threshold.

        Returns
        -------
        list
            List of tuples containing bounding box coordinates and new labels.
        """
        new_labels_with_boxes = []

        for bbox_list, conf_list in zip(bounding_boxes, confidences):
            # Extract bounding boxes and confidence values from lists
            for bbox, conf in zip(bbox_list, conf_list):
                if conf < confidence_threshold:
                    x1, y1, x2, y2 = map(int, bbox)
                    sub_image = frame[y1:y2, x1:x2]
                    _, img_encoded = cv2.imencode('.jpg', sub_image)
                    new_label = self.send_image_to_api(img_encoded)
                    new_labels_with_boxes.append((bbox, new_label))

        return new_labels_with_boxes

    def send_image_to_api(self, img_encoded):
        """
        Sends an image to the OpenAI API and returns the response.

        Parameters
        ----------
        img_encoded : bytes
            The encoded image bytes.

        Returns
        -------
        str
            The label returned by the API.
        """
        # Convert image to base64
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{img_base64}"

        os.environ["OPENAI_API_KEY"] = self.api_key

        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What’s in this image? In one word"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        print("Response", response)

        return response.choices[0].message.content

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
        for (bbox, label) in labels_with_boxes + yolo_labels_with_boxes:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow('Image with Labels', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()