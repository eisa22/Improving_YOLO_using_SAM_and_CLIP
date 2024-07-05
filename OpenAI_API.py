from openai import OpenAI
import openai
import cv2
import base64
import os
import json
import numpy as np
import json
import cv2
import numpy as np
import requests
from io import BytesIO

class Webcrawler:
    def __init__(self, api_key):
        self.api_key = api_key   # Initialize OpenAI client without arguments

    def filter_and_send_to_api(self, json_path, confidence_threshold, output_path):
        """
        Filters bounding boxes by confidence threshold from the provided JSON and sends the corresponding images to the API for relabeling.
        Saves the results incrementally in output_path with GPT_predictions and ground_truths.

        Parameters
        ----------
        json_path : str
            Path to the JSON file containing results.
        confidence_threshold : float
            The confidence threshold.
        output_path : str
            Path to the output JSON file.

        Returns
        -------
        None
        """
        # Read the JSON file
        with open(json_path, 'r') as file:
            data = json.load(file)

        # Open the output JSON file in append mode
        with open(output_path, 'w') as outfile:
            outfile.write('[')  # Start the JSON array

            first_entry = True

            for image_entry in data:
                # Load the image from the URL
                image_url = image_entry['url']
                response = requests.get(image_url)
                if response.status_code == 200:
                    image_data = BytesIO(response.content)
                    frame = cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), cv2.IMREAD_COLOR)
                else:
                    print(f"Failed to load image from URL: {image_url}")
                    continue  # Skip if the image can't be loaded

                new_predictions = []

                # Process predictions
                predictions = image_entry.get('predictions', [])
                for prediction in predictions:
                    bbox = prediction['bbox']
                    category = prediction['category']

                    x1, y1, x2, y2 = map(int, bbox)
                    sub_image = frame[y1:y2, x1:x2]
                    _, img_encoded = cv2.imencode('.jpg', sub_image)

                    try:
                        new_label = self.send_image_to_api(img_encoded)
                        parts = new_label.split(',')
                        label_text = parts[0].strip()
                        confidence = float(parts[1].strip().replace('%', '')) / 100.0  # Convert percentage to a decimal

                        if confidence >= confidence_threshold:
                            new_predictions.append({
                                "bbox": bbox,
                                "category": label_text,
                                "confidence": confidence
                            })
                    except requests.exceptions.RequestException as e:
                        # Log the error and continue
                        print(f"Error processing image: {e}")
                        continue
                    except Exception as e:
                        # Handle other exceptions, such as content policy violations
                        if isinstance(e, requests.HTTPError) and e.response.status_code == 400:
                            error_info = e.response.json()
                            if error_info.get('error', {}).get('code') == 'content_policy_violation':
                                print(f"Content policy violation for image: {image_url}")
                                continue
                        else:
                            print(f"Unexpected error: {e}")
                            continue

                new_entry = {
                    "image_id": image_entry['image_id'],
                    "url": image_entry['url'],
                    "GPT_predictions": new_predictions,
                    "ground_truths": image_entry.get('ground_truths', [])
                }

                if not first_entry:
                    outfile.write(',\n')
                json.dump(new_entry, outfile, indent=4)
                first_entry = False

            outfile.write(']')  # End the JSON array
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
                {"role": "system",
                 "content": "You are an assistant that identifies objects in images. Use only labels mentioned in coco dataset."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": "What object is in this image? Answer in one word and give confidence. Format: Label, Confidence in %"},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            max_tokens=100,  # Adjust max tokens to a reasonable limit
        )
        print("Response", response)

        return response.choices[0].message.content

