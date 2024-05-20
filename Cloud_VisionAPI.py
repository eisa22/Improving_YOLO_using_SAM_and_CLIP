import os
import io
import cv2
from google.cloud import vision
import numpy as np

# Set the path to the Google Cloud service account key file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'Cloud_Vision_Key/visionapi-423909-843341ebf44d.json'

# Initialize the Google Cloud Vision client
client = vision.ImageAnnotatorClient()

def detect_labels(image):
    """Detects labels in the image."""
    content = cv2.imencode('.jpg', image)[1].tobytes()
    image = vision.Image(content=content)
    response = client.label_detection(image=image)
    labels = response.label_annotations

    if response.error.message:
        raise Exception(f'{response.error.message}')

    return labels

def annotate_image(image_path, bounding_boxes):
    """Annotates the image with labels for each bounding box."""
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image {image_path}")

    # Check if the image size is 255x255, if not, resize it
    if image.shape[:2] != (256, 256):
        image = cv2.resize(image, (256, 256))

    labeled_image = image.copy()

    for (x_min, y_min, x_max, y_max) in bounding_boxes:
        # Crop the region of interest (ROI)
        roi = image[y_min:y_max, x_min:x_max]

        # Detect labels for the ROI
        labels = detect_labels(roi)
        if labels:
            # Get the description of the first label
            label = labels[0].description
            print(f"Detected label for bounding box ({x_min}, {y_min}, {x_max}, {y_max}): {label}")

            # Draw the bounding box and label on the image
            cv2.rectangle(labeled_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(labeled_image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save and display the annotated image
    output_path = os.path.splitext(image_path)[0] + '_labeled.jpg'
    cv2.imwrite(output_path, labeled_image)
    print(f"Annotated image saved as {output_path}")

    # Display the annotated image
    cv2.imshow("Annotated Image", labeled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    image_path = 'Images/Table_with_objects.jpg'
    bounding_boxes = [(50, 50, 150, 150), (200, 200, 300, 300)]  # Example bounding boxes
    annotate_image(image_path, bounding_boxes)
