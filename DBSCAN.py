import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time
import threading
import os


class DBSCANWithTimeout:
    def __init__(self, eps, min_samples, features):
        self.eps = eps
        self.min_samples = min_samples
        self.features = features  # Features are already normalized in extract_features
        self.labels = None
        self.exception = None

    def run_dbscan(self):
        try:
            dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1).fit(self.features)
            self.labels = dbscan.labels_
        except Exception as e:
            self.exception = e

class DBSCANProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = self.preprocess_image(self.image_path)
        self.features = self.extract_features(self.image)

    def preprocess_image(self, image_path):
        # Load the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Resize the image to a standard size
        image = cv2.resize(image, (256, 256))

        return image

    def extract_features(self, image):
        # Reshape the image to a 2D array of pixels
        pixels = image.reshape(-1, 3)  # Only use color features

        # Optionally, add spatial coordinates
        coords = np.indices((image.shape[0], image.shape[1])).reshape(2, -1).T
        features = np.concatenate([pixels, coords], axis=1)

        # Normalize the features
        features = features / [255.0, 255.0, 255.0, image.shape[0], image.shape[1]]

        # Dimensionality reduction using PCA
        pca = PCA(n_components=5)
        features = pca.fit_transform(features)

        return features

    def segment_image(self, features, image_shape, eps, min_samples, prev_eps=None, prev_min_samples=None, timeout=8):
        start_time = None

        # Check if the eps or min_samples values have changed
        if eps != prev_eps or min_samples != prev_min_samples:
            start_time = time.time()  # Start the timer
            print(f"New parameters detected: eps={eps}, min_samples={min_samples}. Timer started at {start_time} seconds.")

        # Create a DBSCANWithTimeout instance
        dbscan_instance = DBSCANWithTimeout(eps, min_samples, features)
        dbscan_thread = threading.Thread(target=dbscan_instance.run_dbscan)

        # Start the DBSCAN thread
        dbscan_thread.start()

        # Monitor the execution time
        dbscan_thread.join(timeout)  # Wait for up to 8 seconds

        if dbscan_thread.is_alive():
            print("DBSCAN aborted due to timeout")
            dbscan_thread.join()  # Ensure the thread is properly cleaned up
            raise TimeoutError("DBSCAN exceeded the time limit of 8 seconds")

        # Check for exceptions
        if dbscan_instance.exception:
            raise dbscan_instance.exception

        # Get the labels
        labels = dbscan_instance.labels

        # Reshape the labels to the original image shape
        segmented_image = labels.reshape(image_shape[:2])

        return segmented_image, labels

    def visualize_segmentation(self, image, labels):
        # Reshape labels to 2D for boolean indexing
        labels_2d = labels.reshape(image.shape[:2])

        # Create a color map for visualization
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        colors = plt.cm.Spectral(np.linspace(0, 1, num_clusters))

        # Create an output image with the same shape as the input
        output_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        bounding_boxes = []  # List to store bounding box coordinates

        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black color for noise
                col = np.array([0, 0, 0])
            else:
                col = (col[:3] * 255).astype(int)
            mask = labels_2d == k
            output_image[mask] = col

            # Draw bounding box around each cluster
            if k != -1:  # Skip noise
                y, x = np.where(labels_2d == k)
                if len(x) > 0 and len(y) > 0:
                    x_min, x_max = x.min(), x.max()
                    y_min, y_max = y.min(), y.max()
                    cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
                    bounding_boxes.append((x_min, y_min, x_max, y_max))

        return output_image, bounding_boxes

def run_dbscan_segmentation(image_path):
    """
        This function runs the DBSCAN segmentation on an image and returns the best segmented image, labels, and bounding boxes.

        Parameters:
        image_path (str): The path to the image file.

        Returns:
        output_image (ndarray): The best segmented image. If no valid clusters are found, it returns None.
        best_labels (ndarray): The labels of the best segmented image. If no valid clusters are found, it returns None.
        bounding_boxes (list): The list of bounding boxes for the best segmented image. If no valid clusters are found, it returns None.
    """
    processor = DBSCANProcessor(image_path)

    # Tune DBSCAN parameters
    eps_values = [0.1, 0.2, 0.3]
    min_samples_values = [50, 100, 200]

    best_segmented_image = None
    best_labels = None
    max_clusters = 0
    best_eps = None
    best_min_samples = None

    for eps in eps_values:
        for min_samples in min_samples_values:
            print(f"DBSCAN with eps={eps}, min_samples={min_samples}")

            try:
                segmented_image, labels = processor.segment_image(processor.features, processor.image.shape, eps, min_samples)
            except TimeoutError as e:
                print(e)
                continue

            unique_labels = np.unique(labels)
            num_clusters = len(unique_labels) - (1 if -1 in labels else 0)

            if num_clusters > max_clusters:
                max_clusters = num_clusters
                best_segmented_image = segmented_image
                best_labels = labels
                best_eps = eps
                best_min_samples = min_samples

    # Visualize the best segmentation result
    if best_segmented_image is not None and best_labels is not None:
        output_image, bounding_boxes = processor.visualize_segmentation(processor.image, best_labels)
        print(f"Best parameters: eps={best_eps}, min_samples={best_min_samples} with {max_clusters} clusters")
        # Safe image to Images/DBSCAN_Results.jpg
        cv2.imwrite("Images/DBSCAN_Results.jpg", cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
        return output_image, best_labels, bounding_boxes
    else:
        print("No valid clusters found.")
        return None, None, None


if __name__ == "__main__":
    image_path = "Images/Table_with_objects.jpg"
    output_image, best_labels, bounding_boxes = run_dbscan_segmentation(image_path)





    if output_image is not None and best_labels is not None:
        cv2.imshow("Segmented Image", cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("No valid clusters found.")