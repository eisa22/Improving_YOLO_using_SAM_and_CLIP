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

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Resize the image to a standard size
    image = cv2.resize(image, (256, 256))

    return image

def extract_features(image):
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

def segment_image(features, image_shape, eps, min_samples, prev_eps=None, prev_min_samples=None, timeout=8):
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

def visualize_segmentation(image, segmented_image, labels):
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

    # Display the original and segmented images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.title("Segmented Image")
    plt.imshow(output_image)
    plt.show()

    return output_image, bounding_boxes

# Create output directory if it doesn't exist
output_dir = 'Images/DBSCAN_Results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process a single image for tuning
image_path = 'Images/Table_with_objects.jpg'
image = preprocess_image(image_path)
features = extract_features(image)

# Visualize features using PCA components
plt.figure(figsize=(8, 6))
plt.scatter(features[:, 0], features[:, 1], s=1)
plt.title("Feature Space Visualization")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

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
            segmented_image, labels = segment_image(features, image.shape, eps, min_samples)
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
    output_image, bounding_boxes = visualize_segmentation(image, best_segmented_image, best_labels)
    print(f"Best parameters: eps={best_eps}, min_samples={best_min_samples} with {max_clusters} clusters")

    # Save each bounding box as a separate image
    for i, (x_min, y_min, x_max, y_max) in enumerate(bounding_boxes):
        roi = image[y_min:y_max+1, x_min:x_max+1]
        roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
        roi_filename = os.path.join(output_dir, f'cluster_{i}.jpg')
        cv2.imwrite(roi_filename, roi_bgr)
        print(f"Saved bounding box {i} as {roi_filename}")
else:
    print("No valid clusters found.")

# Save the best segmented image as a JPG file
output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
cv2.imwrite(os.path.join(output_dir, 'DBS_clustered_image.jpg'), output_image_bgr)
print("Cluster map saved successfully as a JPG image.")
