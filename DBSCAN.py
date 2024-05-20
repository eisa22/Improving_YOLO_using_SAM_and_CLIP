import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time


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


def segment_image(features, image_shape, eps, min_samples, prev_eps=None, prev_min_samples=None):
    # Check if the eps or min_samples values have changed
    if eps != prev_eps or min_samples != prev_min_samples:
        start_time = time.time()  # Start the timer
        print(f"New parameters detected: eps={eps}, min_samples={min_samples}. Timer started at {start_time} seconds.")

    # Apply DBSCAN with specified eps and min_samples
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(features)
    labels = dbscan.labels_

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

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black color for noise
            col = np.array([0, 0, 0])
        else:
            col = (col[:3] * 255).astype(int)
        mask = labels_2d == k
        output_image[mask] = col

    # Display the original and segmented images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.title("Segmented Image")
    plt.imshow(output_image)
    plt.show()


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

for eps in eps_values:
    for min_samples in min_samples_values:
        print(f"DBSCAN with eps={eps}, min_samples={min_samples}")

        #start_time = time.time()  # Record the start time

        segmented_image, labels = segment_image(features, image.shape, eps, min_samples)
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels) - (1 if -1 in labels else 0)

        #end_time = time.time()  # Record the end time
        #elapsed_time = end_time - start_time  # Calculate the elapsed time

        #print(f"DBSCAN completed in {elapsed_time} seconds")

        #if elapsed_time > 8:
            #print("DBSCAN took too long to run. Trying next parameters.")
            #continue

        if num_clusters > max_clusters:
            max_clusters = num_clusters
            best_segmented_image = segmented_image
            best_labels = labels

        if num_clusters > max_clusters:
            max_clusters = num_clusters
            best_segmented_image = segmented_image
            best_labels = labels

# Visualize the best segmentation result
if best_segmented_image is not None and best_labels is not None:
    visualize_segmentation(image, best_segmented_image, best_labels)
    print(f"Best parameters: eps={eps}, min_samples={min_samples} with {max_clusters} clusters")
else:
    print("No valid clusters found.")

# Save the best segmented image to Images/DBSCAN_Results
cv2.imwrite('Images/DBSCAN_Results/DBS_clustered_image.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
