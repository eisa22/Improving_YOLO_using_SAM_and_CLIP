import pandas as pd

# Load class description file
class_description_path = 'class-descriptions-boxable.csv'
class_descriptions = pd.read_csv(class_description_path, header=None, names=['LabelName', 'LabelNameReadable'])

# Create a dictionary for mapping
class_mapping = dict(zip(class_descriptions['LabelName'], class_descriptions['LabelNameReadable']))

# Example DataFrame for Open Images labels
data = {
    'ImageID': ['img1', 'img2'],
    'XMin': [0.1, 0.2],
    'XMax': [0.5, 0.6],
    'YMin': [0.1, 0.2],
    'YMax': [0.5, 0.6],
    'LabelName': ['/m/07j7r', '/m/0123d']  # Example class codes
}

df = pd.DataFrame(data)

# Function to convert bounding box coordinates to YOLO format
def convert_to_yolo_format(row, img_width, img_height):
    x_center = (row['XMin'] + row['XMax']) / 2
    y_center = (row['YMin'] + row['YMax']) / 2
    width = row['XMax'] - row['XMin']
    height = row['YMax'] - row['YMin']
    return [x_center, y_center, width, height]

# Conversion
yolo_labels = []
for index, row in df.iterrows():
    yolo_class_name = class_mapping[row['LabelName']]
    bbox = convert_to_yolo_format(row, img_width=1, img_height=1)  # Assuming normalized coordinates
    yolo_labels.append([yolo_class_name] + bbox)

# Output YOLO labels with class names
for label in yolo_labels:
    print(" ".join(map(str, label)))
