
from PIL import Image, ImageDraw
import requests
from io import BytesIO
import json
import os

class Validation:
    import os


    def coco_to_yolo(self, bbox):
        x, y, width, height = bbox
        center_x = x + width / 2
        center_y = y + height / 2
        return [center_x, center_y, width, height]

    def yolo_to_coco(self, bbox):
        center_x, center_y, width, height = bbox
        x = center_x - width / 2
        y = center_y - height / 2
        return [x, y, width, height]

    def is_yolo_format(self, bbox):
        # Check if bbox is in YOLO format
        x, y, width, height = bbox
        return (x < width and y < height)  # Assuming coordinates are smaller than dimensions

    def calculate_IoU(self, cocoDataList, coco_gt, images_url, debug_mode):
        iou_scores = []

        for data in cocoDataList:
            image_id = data['image_id']
            bbox = data['bbox']
            category_id = data['category_id']
            print(f"Processing image_id: {image_id}, category_id: {category_id}")

            # Check and convert bbox to COCO format if necessary
            if self.is_yolo_format(bbox):
                bbox = self.yolo_to_coco(bbox)

            # Load all ground truth data (not filtering by category)
            ann_ids = coco_gt.getAnnIds()
            anns = coco_gt.loadAnns(ann_ids)

            highest_iou = 0
            best_match = None

            for ann in anns:
                ann_bbox = ann['bbox']

                # Convert bounding boxes to [x1, y1, x2, y2] format
                box1 = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                box2 = [ann_bbox[0], ann_bbox[1], ann_bbox[0] + ann_bbox[2], ann_bbox[1] + ann_bbox[3]]

                # Calculate IoU
                iou_score = self.iou(box1, box2)
                if iou_score > highest_iou:
                    highest_iou = iou_score
                    best_match = ann

            if best_match and highest_iou > 0.1:  # Check if the IoU score is higher than 0.6
                iou_scores.append({
                    'image_id': image_id,
                    'category_id': category_id,
                    'iou_score': highest_iou,
                    'matched_image_id': best_match['image_id']
                })
                if debug_mode:
                    # Download the image
                    image_url = f"{images_url}"  # Adjust the URL as needed
                    response = requests.get(image_url)
                    if response.status_code == 200:
                        image = Image.open(BytesIO(response.content))
                        draw = ImageDraw.Draw(image)

                        # Draw the predicted bounding box (in red)
                        draw.rectangle([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], outline="red", width=2)

                        # Draw the ground truth bounding box (in green)
                        gt_bbox = best_match['bbox']
                        draw.rectangle([gt_bbox[0], gt_bbox[1], gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]], outline="green", width=2)

                        # Draw the matched bounding box (in purple)
                        matched_bbox = best_match['bbox']
                        draw.rectangle([matched_bbox[0], matched_bbox[1], matched_bbox[0] + matched_bbox[2], matched_bbox[1] + matched_bbox[3]], outline="purple", width=2)

                        # Save the image with bounding boxes
                        debug_image_path = f"{image_id}_debug.jpg"
                        image.save(debug_image_path)
                        print(f"Saved debug image with bounding boxes to {debug_image_path}")
                    else:
                        print(f"Failed to download image from {image_url}")

        # Store IoU scores in a JSON file
        if os.path.exists('Results_IoU.json'):
            with open('Results_IoU.json', 'r+') as f:
                existing_data = json.load(f)
                existing_data.extend(iou_scores)
                f.seek(0)
                json.dump(existing_data, f)
        else:
            with open('Results_IoU.json', 'w') as f:
                json.dump(iou_scores, f)

    def iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        w_intersection = min(x2, x2_2) - max(x1, x1_2)
        h_intersection = min(y2, y2_2) - max(y1, y1_2)
        if w_intersection <= 0 or h_intersection <= 0:  # No overlap
            return 0

        I = w_intersection * h_intersection
        U = (x2 - x1) * (y2 - y1) + (x2_2 - x1_2) * (y2_2 - y1_2) - I  # Union = Total Area - I

        if U == 0:  # Prevent division by zero
            return 0

        iou = I / U
        print(f"IoU: {iou}, Intersection: {I}, Union: {U}, Box1: {box1}, Box2: {box2}")  # Debug print
        return iou

    def calculate_average_iou(self):
        iou_scores = []

        # Check if the Results_IoU.json file exists
        if self.os.path.exists('Results_IoU.json'):
            with open('Results_IoU.json', 'r') as f:
                data = json.load(f)

                # Extract all IoU scores
                for entry in data:
                    iou_scores.append(entry['iou_score'])

            # Calculate the average IoU score
            if iou_scores:
                average_iou = sum(iou_scores) / len(iou_scores)
                print("Average IoU score: ", average_iou)
                return average_iou
            else:
                return 0
        else:
            print("The file 'Results_IoU.json' does not exist.")
            return 0
