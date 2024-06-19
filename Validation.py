import json
import numpy as np


class Validation:
    @staticmethod
    def calculate_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxB[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    def calculate_image_iou(self, predictions, ground_truths):
        total_iou = 0
        count = 0
        for pred in predictions:
            max_iou = 0
            for gt in ground_truths:
                iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                if iou > max_iou:
                    max_iou = iou
            total_iou += max_iou
            count += 1
        return total_iou / count if count > 0 else 0

    def process_results(self, results):
        results_iou = {}
        overall_iou_sum = 0
        overall_count = 0

        for result in results:
            image_id = result['image_id']
            predictions = result['predictions']
            ground_truths = result['ground_truths']

            image_iou = self.calculate_image_iou(predictions, ground_truths)
            results_iou[image_id] = image_iou

            overall_iou_sum += image_iou
            overall_count += 1

        overall_iou = overall_iou_sum / overall_count if overall_count > 0 else 0

        with open('results_IoU.txt', 'w') as file:
            for image_id, iou in results_iou.items():
                file.write(f'Image ID: {image_id}, IoU: {iou}\n')
            file.write(f'Overall IoU: {overall_iou}\n')

        return results_iou, overall_iou

    @staticmethod
    def calculate_YOLO_accuracy():
        # Load the data from results.json
        with open('results.json') as f:
            data = json.load(f)

        total_matches = 0
        total_gt_boxes = 0

        # Open Accuracy_YOLO.txt for writing
        with open('Accuracy_YOLO.txt', 'w') as f:
            # Process each image in the data
            for image in data:
                # Extract the categories from the predictions and ground truths
                pred_categories = {pred['category'] for pred in image['predictions']}
                gt_categories = {gt['category'] for gt in image['ground_truths']}

                # Compare the categories and count the matches
                matches = len(pred_categories & gt_categories)
                total_matches += matches

                # Update the total number of ground truth boxes
                total_gt_boxes += len(gt_categories)

                # Calculate the accuracy for the image
                accuracy = matches / len(gt_categories) if gt_categories else 0

                # Write the result for the image to Accuracy_YOLO.txt
                f.write(f"Image ID: {image['image_id']}, Accuracy: {accuracy}\n")

            # Calculate the overall accuracy (mAP)
            overall_accuracy = total_matches / total_gt_boxes if total_gt_boxes else 0

            # Write the overall accuracy to Accuracy_YOLO.txt
            f.write(f"Overall mAP: {overall_accuracy}\n")




    @staticmethod
    def calculate_YOLO_API_accuracy():
        # Load the data from results.json
        with open('results_ChatGPT.json') as f:
            data = json.load(f)

        total_matches = 0
        total_gt_boxes = 0

        # Open Accuracy_YOLO_API.txt for writing
        with open('Accuracy_YOLO_API.txt', 'w') as f:
            # Process each image in the data
            for image in data:
                # Extract the categories from the predictions and ground truths
                pred_categories = {pred['category'] for pred in image['GPT_predictions']}
                gt_categories = {gt['category'] for gt in image['ground_truths']}

                # Compare the categories and count the matches
                matches = len(pred_categories & gt_categories)
                total_matches += matches

                # Update the total number of ground truth boxes
                total_gt_boxes += len(gt_categories)

                # Calculate the accuracy for the image
                accuracy = matches / len(gt_categories) if gt_categories else 0

                # Write the result for the image to Accuracy_YOLO_API.txt
                f.write(f"Image ID: {image['image_id']}, Accuracy: {accuracy}\n")

            # Calculate the overall accuracy (mAP)
            overall_accuracy = total_matches / total_gt_boxes if total_gt_boxes else 0

            # Write the overall accuracy to Accuracy_YOLO_API.txt
            f.write(f"Overall mAP: {overall_accuracy}\n")

    @staticmethod
    def calculate_FINAL_accuracy():
        # Load the data from final_result.json
        with open('final_result.json') as f:
            data = json.load(f)

        total_matches = 0
        total_gt_boxes = 0

        # Open Accuracy_Final.txt for writing
        with open('Accuracy_Final.txt', 'w') as f:
            # Process each image in the data
            for image in data:
                # Extract the categories from the predictions and ground truths
                pred_categories = {pred['category'] for pred in image['predictions']}
                gt_categories = {gt['category'] for gt in image['ground_truths']}

                # Compare the categories and count the matches
                matches = len(pred_categories & gt_categories)
                total_matches += matches

                # Update the total number of ground truth boxes
                total_gt_boxes += len(gt_categories)

                # Calculate the accuracy for the image
                accuracy = matches / len(gt_categories) if gt_categories else 0

                # Write the result for the image to Accuracy_Final.txt
                f.write(f"Image ID: {image['image_id']}, Accuracy: {accuracy}\n")

            # Calculate the overall accuracy (mAP)
            overall_accuracy = total_matches / total_gt_boxes if total_gt_boxes else 0

            # Write the overall accuracy to Accuracy_Final.txt
            f.write(f"Overall mAP: {overall_accuracy}\n")



