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



    def check_bbox_and_labels(self, results):
        results_bbox_and_labels = {}
        overall_correct_predictions = 0
        overall_count = 0

        for result in results:
            image_id = result['image_id']
            predictions = result['predictions']
            ground_truths = result['ground_truths']

            correct_predictions = 0
            for pred in predictions:
                for gt in ground_truths:
                    if pred['category'] == gt['category'] and self.calculate_iou(pred['bbox'], gt[
                        'bbox']) > 0.5:  # IoU threshold can be adjusted
                        correct_predictions += 1
                        break

            results_bbox_and_labels[image_id] = correct_predictions / len(predictions) if len(predictions) > 0 else 0
            overall_correct_predictions += correct_predictions
            overall_count += len(predictions)

        overall_accuracy = overall_correct_predictions / overall_count if overall_count > 0 else 0

        with open('results_bbox_and_labels.txt', 'w') as file:
            for image_id, accuracy in results_bbox_and_labels.items():
                file.write(f'Image ID: {image_id}, Correct BBox and Labels: {accuracy}\n')
            file.write(f'Overall Accuracy of BBox and Labels: {overall_accuracy}\n')

        return results_bbox_and_labels, overall_accuracy


