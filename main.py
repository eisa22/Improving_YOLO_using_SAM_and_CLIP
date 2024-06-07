import YOLO
import Helper_Functions
import Validation
import OpenAI_Key
from pycocotools.coco import COCO


# Control Panel
camera_idx = 0
webcam = False
conf_threshold = 0.5
conf_threshold_crawler = 0.1
sam_checkpoint = 'SegmentAnythingModel/sam_vit_h_4b8939.pth'
openAI_key = OpenAI_Key.openAI_key

debug_mode = False
enable_crawler = True
enable_evaluation_mode = False

# Initialize COCO ground truth
coco_gt = COCO('Datasets/Coco/annotations/instances_val2017_subset.json')

iou_scores = []


if __name__ == "__main__":

    detector = YOLO.ObjectDetection(camera_idx, webcam, debug_mode, conf_threshold)
    helper = Helper_Functions.Helper_Functions()
    validation = Validation.Validation()

    image_ids = coco_gt.getImgIds()
    all_ious = []

    for image_id in image_ids:

        img_info = coco_gt.loadImgs(image_id)[0]
        print("Processing image: ", img_info['file_name'])
        coco_url = img_info['coco_url']
        YoloData, confidences, image_url, frame = detector.get_YOLO_Results(coco_url, image_id)
        validation.calculate_IoU(YoloData, coco_gt, coco_url, debug_mode)

    validation.calculate_average_iou()