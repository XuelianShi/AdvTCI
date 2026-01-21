from ultralytics import YOLO
from PIL import Image
import cv2
import torch


model = torch.hub.load("ultralytics/yolov5", "custom", path = r"C:\Users\Lenovo\OneDrive\桌面\yolov5准备\best1.pt", device = '0')

# path = r"C:\Users\a\Downloads\222.png"
def detect(path):
    result = model(path)
    # result.save()

    shape = result.pandas().xyxy[0].xmin.shape[0]

    if shape == 0:
        xmin, ymin, xmax, ymax, conf = 0, 0, 0, 0, 100

    else:
        xmin, ymin, xmax, ymax, conf = result.pandas().xyxy[0].xmin[0], result.pandas().xyxy[0].ymin[0], result.pandas().xyxy[0].xmax[0], result.pandas().xyxy[0].ymax[0], result.pandas().xyxy[0].confidence[0]

    return int(xmin), int(ymin), int(xmax), int(ymax), conf, shape
    # return result


# xmin, ymin, xmax, ymax, conf, shape = detect(path)
# print(xmin, ymin, xmax, ymax, conf, shape)

def detect_save(path):
    result = model(path)
    result.save()

    shape = result.pandas().xyxy[0].xmin.shape[0]

    if shape == 0:
        xmin, ymin, xmax, ymax, conf = 0, 0, 0, 0, 100

    else:
        xmin, ymin, xmax, ymax, conf = result.pandas().xyxy[0].xmin[0], result.pandas().xyxy[0].ymin[0], result.pandas().xyxy[0].xmax[0], result.pandas().xyxy[0].ymax[0], result.pandas().xyxy[0].confidence[0]

    return int(xmin), int(ymin), int(xmax), int(ymax), conf, shape
    # return result

# path1 = r'C:\Users\Lenovo\OneDrive\桌面\yolov5准备\TT00K-1\images\527.jpg'
# # path2 = r"digital_sample_show/16065.jpg"
# #
# A1 = detect_save(path1)
# A2 = detect_save(path2)






