import os
import sys
import cv2
import json
import time
import torch
import traceback

from PIL import Image
from loguru import logger
from torchvision import transforms


basePath = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir)))
sys.path.extend([basePath, os.path.join(basePath, "model")])

from model import createModel
from logger import Logger
from checker import gpuChecker, initGpuChecker

gpuNo = initGpuChecker()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpuNo


class Predictor():
    def __init__(self, pathInfo):

        self.pathInfo = pathInfo
        self.modelPath = self.pathInfo["modelPath"] if "modelPath" in self.pathInfo else '/app'
        self.weightPath = self.pathInfo["weightPath"] if "weightPath" in self.pathInfo else "/app/weight"
        self.log = Logger(logPath=os.path.join(self.modelPath, "log/predict.log"), logLevel="info")

        # set cpu/gpu
        self.setGpu()

        if os.path.isfile(os.path.join(self.weightPath, "weight.pth")):
            with open(os.path.join(self.weightPath, "classes.json"), "r") as jsonFile:
                self.classesJson = json.load(jsonFile)

            self.classNameList = [classInfo["className"] for classInfo in self.classesJson["classInfo"]]
            self.imageSize = self.classesJson["imageInfo"]["imageSize"] if "imageSize" in self.classesJson["imageInfo"] else 300
            self.grayScale = int(self.classesJson["imageInfo"]["imageChannel"])

            if self.grayScale == 1:
                self.transform = transforms.Compose([
                    transforms.Resize((self.imageSize, self.imageSize)),
                    transforms.Grayscale(num_output_channels=self.grayScale),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=0.5, std=0.5)
                ])

            else:
                self.transform = transforms.Compose([
                    transforms.Resize((self.imageSize, self.imageSize)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

            # model load
            logger.info("Model Loading ...")

            modelLoadStartTime = time.time()

            self.model = createModel(
                pretrained=False,
                channel=self.grayScale,
                numClasses=len(self.classNameList),
                backboneWeightPath="/app/backboneWeight",
                device=self.device
            )

            self.model.load_state_dict(torch.load(os.path.join(self.weightPath, "weight.pth"), map_location=self.device))
            self.model.eval()
            self.model.to(self.device)

            modelLoadTime = time.time() - modelLoadStartTime
            logger.debug(f"Model Load Success, Duration : {round(modelLoadTime, 4)} sec")

        else:
            raise Exception("This Model is not Trained Model, Not Found Model's Weight File")

    def setGpu(self):
        self.device, self.deviceType = gpuChecker(log=self.log, gpuIdx="auto")

    def runPredict(self, predImage):

        try:
            logger.info("Starting Model Predict...")
            logger.info("-"*100)
            logger.info("  Device:             {}  ".format(self.device.type))
            logger.info("  Image Scaling:      {}  ".format((self.imageSize, self.imageSize, self.grayScale)))
            logger.info("  Labels:             {}  ".format(self.classNameList))

            totalStartTime = time.time()

            # 이미지 예측을 위한 전처리
            logger.info("Input Data Preprocessing for Model...")
            preProStartTime = time.time()

            result = []
            heatMapImage = None
            originImage = predImage.copy()
            height, width = originImage.shape[:2]

            if self.grayScale == 1:
                predImage = cv2.cvtColor(predImage, cv2.COLOR_BGR2GRAY)
            else:
                predImage = cv2.cvtColor(predImage, cv2.COLOR_BGR2RGB)

            predImage = Image.fromarray(predImage)
            predImage = self.transform(predImage)
            predImage = predImage.unsqueeze(0)
            predImage = predImage.to(self.device)

            preProTime = time.time() - preProStartTime
            logger.debug(f"Input Data Preprocessing Success, Duration : {round(preProTime, 4)} sec")

            # 이미지 예측시작
            logger.info("Predict Start...")

            predStartTime = time.time()
            with torch.no_grad():
                predicted_locs, predicted_scores = self.model(predImage)

            predTime = time.time() - predStartTime
            logger.debug(f"Predict Success, Duration : {round(predTime, 4)} sec")

            # 예측 결과 형태 변환
            transferOutputStartTime = time.time()
            logger.info("Output Format Transfer...")

            det_boxes, det_labels, det_scores = self.model.detect_objects(predicted_locs, predicted_scores, min_score=0.5, max_overlap=0.5, top_k=200)

            original_dims = torch.FloatTensor([width, height, width, height]).to(self.device).unsqueeze(0)

            det_boxes = det_boxes[0] * original_dims
            det_labels = [self.classNameList[i] for i in det_labels[0].to(self.device).tolist()]
            det_scores = det_scores[0].tolist()

            for i in range(det_boxes.size(0)):
                box_location = det_boxes[i].tolist()
                accuracy = round(float(det_scores[i]), 4)
                className = det_labels[i]

                x1 = float(box_location[0])
                y1 = float(box_location[1])
                x2 = float(box_location[2])
                y2 = float(box_location[3])

                # cv2.rectangle(originImage, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 5)
                newX1 = max(min(x1, width), 0)
                newY1 = max(min(y1, height), 0)
                newX2 = max(min(x2, width), 0)
                newY2 = max(min(y2, height), 0)

                tmpResult = {
                    "className": className,
                    "accuracy": accuracy,
                    "cursor": 'isRect',
                    "needCount": 2,
                    "position": [
                        {"x": newX1, "y": newY1},
                        {"x": newX2, "y": newY2},
                    ]
                }
                result.append(tmpResult)

            # cv2.imwrite("./test.png", originImage)
            logger.debug(result)
            trasferTime = time.time() - transferOutputStartTime
            logger.debug(f"Output Format Transfer Success, Duration : {round(trasferTime, 4)} sec")

            totalTime = time.time() - totalStartTime
            logger.info(f"Finish Model Predict, Duration : {round(totalTime, 4)} sec")
            logger.info("-"*100)

        except Exception as e:
            logger.error(f"Error :{str(e)}")
            logger.error(f"Traceback : {str(traceback.format_exc())}")

        return result, heatMapImage


# if __name__ == "__main__":
#     pathInfo = {
#         "modelPath": "/data/mjkim/docker_det/ssd",
#         "weightPath": "/data/mjkim/docker_det/ssd/weight",
#     }

#     path = "/data/mjkim/docker_det/sample/img/5022.jpg"
#     # img = cv2.imread("/data/mjkim/deeplabv3/catdog.png")
#     img = cv2.imread(path)
#     p = Predictor(pathInfo)

#     # while True:
#     predResult, heatMapImage = p.runPredict(img)
