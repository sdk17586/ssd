import os
import gc
import sys
import json
import time
import torch
import numpy as np
import pytorch_lightning as pl

from loguru import logger
from sklearn.metrics import accuracy_score
from torch.optim import Adam, AdamW, SGD, RMSprop
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


basePath = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir)))
sys.path.extend([basePath, os.path.join(basePath, "model")])


from sender import sendMsg
from model import createModel
from utils import MultiBoxLoss


class CustomEarlyStopping(EarlyStopping):
    def on_validation_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        pass

    def on_train_end(self, trainer, pl_module):
        # instead, do it at the end of training loop
        self._run_early_stopping_check(trainer)


class Callback(pl.LightningModule):
    def __init__(self, param):
        super(Callback, self).__init__()

        self.param = param

        self.param.Log.info("Create Model...")
        createModelStartTime = time.time()

        self.model = createModel(
            pretrained=self.param.pretrained,
            preChannel=self.param.preChannel,
            channel=self.param.grayScale,
            preNumClasses=self.param.preNumClasses,
            numClasses=len(self.param.classNameList),
            weightPath=self.param.originWeightPath,
            backboneWeightPath=self.param.backboneWeightPath,
            device=self.param.device
        )

        createModelTime = time.time() - createModelStartTime
        self.param.Log.info(f"Finsh Create Model, Duration : {round(createModelTime, 4)} sec")

        self.model.names = list(self.param.classNameList)
        self.model.nc = len(self.param.classNameList)
        self.model.n_classes = len(self.param.classNameList)
        self.model.to(self.param.device)

        self.curEpoch = 1
        self.trainBatchEpoch = int(len(self.param.trainDataLoader.dataset)) / int(self.param.batchSize)
        self.validBatchEpoch = int(len(self.param.validDataLoader.dataset)) / int(self.param.batchSize)

        self.trainLoss = 0
        self.trainAccuracy = 0
        self.validLoss = 0
        self.validAccuracy = 0

        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = MultiBoxLoss(priors_cxcy=self.model.priors_cxcy, device=self.param.device)

        self.servInfo = self.param.servInfo
        self.sendStatusUrl = "{}:{}/{}".format(
            self.servInfo["servIp"],
            self.servInfo["servPort"],
            self.servInfo["sendStatusUrl"]
        )

        self.sendResultUrl = "{}:{}/{}".format(
            self.servInfo["servIp"],
            self.servInfo["servPort"],
            self.servInfo["sendResultUrl"]
        )

    def forward(self, image):
        predicted_locs, predicted_scores = self.model(image)

        return predicted_locs, predicted_scores

    def configure_callbacks(self):
        minModeList = ["loss", "valLoss"]

        if str(self.param.monitor) in minModeList:
            mode = "min"
        else:
            mode = "max"

        if self.param.earlyStopping:
            self.earlyStop = CustomEarlyStopping(monitor=self.param.monitor, mode=mode, patience=self.param.patience, verbose=True)

            return [self.earlyStop]
        else:
            return None

    def configure_optimizers(self):
        lr = float(self.param.learningRate)

        biases = list()
        not_biases = list()
        for param_name, param in self.model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)

        if self.param.optimizer == 'adam':
            self.optimizer = Adam(self.model.parameters(), lr=lr)
        elif self.param.optimizer == 'adamw':
            self.optimizer = AdamW(self.model.parameters(), lr=lr)
        elif self.param.optimizer == 'sgd':
            # self.optimizer = SGD(params=self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            self.optimizer = SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}], lr=lr, momentum=0.9, weight_decay=5e-4)
        elif self.param.optimizer == 'rmsprop':
            self.optimizer = RMSprop(self.model.parameters(), lr=lr, momentum=0, weight_decay=1e-5)

    def train_dataloader(self):

        return self.param.trainDataLoader
    
    def val_dataloader(self):

        return self.param.validDataLoader

    def on_train_start(self):

        self.param.Log.info("Starting Training...")
        self.param.Log.info("   Epochs:             {}".format(self.param.epoch))
        self.param.Log.info("   Batch size:         {}".format(self.param.batchSize))
        self.param.Log.info("   Device:             {}".format(self.param.device.type))
        self.param.Log.info("   Optimizer:          {}".format(self.param.optimizer))
        self.param.Log.info("   LearningRate:       {}".format(self.param.learningRate))
        self.param.Log.info("   Image Scaling:      {}".format((self.param.imageSize, self.param.imageSize, self.param.grayScale)))
        self.param.Log.info("   Labels:             {}".format(self.param.classNameList))
        self.param.Log.info("   EarlyStopping:      {}".format(self.param.earlyStopping))
        self.param.Log.info("   Mode/Patience:      {} / {}".format(self.param.monitor, self.param.patience))

        self.param.Log.info("Training epoch={}/{}".format(
            self.curEpoch,
            self.param.epoch
        ))

        self.epochStartTime = time.time()

        logger.debug("Starting Training...")

        logger.debug("Training epoch={}/{}".format(
            self.curEpoch,
            self.param.epoch
        ))

    def training_step(self, batch, batch_idx):
        
        image, bbox, label = batch

        image = image.to(self.param.device)
        boxes = [box.to(self.param.device) for box in bbox]
        labels = [_label.to(self.param.device) for _label in label]

        self.model.eval()
        predicted_locs, predicted_scores = self(image)

        trainBatchLoss = self.criterion(predicted_locs, predicted_scores, boxes, labels)
        _, _, det_score = self.model.detect_objects(predicted_locs, predicted_scores, min_score=0.5, max_overlap=0.5, top_k=200)

        trainBatchAccuracy = det_score[0].detach().cpu().numpy()
        trainBatchAccuracy = np.mean(trainBatchAccuracy)

        self.model.train()

        self.optimizer.zero_grad()
        trainBatchLoss.backward()
        self.optimizer.step()

        if torch.isfinite(torch.tensor(float(trainBatchLoss.item()))):
            self.trainLoss += float(trainBatchLoss.item())

        self.trainAccuracy += float(trainBatchAccuracy)

        self.param.Log.info("Training batch={}/{}:{}, loss={}, accuracy={}".format(
            int(batch_idx + 1),
            int(self.trainBatchEpoch),
            int(self.curEpoch),
            float("{0:.4f}".format(float(trainBatchLoss.item()))),
            float("{0:.4f}".format(float(trainBatchAccuracy)))
        ))

        logger.debug("Training batch={}/{}:{}, loss={}, accuracy={}".format(
            int(batch_idx + 1),
            int(self.trainBatchEpoch),
            int(self.curEpoch),
            float("{0:.4f}".format(float(trainBatchLoss.item()))),
            float("{0:.4f}".format(float(trainBatchAccuracy)))
        ))

    def validation_step(self, batch, batch_idx):
        image, bbox, label = batch

        image = image.to(self.param.device)
        boxes = [box.to(self.param.device) for box in bbox]
        labels = [_label.to(self.param.device) for _label in label]

        self.model.eval()

        with torch.no_grad():
            # set the model in evaluation mode
            predicted_locs, predicted_scores = self(image)

        validBatchLoss = self.criterion(predicted_locs, predicted_scores, boxes, labels)
        _, _, det_score = self.model.detect_objects(predicted_locs, predicted_scores, min_score=0.5, max_overlap=0.5, top_k=200)

        validBatchAccuracy = det_score[0].detach().cpu().numpy()
        validBatchAccuracy = np.mean(validBatchAccuracy)

        if torch.isfinite(torch.tensor(float(validBatchLoss.item()))):
            self.validLoss += float(validBatchLoss.item())

        self.validAccuracy += float(validBatchAccuracy)

    def validation_epoch_end(self, outs):

        self.validLoss /= self.validBatchEpoch
        self.validAccuracy /= self.validBatchEpoch

        self.validLoss = round(float(self.validLoss), 4)
        self.validAccuracy = round(float(self.validAccuracy), 4) if float(self.validAccuracy) <= 1 else 1.0

        self.log("valLoss", self.validLoss)
        self.log("valAccuracy", self.validAccuracy)

        gc.collect()

    def training_epoch_end(self, outputs):
        self.trainLoss /= self.trainBatchEpoch
        self.trainAccuracy /= self.trainBatchEpoch

        self.trainLoss = round(float(self.trainLoss), 4)
        self.trainAccuracy = round(float(self.trainAccuracy), 4) if float(self.trainAccuracy) <= 1 else 1.0

        self.epochEndTime = time.time() - self.epochStartTime
        self.remaningTime = self.epochEndTime * (self.param.epoch - self.curEpoch)
        self.curEpoch += 1

        self.log("loss", self.trainLoss)
        self.log("accuracy", self.trainAccuracy)

        gc.collect()

    def on_train_epoch_end(self):
        self.param.Log.info("Result epoch={}/{}, loss={}, valLoss={}, accuracy={}, valAccuracy={}".format(
            self.curEpoch,
            self.param.epoch,
            self.trainLoss,
            self.validLoss,
            self.trainAccuracy,
            self.validAccuracy,
        ))

        logger.debug("Result epoch={}/{}, loss={}, valLoss={}, accuracy={}, valAccuracy={}".format(
            self.curEpoch - 1,
            self.param.epoch,
            self.trainLoss,
            self.validLoss,
            self.trainAccuracy,
            self.validAccuracy,
        ))

        trainTrialResult = {
            "epoch": self.curEpoch - 1,
            "purposeType": self.param.dataInfo["purposeType"],
            "mlType": self.param.dataInfo["mlType"],
            "loss": self.trainLoss,
            "valLoss": self.validLoss,
            "accuracy": self.trainAccuracy,
            "valAccuracy": self.validAccuracy,
            "isLast": False,
            "remaningTime": self.remaningTime,
            "elapsedTime": self.epochEndTime
        }

        if self.param.earlyStopping:
            self.param.Log.info(f"Early Stopping Patience={self.earlyStop.wait_count}/{int(self.param.patience)}")
            logger.debug(f"Early Stopping Patience={self.earlyStop.wait_count}/{int(self.param.patience)}")
            logger.debug(trainTrialResult)
            torch.save(self.model.state_dict(), self.param.modelWeightPath)
            
            if (self.earlyStop.wait_count == int(self.param.patience)) or (int(self.curEpoch - 1) == self.param.epoch):
                self.param.Log.info("Model is Stopped!")
                logger.debug("Model is Stopped!")

                trainTrialResult["isLast"] = True
                
                logger.debug("Model Weight Save Success!")

                trainResult = {
                    "score": round(float(self.validAccuracy), 4),
                    "trainInfo": trainTrialResult
                }

                tranResultPath = os.path.join(self.param.pathInfo["modelPath"], "trainResult.json")

                with open(tranResultPath, "w") as f:
                    json.dump(trainResult, f)
            
            else:
                print("error")

        else:
            logger.debug(trainTrialResult)
            torch.save(self.model.state_dict(), self.param.modelWeightPath)
            
            if int(self.curEpoch - 1) == self.param.epoch:

                # Trial Send (isLast=True)
                trainTrialResult["isLast"] = True
                logger.debug("Model Weight Save Success!")

                trainResult = {
                    "score": round(float(self.validLoss), 4),
                    "trainInfo": trainTrialResult
                }

                tranResultPath = os.path.join(self.param.pathInfo["modelPath"], "trainResult.json")

                with open(tranResultPath, "w") as f:
                    json.dump(trainResult, f)

            else:
                print("err")
        # 초기화
        self.trainLoss = 0
        self.validLoss = 0
        self.trainAccuracy = 0
        self.validAccuracy = 0