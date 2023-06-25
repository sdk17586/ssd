import os
import json
import cv2
import torch

from torchvision import transforms
from PIL import Image


# Get file list each className
def getFileList(classInfo, className):

    for _class in classInfo:
        if className == _class["className"]:
            return className
        elif "sourceClassName" in _class and className == _class["sourceClassName"]:
            return _class["className"]
    
    return None


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, imagePathList, classInfo, classNameList, imageSize, batchSize, grayScale):
        self.imagePathList = imagePathList
        self.classNameList = classNameList
        self.classInfo = classInfo
        self.grayScale = grayScale
        self.imageSize = imageSize
        self.batchSize = batchSize

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

    def __len__(self):
        return len(self.imagePathList)

    def __getitem__(self, idx):
        bbox = []
        label = []
        difficulty = []

        imagePath = self.imagePathList[idx]
        rootPath, file = os.path.split(imagePath)
        fileName, _ = os.path.splitext(file)

        image = cv2.imread(imagePath)
        height, width = image.shape[:2]

        if self.grayScale == 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image)
        image = self.transform(image)

        # transform = T.ToPILImage()
        # newImage = transform(image)
        # draw = ImageDraw.Draw(newImage)
        datData = os.path.join(rootPath, fileName + ".dat")
        with open(datData, "r") as f:
            datData = json.load(f)

        polygonData = datData["polygonData"]
        for polygon in polygonData:
            x1 = float(polygon["position"][0]["x"] / width)
            y1 = float(polygon["position"][0]["y"] / height)
            x2 = float(polygon["position"][1]["x"] / width)
            y2 = float(polygon["position"][1]["y"] / height)

            # draw.rectangle((int(x1), int(y1), int(x2), int(y2)), outline="green", width=3)
            newX1 = max(min(x1, width), 0)
            newY1 = max(min(y1, height), 0)
            newX2 = max(min(x2, width), 0)
            newY2 = max(min(y2, height), 0)

            className = getFileList(self.classInfo, polygon["className"])
            if className is None:
                continue

            bbox.append((newX1, newY1, newX2, newY2))
            label.append(self.classNameList.index(className))
            difficulty.append(0)

        # import time
        # newImage.save("{}.png".format(time.time()))

        if int(self.batchSize) == 1:
            image = image.squeeze(0)

        # image = torch.from_numpy(np.array(image)).float()
        bbox = torch.FloatTensor(bbox)
        label = torch.LongTensor(label)

        # difficulty = torch.LongTensor(difficulty)

        return image, bbox, label

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)

        return images, boxes, labels  # tensor (N, 3, 300, 300), 3 lists of N tensors each

# if __name__ == "__main__":
#     from PIL import ImageDraw
#     import torchvision.transforms as T
#     imagePathList = ["/data/mjkim/ssd/sample/img/catdog100/cat-7.jpg"]

#     classInfo = [
#         {
#             "classId": "e99f5",
#             "className": "cat",
#             "color": "#ef8783",
#             "desc": "",
#             "dsId": "e3375",
#             "dsName": "[CLF] LG\uc5d4\uc194_Train_dataset",
#             "isClick": False,
#             "showTf": True
#         },
#         {
#             "classId": "76e2e",
#             "className": "dog",
#             "color": "#0000ff",
#             "desc": "",
#             "dsId": "e337e",
#             "dsName": "[CLF] LG\uc5d4\uc194_Train_dataset",
#             "isClick": False,
#             "showTf": True
#         }
#     ]
    
#     classNameList = ["cat", "dog"]
#     imageSize = 300
#     batchSize = 1
#     grayScale = 3
#     b = CustomImageDataset(imagePathList, classInfo, classNameList, imageSize, batchSize, grayScale)
#     transform = T.ToPILImage()
#     for data in b:
#         image, bbox, label = data
#         print(bbox)
#         img = transform(image)
#         draw = ImageDraw.Draw(img)
        
#         draw.rectangle([(bbox[0][0], bbox[0][1]), (bbox[0][2], bbox[0][3])], fill=(255, 0, 0))
#         draw.rectangle([(bbox[1][0], bbox[1][1]), (bbox[1][2], bbox[1][3])], fill=(0, 255, 0))
        
#         img.save("./img.png")
