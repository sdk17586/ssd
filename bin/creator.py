from loguru import logger
from checker import  pathChecker, datChecker
from torch.utils.data import DataLoader, random_split
from datalib import CustomImageDataset
from maker import makeMaskImage


def createDataLoader(filePathList, classInfo, classNameList, imageSize, batchSize, grayScale, trainRate):

    logger.info("createDataLoader")
    imagePathList = pathChecker(filePathList, dataType="image")
    imagePathList = datChecker(imagePathList)

    if len(imagePathList) > 50:
        logger.warning("Dataset size > 50")

        trainSize = int(len(imagePathList) * float(trainRate["train"] / 100))
        validSize = len(imagePathList) - int(trainSize)

        trainImageList, validImageList = random_split(imagePathList, [trainSize, validSize])

        trainDataset = CustomImageDataset(trainImageList, classInfo, classNameList, imageSize, batchSize, grayScale)
        trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, drop_last=True, pin_memory=True, collate_fn=trainDataset.collate_fn)

        validDataset = CustomImageDataset(validImageList, classInfo, classNameList, imageSize, batchSize, grayScale)
        validDataLoader = DataLoader(validDataset, batch_size=batchSize, shuffle=True, drop_last=True, pin_memory=True, collate_fn=validDataset.collate_fn)

        return trainDataLoader, validDataLoader

    else:
        logger.warning("Dataset size < 50")
        trainSize = len(imagePathList)
        validSize = 0

        trainImgList, validImageList = random_split(imagePathList, [int(trainSize), int(validSize)])
        
        trainDataset = CustomImageDataset(trainImgList, classInfo, classNameList, imageSize, batchSize, grayScale)
        trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, drop_last=True, pin_memory=True, collate_fn=trainDataset.collate_fn)

        return trainDataLoader, trainDataLoader