import os
import gc
import cv2
import glob
import json
import time
import torch
import mimetypes
import GPUtil as gp
from loguru import logger


def initGpuChecker():

    ''' 
        정보 1. torch, pytorch_lightning에서 gpu가 할당이 안되는 이슈로 수정
            - torch.cuda.set_device(device) 적용안됨
            - torch가 import 된 이 후 시점에서, os.environ을 세팅해도 적용이 안되는 문제
                ex) 내부 변수들은 .to(device)로 처리가 가능하나, pl.lightning 실행시 특정 타겟의 gpu를 설정할 수 없음 
                ex) device=cuda:1 일 경우, pl.lightning은 cuda:0을 사용하는데 다른 부분들이 cuda:1로 할당이 되어 에러 발생

        * 따라서 initChecker를 통해 가용 gpuIdx를 확인하고, torch가 import 되기전에 os.environ으로 환경변수를 설정하여 적용해야 동작됨
            - 기존에 device="cuda:0" 방식에서 device("cuda")로 변경 (환경변수 설정으로 인하여, 기본 gpu 1장이 할당되기 때문)
    '''

    gpus = gp.getGPUs()

    if len(gpus) > 0:
        usableDevices = {str(gpu.id): int(gpu.memoryFree) for gpu in gpus}
        gpuNo = max(usableDevices, key=usableDevices.get)
    else:
        gpuNo = ""

    return gpuNo


def gpuChecker(log=None, gpuIdx="auto", gpuNo=None):

    '''
        정보 1. Gpu는 기본적으로 "auto" 모드로 동작
            - parameter로 gpuIdx를 받게되는데, 앞서 initChecker에서 gpu를 할당하므로, device 종류는 ["cpu", "cuda"]가 됨
    '''

    if gpuIdx == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if str(device) == "cpu":
            deviceType = "cpu"
            device = torch.device("cpu")

            if log is not None:
                log.warning("Cuda is not available...")
                log.info("Set Device: CPU")

        else:
            deviceType = "gpu"
            device = torch.device("cuda")

            if log is not None:
                log.info("Checking gpuIdx auto...")
                log.info("Set Device: GPU:{}".format(gpuNo))

    return device, deviceType


def pathChecker(pathList, type="dir", dataType="image"):

    '''
        정보 1. 데이터 타입이 이미지/비디오인지 체크 / 파일이 열리는지 안열리는지 체크하여 열리는 파일만 리스트 형태로 변환
    '''

    logger.info("Checking Path for dataset...")
    pathChekerTime = time.time()

    filterStrList = ["THUM", "MASK_"]
    filePathList = []

    if type == "dir":
        for fileDirPath in pathList:
            # logger.debug(f"Checking fileDirPath : {fileDirPath}")

            # glob 리스트
            globList = glob.glob(os.path.join(fileDirPath, "*.*"))

            # filterStrList, fileType != None, fileType == "image/video" 체크
            filteredList = [globPath for globPath in globList if not any(fstr in globPath for fstr in filterStrList) and
                            mimetypes.guess_type(globPath)[0] is not None and
                            mimetypes.guess_type(globPath)[0].split("/")[0] == dataType]

            # 이미지 파일이 열리는지 체크
            if dataType == "image":
                filePathList.extend([filePath for filePath in filteredList if cv2.imread(filePath) is not None])

            # 비디오 파일이 열리는지 체크
            elif dataType == "video":
                filePathList.extend([filePath for filePath in filteredList if cv2.VideoCapture(filePath).read()[0] is True])

    else:
        # filterStrList, fileType != None, fileType == "image/video" 체크
        filteredList = [path for path in pathList if not any(fstr in pathList for fstr in filterStrList) and
                        mimetypes.guess_type(path)[0] is not None and
                        mimetypes.guess_type(path)[0].split("/")[0] == dataType]

        # 이미지 파일이 열리는지 체크
        if dataType == "image":
            filePathList.extend([filePath for filePath in filteredList if cv2.imread(filePath) is not None])

        # 비디오 파일이 열리는지 체크
        elif dataType == "video":
            filePathList.extend([filePath for filePath in filteredList if cv2.VideoCapture(filePath).read()[0] is True])

    filteredList = None

    gc.collect()
    pathChekerTotalTime = time.time() - pathChekerTime
    logger.info(f"Finish Checking Path for dataset, Duration : {round(pathChekerTotalTime, 4)} sec")

    return filePathList


# .DAT 존재, 내부 정보가 있는 파일만 추출
def datChecker(filePathList):

    '''
        정보 1. .dat 파일이 있고, 내부 정보가 있는 이미지/비디오 파일만 추출해서 리스트 형태로 변환
    '''

    pathList = []

    for filePath in filePathList:
        rootName, fileExtension = os.path.splitext(filePath)
        datPath = rootName + ".dat"

        if os.path.isfile(datPath):
            with open(datPath, "r") as jsonFile:
                datData = json.load(jsonFile)

            if os.stat(datPath).st_size == 0:
                logger.warning(f'{datPath} file is Empty.')
                continue

            elif not datData["polygonData"] and not datData["brushData"]:
                logger.warning(f'{datPath} file has No Label Data.')
                continue

            else:
                pathList.append(filePath)
        else:
            logger.warning(f'{datPath} file not exist.')

    return pathList