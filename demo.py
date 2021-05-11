# -*- coding: utf-8 -*-
'''
@Time          : 20/04/25 15:49
@Author        : huguanghao
@File          : demo.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
'''

# import sys
# import time
# from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import argparse
import youtube_dl
import os
from os.path import join, isfile

"""hyper parameters"""
use_cuda = True
resultimage = []

def detect_cv2(imgfile, m):
    import cv2

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    class_names = load_class_names(namesfile)

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))
    img = plot_boxes_cv2(img, boxes[0], savename='predictions.jpg', class_names=class_names)
    if(img is not None):
        resultimage.append(img)

def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='./cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='./checkpoints/Yolov4_epoch1.pth',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-videofile', type=str,
                        default='./video.mkv',
                        help='path of your video file.', dest='videofile')
    args = parser.parse_args()

    return args

def makeImage():  
    # ffmpeg 기능
    # 다운로드한 영상을 ffmpeg을 이용해 원하는 포멧으로 변환(copy)
    # -ss [시작시간] 
    # -t [길이만큼 동영상을 뽑아냄] 
    # -i [input 동영상이름] 
    # -r [프레임레이트, 원본의 fps보다 높이 설정하면 의미 없다.]
    # -s [출력해상도, 설정 안할시 원본 해상도] 
    # -qscale:v 2 -f image2 [이미지이름]
    # eg) -t 설정 : 10, -r 설정 : 24  =>  초당 24 프레임 추출 x 10초 = 240장
    os.system("ffmpeg -i video.mkv -ss 00:00:00 -t 10 -r 4 -s 1280x720 -qscale:v 2 -f image2 testdata/test-%d.jpg")
    
    # 변환된(프레임 이미지화된) test image들을 files 배열에 집어넣는다.
    files = [f for f in os.listdir('/content/drive/MyDrive/YoloV4/darknet/testdata') if isfile(join('/content/drive/MyDrive/YoloV4/darknet/testdata', f))]

    # test image 목록 출력
    print("생성된 test-data 목록은 다음과 같습니다.")
    print(files)
    return files

def info(videoPath):
    import cv2
    # VideoCapture 객체 정의
    cap = cv2.VideoCapture(videoPath)
    # 프레임 너비/높이, 초당 프레임 수 확인
    width = cap.get(3)  # (=cv.CAP_PROP_FRAME_WIDTH)
    height = cap.get(4) # (=cv.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(5)  # (=cv.CAP_PROP_FPS)
    print('프레임 너비 : %d, 프레임 높이 : %d, 초당 프레임 수 : %d' %(width, height, fps))

def saveImage():
    import cv2
    for i in range(len(resultimage)):
        img = resultimage[i]
        output = "./resultdata/test-%d.jpg" % (i+1)
        print("savepath"+output)
        cv2.imwrite(output,img)

if __name__ == '__main__':
    import shutil
    #임시로 유튭으로 만
    print("url을 입력하시오")
    link = input("")
    ydl_opts = {
    'outtmpl': 'video',
    'videoformat' : "mkv",
    'postprocessors': [{
        'key': 'FFmpegVideoConvertor',
        'preferedformat': 'mkv',  # one of avi, flv, mkv, mp4, ogg, webm
    }],
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])

    args = get_args()
    m = Darknet(args.cfgfile)
    m.load_weights(args.weightfile)
    print('Loading weights from %s... Done!' % (args.weightfile))
    if use_cuda:
        m.cuda()

    # 동영상을 이미지로 저장할 경로와 결과 이미지를 저장할 경로 
    # weight 파일이 위치한 곳에 없을때의 예외처리들
    testpath = "./testdata"
    resultpath = "./resultdata"
    weightsFilePath = "./"+str(args.weightfile)

    if(os.path.isdir(testpath)):
        shutil.rmtree(testpath)
    if(os.path.isdir(resultpath)):
        shutil.rmtree(resultpath)
    
    os.mkdir(testpath)
    os.mkdir(resultpath)

    if(not os.path.isfile(weightsFilePath)):
        os.system("wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights")
        args.weightfile = "./yolov4.weights"

    info(args.videofile)
    files = makeImage()

    for i in range (0, len(files)):
        imagesPath = "./testdata/"+files[i]
        print(files[i]+"를 학습데이터로 전환합니다.")
        detect_cv2(imagesPath,m)
    saveImage()
