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
from tool.make_annotation import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import argparse
import youtube_dl
import os
import os, glob
import os.path
import cv2
from os.path import join, isfile
 
"""hyper parameters"""
use_cuda = True
resultimage = []
Annotation = []

def detect_cv2(annotationType,labelName,imgfile, m):
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

    annotation = []
    img, annotation = plot_boxes_cv2(labelName,annotationType,img, boxes[0], savename='predictions.jpg', class_names=class_names)
    if(img is not None):
        resultimage.extend(img)
        Annotation.extend(annotation)

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
    parser.add_argument('-labelName',type=str,help='학습 데이터 생성 라벨 입력',dest='labelName',action='append')

    parser.add_argument('-annotationType',type=str,help='어노테이션 타입 설정',dest='annotationType',default='txt')

    parser.add_argument('-urlLink', type=str,
                        default='https://www.youtube.com/watch?v=03eU5eMhGZk', dest = 'urlLink') # url
    parser.add_argument('-endTime', type=int,
                        default=30, dest='endTime') # 종료시간
    parser.add_argument('-mode', type=bool,
                        default=False, dest='mode') # 객체 클래스 추가
    parser.add_argument('-className', type=str,
                        default="", dest='className') # 객체 클래스 추가                              
    args = parser.parse_args()
    return args

# 영상 길이 구하는 함수
def getVideoInfo(filename):
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print("could not open :", filename)
        exit(0)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    time = length // fps
    minu = int(time // 60)
    sec = int(time % 60)
    print("ffmpeg으로 자를 영상의 종료 지점을 입력하시오 ( 영상은 "  +str(minu) +"분 " +str(sec) +"초 입니다.)")

def makeImage(timeInfo):  
    # ffmpeg 기능
    # 다운로드한 영상을 ffmpeg을 이용해 원하는 포멧으로 변환(copy)
    # -ss [시작시간] 
    # -t [길이만큼 동영상을 뽑아냄] 
    # -i [input 동영상이름] 
    # -r [프레임레이트, 원본의 fps보다 높이 설정하면 의미 없다.]
    # -s [출력해상도, 설정 안할시 원본 해상도] 
    # -qscale:v 2 -f image2 [이미지이름]
    # eg) -t 설정 : 10, -r 설정 : 24  =>  초당 24 프레임 추출 x 10초 = 240장
    filename = "./video." + fileExtension # 파일 이름을 확장자를 붙여 만듬
    getVideoInfo(filename) # 그 후 비디오 정보 출력하는 함수 호출
    os.system("ffmpeg -i video.{} -ss 00:00:00 -t {} -r 2 -s {}x{} -qscale:v 2 -f image2 testdata/test-%d.jpg".format(fileExtension, timeInfo, width, height))
    # 변환된(프레임 이미지화된) test image들을 files 배열에 집어넣는다.
    files = [f for f in os.listdir('./testdata') if isfile(join('./testdata', f))]
    # test image 목록 출력
    print("생성된 test-data 목록은 다음과 같습니다.")
    print(files)
    return files

def info(videoPath):
    import cv2
    global width, height
    # VideoCapture 객체 정의
    cap = cv2.VideoCapture(videoPath)
    # 프레임 너비/높이, 초당 프레임 수 확인
    width = cap.get(3)  # (=cv.CAP_PROP_FRAME_WIDTH)
    height = cap.get(4) # (=cv.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(5)  # (=cv.CAP_PROP_FPS)
    width = int(width)
    height = int(height)
    print('프레임 너비 : %d, 프레임 높이 : %d, 초당 프레임 수 : %d' %(width, height, fps))

def saveImage():
    import cv2
    for i in range(len(resultimage)):
        img = resultimage[i]
        output = "./resultdata/test-%d.jpg" % (i+1)
        print("savepath"+output)
        cv2.imwrite(output,img)

def saveAnnotation():
    for i in range(len(Annotation)):
        output = "./resultdata/test-%d.txt" % (i+1)
        print("savepath"+output)
        f = open("./resultdata/test-%d.txt" % (i+1), 'w')
        for j in range(len(Annotation[i])):
          if (j == 0):
            data = "%d " % Annotation[i][j]
          else:
            data = "%f " % Annotation[i][j]
          f.write(data)
        f.close()

# 파일 확장자 추출하는 함수
def extractExtension(): 
    targerdir = r"./" # 상대 경로로 현재 경로를 targetdir으로 설정
    files = os.listdir(targerdir)
    condition = 'video.*' # 와일드 문자 이용해서 video로 시작하는것 불러오기
    csvfiles = glob.glob(condition)
    csvfiles = str(csvfiles)
    global fileExtension # global 변수 filename
    fileExtension = os.path.splitext(csvfiles)[1] 
    fileExtension = fileExtension[1:-2] # 파일 확장자를 뽑아냄

# 실행후 video 삭제 하기
def deleteVideo():
    print("실행 완료 video를 삭제합니다")
    targerdir = r"./" # 상대 경로로 현재 경로를 targetdir으로 설정
    files = os.listdir(targerdir)
    condition = 'video.*' # 와일드 문자 이용해서 video로 시작하는것 불러오기
    if condition != None:
        csvfiles = glob.glob(condition)
        csvfiles = str(csvfiles)
        obj = csvfiles[2:-2]
        os.remove(obj)

if __name__ == '__main__':
    import shutil
    args = get_args()
    if args.mode == False:
      # 유튜브로 다운로드 받은 영상을 자동으로 video 확장자에 맞게 저장함
      # 동영상 url 링크 로딩 ----------------------
      link = args.urlLink
      os.system("youtube-dl -o \"video.%(ext)s\" {}".format(link))

      # 파일 확장자 추출 메소드 호출 후
      extractExtension()

      # 모델 네트워트 로딩 -----------------------
      m = Darknet(args.cfgfile)
      m.load_weights(args.weightfile)
      print('Loading weights from %s... Done!' % (args.weightfile))
      if use_cuda:
          m.cuda()
      # ------------------------------------------

      # 동영상을 이미지로 저장할 경로와 결과 이미지를 저장할 경로 
      # weight 파일이 위치한 곳에 없을때의 예외처리들
      testpath = "./testdata"
      resultpath = "./resultdata"
      weightsFilePath = str(args.weightfile)

      if(os.path.isdir(testpath)):
          shutil.rmtree(testpath)
      if(os.path.isdir(resultpath)):
          shutil.rmtree(resultpath)
      
      os.mkdir(testpath)
      os.mkdir(resultpath)
      # --------------------------------------------------------------
      if(not os.path.isfile(weightsFilePath)):
          os.system("wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights")
          args.weightfile = "./yolov4.weights"
      # -----------------------------------------------------------------

      info(args.videofile)
      timeInfo = args.endTime
      files = makeImage(timeInfo)

      for i in range (0, len(files)):
          imagesPath = "./testdata/"+files[i]
          print(files[i]+"를 학습데이터로 전환합니다.")
          detect_cv2(args.annotationType,args.labelName,imagesPath,m)

      saveImage()
      #saveAnnotation()
      if (args.annotationType=='txt'):
          saveTxtAnnotation(Annotation)
      elif(args.annotationType=='xml'):
          saveXmlAnnotation(m.width, m.height,Annotation)
    
      deleteVideo() 
      print("삭제 완료")

      print()
    else: # 객체 클래스 추가 모드
      # clone darknet repos(다크넷 설치 : 차후 실행시 이미 설치된 폴더가 있으므로 재실행 x)
      os.chdir("/content/drive/MyDrive/yolov4_종설/Yolov4_model/")
      os.system("git clone https://github.com/AlexeyAB/darknet")

      # change makefile to have GPU and OPENCV enabled(gpu와 opencv 사용여부를 true로 바꿈)
      os.chdir("/content/drive/MyDrive/yolov4_종설/Yolov4_model/darknet")
      os.system("sed -i 's/OPENCV=0/OPENCV=1/' Makefile")
      os.system("sed -i 's/GPU=0/GPU=1/' Makefile")
      os.system("sed -i 's/CUDNN=0/CUDNN=1/' Makefile")
      os.system("sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile")

      # verify CUDA(쿠다 확인)
      os.system("/usr/local/cuda/bin/nvcc --version")

      # MakeFile
      os.system("make -j4")

      classes = args.className.split(" ")
      num = len(classes)

      f = open("/content/drive/MyDrive/yolov4_종설/Yolov4_model/custom/classes.txt", 'w')
      for i in range(len(classes)):
          data = classes[i] + "\n"
          f.write(data)
      f.close()

      current_path = os.path.abspath(os.curdir)
      COLAB_DARKNET_ESCAPE_PATH = '/content/drive/MyDrive/yolov4_종설/Yolov4_model' #zip file url
      COLAB_DARKNET_PATH = '/content/drive/MyDrive/yolov4_종설/Yolov4_model'

      os.chdir("/content/drive/MyDrive/yolov4_종설/Yolov4_model/custom")
      os.system("mkdir Person/")
      os.system("unzip /content/drive/MyDrive/Person_90_1128.zip -d ./Person/")


      YOLO_IMAGE_PATH = '/content/drive/MyDrive/yolov4_종설/Yolov4_model/custom/Person'
      YOLO_FORMAT_PATH = '/content/drive/MyDrive/yolov4_종설/Yolov4_model' + '/custom'

      class_count = 0
      test_percentage = 0.2
      paths = []

      with open(YOLO_FORMAT_PATH + '/' + 'classes.names', 'w') as names, \
          open(YOLO_FORMAT_PATH + '/' + 'classes.txt', 'r') as txt:
          for line in txt:
              names.write(line)  
              class_count += 1
          print ("[classes.names] is created")

      with open(YOLO_FORMAT_PATH + '/' + 'custom_data.data', 'w') as data:
          data.write('classes = ' + str(class_count) + '\n')
          data.write('train = ' + COLAB_DARKNET_ESCAPE_PATH + '/custom/' + 'train.txt' + '\n')
          data.write('valid = ' + COLAB_DARKNET_ESCAPE_PATH + '/custom/' + 'test.txt' + '\n')
          data.write('names = ' + COLAB_DARKNET_ESCAPE_PATH + '/custom/' + 'classes.names' + '\n')
          data.write('backup = /content/drive/MyDrive/yolov4_종설/Yolov4_model/custom')
          print ("[custom_data.data] is created")

      os.chdir(YOLO_IMAGE_PATH)
      for current_dir, dirs, files in os.walk('.'):
          for f in files:
              if f.endswith('.jpg'):
                  image_path = '/content/drive/MyDrive/yolov4_종설/Yolov4_model/custom/Person/' + f
                  paths.append(image_path + '\n')

      paths_test = paths[:int(len(paths) * test_percentage)]
      paths = paths[int(len(paths) * test_percentage):]

      with open(YOLO_FORMAT_PATH + '/' + 'train.txt', 'w') as train_txt:
          for path in paths:
              train_txt.write(path)
          print ("[train.txt] is created")

      with open(YOLO_FORMAT_PATH + '/' + 'test.txt', 'w') as test_txt:
          for path in paths_test:
              test_txt.write(path)
          print ("[test.txt] is created")

      #------------------------------------------------------------------------

      f = open("/content/drive/MyDrive/yolov4_종설/Yolov4_model/cfg/yolov3.cfg", 'r')
      f2 = open("/content/drive/MyDrive/yolov4_종설/Yolov4_model/custom/yolov3.txt", 'w')

      while True: 
        line = f.readline()
        if "checkA" in line:
          line = line.replace("checkA", str(num * 2000),1)
        if "checkB" in line:
          line = line.replace("checkB", str(int(num * 2000 * 0.8)), 1)
        if "checkC" in line:
          line = line.replace("checkC", str(int(num * 2000 * 0.9)), 1)
        if "checkD" in line:
          line = line.replace("checkD", str((num + 5) * 3), 1)
        if "checkE" in line:
          line = line.replace("checkE", str(num), 1)
        if not line: 
          break
        f2.write(line)
      f.close()
      f2.close()

      os.system("mv /content/drive/MyDrive/yolov4_종설/Yolov4_model/custom/yolov3.txt /content/drive/MyDrive/yolov4_종설/Yolov4_model/custom/yolov3.cfg")

      # wget명령어를 사용하여  darknet53.conv.74 다운
      os.system("wget -P /content/drive/MyDrive/yolov4_종설/Yolov4_model/custom/ https://pjreddie.com/media/files/darknet53.conv.74")

      os.chdir("/content/drive/MyDrive/yolov4_종설/Yolov4_model/darknet")
      path = "/content/drive/MyDrive/yolov4_종설/Yolov4_model/custom/"

      # 학습 진행
      os.system("./darknet detector train {}/custom_data.data {}/yolov3.cfg {}/darknet53.conv.74 -dont_show".format(path, path, path))
      
      args.cfgfile = "/content/drive/MyDrive/yolov4_종설/Yolov4_model/custom/yolov3.cfg"
      args.weightfile = "/content/drive/MyDrive/yolov4_종설/Yolov4_model/custom/yolov3_final.weights"

      # 유튜브로 다운로드 받은 영상을 자동으로 video 확장자에 맞게 저장함
      # 동영상 url 링크 로딩 ----------------------
      link = args.urlLink
      os.system("youtube-dl -o \"video.%(ext)s\" {}".format(link))

      # 파일 확장자 추출 메소드 호출 후
      extractExtension()

      # 모델 네트워트 로딩 -----------------------
      m = Darknet(args.cfgfile)
      m.load_weights(args.weightfile)
      print('Loading weights from %s... Done!' % (args.weightfile))
      if use_cuda:
          m.cuda()
      # ------------------------------------------

      # 동영상을 이미지로 저장할 경로와 결과 이미지를 저장할 경로 
      # weight 파일이 위치한 곳에 없을때의 예외처리들
      testpath = "./testdata"
      resultpath = "./resultdata"
      weightsFilePath = str(args.weightfile)

      if(os.path.isdir(testpath)):
          shutil.rmtree(testpath)
      if(os.path.isdir(resultpath)):
          shutil.rmtree(resultpath)
      
      os.mkdir(testpath)
      os.mkdir(resultpath)
      # --------------------------------------------------------------
      if(not os.path.isfile(weightsFilePath)):
          os.system("wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights")
          args.weightfile = "./yolov4.weights"
      # -----------------------------------------------------------------

      info(args.videofile)
      timeInfo = args.endTime
      files = makeImage(timeInfo)

      for i in range (0, len(files)):
          imagesPath = "./testdata/"+files[i]
          print(files[i]+"를 학습데이터로 전환합니다.")
          detect_cv2(args.annotationType,args.labelName,imagesPath,m)

      saveImage()
      #saveAnnotation()
      if (args.annotationType=='txt'):
          saveTxtAnnotation(Annotation)
      elif(args.annotationType=='xml'):
          saveXmlAnnotation(m.width, m.height,Annotation)
    
      deleteVideo() 
      print("삭제 완료")

      print()

      


          