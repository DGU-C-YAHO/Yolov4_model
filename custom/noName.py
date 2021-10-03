#PYTHON
import os

print("라벨링할 클래스 개수를 입력하시오 : ")
num =  int(input())

print("라벨링할 클래스 이름을 입력하시오 : ")
classes = []

for i in range(num):
  classes.append(input())

f = open("/content/drive/MyDrive/yolov4_종설/Yolov4_model/custom/classes.txt", 'w')
for i in range(len(classes)):
    data = classes[i] + "\n"
    f.write(data)
f.close()

current_path = os.path.abspath(os.curdir)
COLAB_DARKNET_ESCAPE_PATH = '/content/drive/MyDrive/yolov4_종설/Yolov4_model' #zip file url
COLAB_DARKNET_PATH = '/content/drive/MyDrive/yolov4_종설/Yolov4_model'

os.system("mkdir Person/")
os.system("unzip /content/drive/MyDrive/Person_90_1128.zip -d ./Person/")


YOLO_IMAGE_PATH = '/content/drive/MyDrive/yolov4_종설/Yolov4_model/custom/Person'
YOLO_FORMAT_PATH = '/content/drive/MyDrive/yolov4_종설/Yolov4_model' + '/custom'

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
    data.write('backup = /content/drive/MyDrive/yolov4_종설/Yolov4_model/custom')
    print ("[custom_data.data] is created")

os.chdir(YOLO_IMAGE_PATH)
for current_dir, dirs, files in os.walk('.'):
    for f in files:
        if f.endswith('.jpg'):
            image_path = '/content/drive/MyDrive/yolov4_종설/Yolov4_model/custom/Person/' + f
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

f = open("/content/drive/MyDrive/yolov4_종설/Yolov4_model/cfg/yolov3.cfg", 'r')
f2 = open("/content/drive/MyDrive/yolov4_종설/Yolov4_model/custom/yolov3.txt", 'w')

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

os.system("mv /content/drive/MyDrive/yolov4_종설/Yolov4_model/custom/yolov3.txt /content/drive/MyDrive/yolov4_종설/Yolov4_model/custom/yolov3.cfg")

# wget명령어를 사용하여  darknet53.conv.74 다운
os.system("wget -P /content/drive/MyDrive/yolov4_종설/Yolov4_model/custom/ https://pjreddie.com/media/files/darknet53.conv.74")

os.chdir("/content/drive/MyDrive/yolov4_종설/Yolov4_model/darknet")
path = "/content/drive/MyDrive/yolov4_종설/Yolov4_model/custom/"

# 학습 진행
os.system("./darknet detector train {}/custom_data.data {}/yolov3.cfg {}/darknet53.conv.74 -dont_show".format(path, path, path))