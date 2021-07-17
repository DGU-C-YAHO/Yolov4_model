import sys
import os
import time
import math
import numpy as np

import itertools
import struct  # get_image_size
import imghdr  # get_image_size


#<annotation>
#    <filename>000001</filename>
#    <source>
#        <database>Unknown</database>
#    </source>
#    <size>
#        <width>353</width>
#        <height>500</height>
#        <depth>3</depth>
#    </size>
#    <segmented>0</segmented>
#    <object>
#        <name>M_50s</name>
#        <pose>Unspecified</pose>
#        <truncated>0</truncated>
#        <difficult>0</difficult>
#        <bndbox>
#            <xmin>129</xmin>
#            <ymin>31</ymin>
#            <xmax>298</xmax>
#            <ymax>227</ymax>
#        </bndbox>
#    </object>
#</annotation>



def makeTXT(cls_id,box):
    annotation = []
    annotation.append(cls_id)
    annotation.append((box[0] + box[2]) / 2) # center X 
    annotation.append((box[1] + box[3]) / 2) # center Y
    annotation.append(box[2] - box[0]) # weight
    annotation.append(box[3] - box[1]) # height
    return annotation

def makeXML(labelName,box,class_names=None):
    annotation = []
    annotation.append(class_names) #<name>
    annotation.append(box[0]) #<xmin>
    annotation.append(box[1]) #<ymin>
    annotation.append(box[2]) #<xmax>
    annotation.append(box[3]) #<ymax>
    return annotation

def saveTxtAnnotation(Annotation):
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

def saveXmlAnnotation(width,height,Annotation):
    for i in range(len(Annotation)):
        output = "./resultdata/test-%d.xml" % (i+1)
        filename = "test-%d" % (i+1)
        print("savepath"+output)
        f = open(output, 'w')
        data =  '<annotation>\n'
        data += '    <filename>{}</filename>\n'.format(filename)
        data += '    <source>\n        <database>Unknown</database>\n    </source>\n    <size>\n' 
        data += '        <width>{}</width>\n'.format(width)
        data += '        <height>{}</height>\n'.format(height)
        data += '        <depth>3</depth>\n    </size>\n    <segmented>0</segmented>\n    <object>\n'
        data += '        <name>{}</name>\n'.format(Annotation[i][0])
        data += '        <pose>Unspecified</pose>\n        <truncated>0</truncated>\n        <difficult>0</difficult>\n        <bndbox>\n'
        data += '            <xmin>{}</xmin>\n'.format(Annotation[i][1]*width)
        data += '            <ymin>{}</ymin>\n'.format(Annotation[i][2]*height)
        data += '            <xmax>{}</xmax>\n'.format(Annotation[i][3]*width)
        data += '            <ymax>{}</ymax>\n        </bndbox>\n    </object>\n</annotation>'.format(Annotation[i][4]*height)
        f.write(data)
        f.close()


