import numpy as np
import pandas as pd
import cv2
import glob
from matplotlib import pyplot as plt
from IoU import bb_intersection_over_union
face_cascade = cv2.CascadeClassifier('cascade_new_c013579.xml')


i=0
for filename in glob.glob('TestImages\TestSet\*.jpg'):
    img = cv2.imread(filename)
    #image_list = image_list.append(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=5, minNeighbors=1,
                                             minSize=(70, 70))
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(faces[0][0],faces[0][1]),(faces[0][0]+faces[0][2],faces[0][1]+faces[0][3]),(255,0,0),2)
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    i = i + 1
    #cv2.namedWindow(str(i))
    cv2.imshow(str(i),img)
    cv2.waitKey()
    
     
    
#cv2.destroyAllWindows()

iou = bb_intersection_over_union(gt,faces)    