# coding:utf-8
import face_recognition
import os, sys
import numpy as np
import cv2


path = os.getcwd() + '\\databases\\'
know_face_encodings = []
know_faces = []
cap = cv2.VideoCapture(0)

#遍历所有带名字文件夹的路径
for file_name in os.listdir(path):
    #遍历文件夹里所有的图片，并对图片进行编码
    for image_file in os.listdir(path + file_name):
        img_path = path + file_name + '\\' + image_file #要编码的图片
        img = face_recognition.load_image_file(img_path) #将图片加载到face_recognition库中
        encoding = face_recognition.face_encodings(img)[0] #对图像进行编码，默认只有一张人脸，只有一张人脸图时输出encoding都一样，但是不加[0]会出错？？
        #对一个文件夹中的所有图片编码，并命名
        know_face_encodings.append(encoding) 
        name = file_name
        know_faces.append(name)
        

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ok, frame = cap.read() #读取一帧数据
    if not ok: break
    #缩小帧,说是可以提高速度？用了以后识别框不在脸上？？？感觉怪怪的
    #small_frame = cv2.resize(frame,(0,0),fx = 0.25, fy = 0.25 )
    #opencv使用BGR获取图像，face_recognition需要RGB。
    #格式转换，但是不使用也能运行？
    #rgb_frame = frame[:,:,::-1]
    
    face_locations = face_recognition.face_locations(frame) #获取图片中所有的人脸

    unknow_image_encoding = face_recognition.face_encodings(frame , face_locations) #将所有的人脸进行编码处理

    #遍历所有人脸的编码，并将每个人脸和编码一一对应
    for i in range(len(unknow_image_encoding)): 
        unknow_encoding = unknow_image_encoding[i]
        face_location = face_locations[i]
        #获取每个人脸的位置并将其框出来
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom),(0,255,0),2)

        #对比要识别的人脸的编码和已知人脸的编码，并返回结果
        results = face_recognition.compare_faces(know_face_encodings,unknow_encoding)
        
        #results是一个True和False的列表，true表示和已知人脸一致
        if True in results:
            the_name_index = results.index(True)
            the_name = know_faces[the_name_index]
        else: 
            the_name = 'unknow'
        '''
        #和已知人脸对比得到一个欧式距离，这个距离越小，越相似，最佳值为0.6
        face_distances = face_recognition.face_distance(know_face_encodings, unknow_encoding)
        best_result_index = np.argmin(face_distances)#获取列表最小值的下标

        if results[best_result_index]: 
            the_name = know_faces[best_result_index]
        '''

        cv2.rectangle(frame,(left, bottom - 35),(right,bottom),(0,0, 255),cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, the_name, (left + 6,bottom - 6), font, 1.0, (255, 255, 255),1)

    
    cv2.imshow('face',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
 

cap.release()
cv2.destroyAllWindows()


