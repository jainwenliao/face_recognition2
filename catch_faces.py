# coding:utf-8
'''利用opencv调用摄像头捕获人脸'''

import cv2
import os, sys
from PIL import Image

def get_faces(window_name, camera, pic_num, name, save_path):
    if not os.path.exists(save_path): 
        os.makedirs(save_path)

    cv2.namedWindow(window_name)#窗口名字
    cap = cv2.VideoCapture(camera)#调用摄像头

    #opencv的人脸特诊分类器，分类器所在的位置
    classfier = cv2.CascadeClassifier(r'E:\\Opencv\\haarcascades\\haarcascade_frontalface_default.xml')

    num = 1
    while cap.isOpened():
        ret, frame = cap.read() #读取一帧的数据
        if not ret:
            break

        #后面使用face_recognition来识别人脸，需要RGB图像这里就不灰度化帧了
        #人脸识别

        face_rect =classfier.detectMultiScale(frame, scaleFactor = 1.2, minNeighbors = 3, minSize = (64, 64))
        
        #如果检测到人脸，将帧保存为图像，face_recognition 有提取人脸的模块，这里不对帧进行人脸提取

        if len(face_rect) > 0: 
            #将人脸框出来 
            
            image_name = name + str(num) 
            img_name = '%s/%s.jpg'%(save_path,image_name)
            img = cv2.resize(frame,(250,250))#将帧变为250*250大小
            cv2.imwrite(img_name, img)

            for (x, y, w, h) in face_rect: 
                cv2.rectangle(frame, (x - 10, y -10), (x + w +10, y + h + 10), (255, 0, 0), 2)

            num +=1
             #显示拍摄的图片数
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,
            'num: %d'%(num),
            (x + 30, y + 30),
            font, 
            2, 
            (0,255,0), 4)

               

        if num > pic_num: 
            break

        #显示图像
        cv2.imshow(window_name,frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):  
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    get_faces('face recognition', 0, 10, 'liaojianwen', 'E:\\face_recognization_projects\\face_recognition5\\liaojianwen')
    

