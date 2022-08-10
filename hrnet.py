# -*- coding: utf-8 -*-
# @Author: LlHai
# @Date:   2022-03-07 10:13:53
# @Last Modified by:   LlHai
# @Last Modified time: 2022-05-17 19:14:13
import cv2
import numpy as np
import math
import time
# import pyzed.sl as sl
import matplotlib.pyplot as plt
import torch

from tools.utils import draw_3Dimg, videoInfo, resize_img, draw_2Dimg
from action_recognition import ActionRecognition
torch.backends.cudnn.enabled = False

def loadHrnetModel():
    from joints_detectors.hrnet.pose_estimation.video import getTwoModel, getKptsFromImage
    bboxModel, poseModel = getTwoModel()
    return getKptsFromImage,bboxModel,poseModel

    

def main():
    # 创建plt画布
    # plt.ion()
    # fig = plt.figure(figsize=(12,6))


    # 加载HrNet模型
    interface2D,bboxModel,poseModel = loadHrnetModel()

    # 初始化相机参数
    cap = cv2.VideoCapture("videos/hospital2/jiao_chang_shen.mp4")
    # cap = cv2.VideoCapture(0)
    cap.set(3, 2560)
    cap.set(4, 720)
    frame_width = cap.get(3)
    frame_height = cap.get(4)
    frame_rate = cap.get(5)
    frame_cnt = cap.get(7)
    print('frame width : {}, frame height : {}'.format(frame_width, frame_height))
    print('frame rate : {}, frame cnt : {}'.format(frame_rate, frame_cnt))

    frame_id = 0
    
    # 倒计时
    max_frame_id = 300

    fourCC = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter("videos/debug.mp4", fourCC, 15, (1280, 720))
    while True:
        _, frame = cap.read()
        frame_id += 1
        # frame, W, H = resize_img(frame)
        h, w = frame.shape[:2]
        w = w/2
        frame = frame[0:int(h), 0:int(w)]

        if frame_id < max_frame_id:
            cv2.putText(frame, 'count down : {}'.format(5 - int(frame_id / 60)), (400, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.imshow("test",frame)
            cv2.waitKey(5)
            continue
        
        # frame = cv2.imread("videos/1.jpg")
        # cv2.imshow("test",frame)
        # cv2.waitKey(5)
        # print(type(frame))
        joint2D_with_score = interface2D(bboxModel, poseModel, frame)
        # print(type(joint2D_with_score))
        if len(joint2D_with_score) == 0:
            cv2.imshow("test",frame)
            out.write(frame)
            cv2.waitKey(5)
            continue
        joint2D = joint2D_with_score[:,:2]
       
        frame = draw_2Dimg(frame, joint2D)
        cv2.imshow("test",frame)
        out.write(frame)
        cv2.waitKey(5)
    
    out.release()
        


if __name__ == '__main__':
    #args = parserInit()
    # joint3D = []
    # joint3D.append(( 595, 315, 1759.59682168))
    # joint3D.append(( 595, 405, 1664.15672024))
    # joint3D.append(( 580, 566, 1729.3586998))
    
    # joint3D = np.array(joint3D)
    # print(joint3D)
    # print('angle: ', getAngleOfVector(joint3D[1], joint3D[0], joint3D[2]))
    main()