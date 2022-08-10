# -*- coding: utf-8 -*-
# @Author: LlHai
# @Date:   2022-03-07 10:13:53
# @Last Modified by:   LlHai
# @Last Modified time: 2022-05-25 10:27:29
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


def getAngleOfVector(pC, p1, p2):
    norm1 = p1 - pC
    # print(norm1)
    norm2 = p2 - pC
    # print(norm2)
    # print(norm1.dot(norm2))
    # print(norm1.dot(norm1))
    # print(norm2.dot(norm2))
    # print(np.sqrt(norm1.dot(norm1)) * np.sqrt(norm2.dot(norm2)))
    angle_hu = np.arccos(norm1.dot(norm2) / (np.sqrt(norm1.dot(norm1)) * np.sqrt(norm2.dot(norm2))))
    # print(angle_hu)
    
    angle_jiao = angle_hu *180 / np.pi
    return angle_jiao

def getAllAngleS(points):
    angle_dict = {}
    
    # Angle Arm left
    pc = points[6]
    p1 = points[10]
    p2 = pc + [0, 100]
    angle = getAngleOfVector(pc, p1, p2)
    angle_dict['arm_left'] = angle
    
    # Angle Arm right
    pc = points[5]
    p1 = points[9]
    p2 = pc + [0, 100]
    angle = getAngleOfVector(pc, p1, p2)
    angle_dict['arm_right'] = angle

    # Angle leg left
    pc = points[12]
    p1 = points[16]
    p2 = pc + [0, 100]
    angle = getAngleOfVector(pc, p1, p2)
    angle_dict['leg_left'] = angle

    # Angle leg right
    pc = points[11]
    p1 = points[13]
    p2 = pc + [0, 100]
    angle = getAngleOfVector(pc, p1, p2)
    angle_dict['leg_right'] = angle

    # Angle knee left
    pc = points[14]
    p1 = points[12]
    p2 = points[16]
    angle = getAngleOfVector(pc, p1, p2)
    angle_dict['knee_left'] = angle

    # Angle knee right
    pc = points[13]
    p1 = points[11]
    p2 = points[15]
    angle = getAngleOfVector(pc, p1, p2)
    angle_dict['knee_right'] = angle
    
    # Angle elbow left
    pc = points[8]
    p1 = points[6]
    p2 = points[10]
    angle = getAngleOfVector(pc, p1, p2)
    angle_dict['elbow_left'] = angle

    # Angle elblw right
    pc = points[7]
    p1 = points[5]
    p2 = points[9]
    angle = getAngleOfVector(pc, p1, p2)
    angle_dict['elbow_right'] = angle

    # Angle body
    pc = (points[11] + points[12]) / 2
    p1 = (points[5] + points[6]) / 2
    p2 = p1 - [0, 10]
    angle = getAngleOfVector(pc, p1, p2)
    angle_dict['angle_body'] = angle

    # Upper Arm Left
    pc = points[6]
    p1 = points[8]
    p2 = pc + [0, 100]
    angle = getAngleOfVector(pc, p1, p2)
    angle_dict['upper_arm_left'] = angle

    # Upper Arm Right
    pc = points[5]
    p1 = points[7]
    p2 = pc + [0, 100]
    angle = getAngleOfVector(pc, p1, p2)
    angle_dict['upper_arm_right'] = angle

    return angle_dict

def dictToString(dict):
    res_str = ""
    for key,value in dict.items():
        res_str += '{} : {:.2f} \n'.format(key, value)
    
    return res_str

def showTextOnFrame(frame, str):
    y0, dy = 50, 30
    for i, line in enumerate(str.split('\n')):
        y = y0 + i*dy
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
    
    return frame
    

def main():
    # 创建plt画布
    # plt.ion()
    # fig = plt.figure(figsize=(12,6))

    #ac = ActionRecognition('rules/actions_define_shuang_shou_nao_hou.json')
    #ac = ActionRecognition('rules/actions_define_jiao_chang_shen.json')
    ac = ActionRecognition('rules/actions_define_bai_yao.json')
    action_nums = ac.getActionNum()
    # 加载HrNet模型
    interface2D,bboxModel,poseModel = loadHrnetModel()

    # 初始化相机参数
    # cap = cv2.VideoCapture("videos/hospital2/shuangshou_naohou.mp4")
    cap = cv2.VideoCapture(0)
    cap.set(3, 2560)
    cap.set(4, 720)
    frame_width = cap.get(3)
    frame_height = cap.get(4)
    frame_rate = cap.get(5)
    frame_cnt = cap.get(7)
    print('frame width : {}, frame height : {}'.format(frame_width, frame_height))
    print('frame rate : {}, frame cnt : {}'.format(frame_rate, frame_cnt))

    ac.setFrameRate(frame_rate)
    
    frame_id = 0
    max_frame_id = 300
    
    fourCC = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter("videos/debug.mp4", fourCC, 30, (1280, 720))
    while True:
        _, frame = cap.read()
        frame_id += 1
        # frame, W, H = resize_img(frame)
        h, w = frame.shape[:2]
        w = w/2
        # frame = frame[0:int(h), 0:int(w)]
        
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
        # print(joint2D)
        angle_dict = getAllAngleS(joint2D)
        angle_str = dictToString(angle_dict)
        frame = showTextOnFrame(frame, angle_str)

        ac_res, ac_scores = ac.recognize(joint2D, frame_id)
        print('[Action Recognition Res] : res : {}, scores : {}'.format(ac_res, ac_scores))
        info_str = 'actionRecognition progress : {} / {}'.format(len(ac_scores), action_nums)
        score_str = 'actionRecognition score : {}'.format(ac_scores)
        cv2.putText(frame, info_str, (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        if ac_res:
            cv2.putText(frame, 'Action Recognition Finish, score is {}'.format(np.mean(ac_scores)), (0, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, score_str, (400, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        
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