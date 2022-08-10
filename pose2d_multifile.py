# -*- coding: utf-8 -*-
# @Author: LlHai
# @Date:   2022-03-07 10:13:53
# @Last Modified by:   LlHai
# @Last Modified time: 2022-06-07 23:32:33
from tkinter import Frame
import cv2
import numpy as np
import math
import time
# import pyzed.sl as sl
import matplotlib.pyplot as plt
import torch

from tools.utils import draw_3Dimg, videoInfo, resize_img, draw_2Dimg
from action_recognition import ActionRecognition
from action_manager import ActionManager
from PIL import Image, ImageDraw, ImageFont
torch.backends.cudnn.enabled = False

def loadHrnetModel():
    from joints_detectors.hrnet.pose_estimation.video import getTwoModel, getKptsFromImage
    bboxModel, poseModel = getTwoModel()
    return getKptsFromImage,bboxModel,poseModel

W = 1280
H = 720

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

def getMainJoint(joint2D_with_score_list):
    max_score = 0
    center = W / 2.0
    main_joint = []
    for joint2D_with_score in joint2D_with_score_list:
        joint = joint2D_with_score[:,:2]
        joint_x = joint2D_with_score[:,0]
        joint_score = joint2D_with_score[:,2]
        mean_x = np.mean(joint_x)
        mean_score = np.mean(joint_score)
        cur_score = mean_score * 0.5 + (center - abs(mean_x-center)) / center * 0.5

        if(cur_score > max_score):
            max_score = cur_score
            main_joint = joint
    
    return main_joint, max_score

def getScoreLevel(score):
    if (score >= 80):
        return 'B+'
    elif (score >= 60):
        return 'B'
    else:
        return 'B-' 

def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  
    # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def main():
    
    # 加载HrNet模型
    interface2D,bboxModel,poseModel = loadHrnetModel()

    # 初始化相机参数
    #cap = cv2.VideoCapture("videos/13.mp4")
    cap2 = cv2.VideoCapture("videos/000_.mp4")
    cap = cv2.VideoCapture(0)
    cap.set(3, 2*W)
    cap.set(4, H)
    frame_width = cap.get(3)
    frame_height = cap.get(4)
    frame_rate = cap.get(5)
    frame_cnt = cap.get(7)
    print('frame width : {}, frame height : {}'.format(frame_width, frame_height))
    print('frame rate : {}, frame cnt : {}'.format(frame_rate, frame_cnt))
    
    # 输出文件定义
    fourCC = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter("videos/debug.mp4", fourCC, 30, (W, H))

    # ActionManager
    action_manager = ActionManager('rules/config.json')
    file_nums = action_manager.getFileNums()

    # socre
    score = 0
    i = 0
    score_list = []
    while True:
        ac, score_weight = action_manager.getNextActionRecognition()
        action_nums = ac.getActionNum()
        action_name = ac.getActionName()
        ac.reset()
        ac.setFrameRate(frame_rate)
        frame_id = 0
        count_down_nums = 30

        # print('video path' + video_path)
        #cap2 = cv2.VideoCapture(video_path)
        while True:
            _, frame = cap.read()
            _example, frame_example = cap2.read()
            frame_id += 1
            if _example == False:
                cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
                _example, frame_example = cap2.read()
            h, w = frame.shape[:2]
            frame = frame[0:int(h), 0:int(w)]

            # 显示信息到界面
            # 总进度
            if i < file_nums:
                info_str = 'total progress : {} / {}'.format(i+1, file_nums)
            else:
                info_str = 'ActionRecognition Finish!!  Level: {}   Action 1: {}  Action 2: {}  Action 3: {}'.format(getScoreLevel(score),score_list[0],score_list[1],score_list[2])
                if score_list[0] < 60:
                    score_str1 = "动作一完成度不够，存在肩关节活动障碍,常见于肩周炎\劲椎病\颈肩综合征\中风后遗症等"
                    frame=cv2AddChineseText(frame,score_str1, (123, 123),(0, 255, 0), 30)
                    cv2.imshow('test',frame)
                if score_list[1] < 60:
                    score_str2 = "动作二完成度不够,可见于关节退行性病变\心脑血管疾病\低血糖\低血压"
                    frame=cv2AddChineseText(frame,score_str2, (150, 150),(0, 255, 0), 30)
                    cv2.imshow('test',frame)
                if score_list[2] <60:
                    score_str3 = "动作三完成度不够,可见于关节退行性病变\脑血管病变\美尼尔综合征等"
                    frame=cv2AddChineseText(frame,score_str3, (180, 180),(0, 255, 0), 30)
                    cv2.imshow('test',frame)
            cv2.putText(frame, info_str, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            info_str = 'current action : {}'.format(action_name)
            cv2.putText(frame, info_str, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            # 总分
            info_str = 'Total scores : {}'.format(score)
            cv2.putText(frame, info_str, (50, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        

            # 检测人体骨架
            joint2D_with_score = interface2D(bboxModel, poseModel, frame)

            if frame_id < count_down_nums:
                cv2.putText(frame, 'count down : {}'.format(5 - int(frame_id / (count_down_nums/5))), (400, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.imshow("frame_example",frame_example)
                cv2.imshow("test",frame)
                cv2.waitKey(5)
                continue
            
            # 检测人体骨架
            joint2D_with_score = interface2D(bboxModel, poseModel, frame)
            # print(type(joint2D_with_score))
            if len(joint2D_with_score) == 0:
                cv2.imshow("frame_example",frame_example)
                cv2.imshow("test",frame)
                out.write(frame)
                cv2.waitKey(5)
                continue

            # 根据骨架位置与评分，得出评分最高骨架
            joint2D, joint2D_score = getMainJoint(joint2D_with_score)
            
            # 动作识别
            ac_res, ac_scores = ac.recognize(joint2D, frame_id)

            # 显示当前动作识别进度
            info_str = 'actionRecognition progress : {} / {}'.format(len(ac_scores), action_nums)
            cv2.putText(frame, info_str, (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            
            # 显示图像
            frame = draw_2Dimg(frame, joint2D)
            cv2.imshow("frame_example",frame_example)
            cv2.imshow("test",frame)
            out.write(frame)
            cv2.waitKey(5)

            if ac_res:
                i += 1
                if i <= file_nums:
                    score += np.mean(ac_scores) * score_weight
                    score_list.append(float(np.mean(ac_scores)))
                if i < file_nums :
                    break
                
            
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