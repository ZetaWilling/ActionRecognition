# -*- coding: utf-8 -*-
# @Author: LlHai
# @Date:   2019-11-28 16:17:06
# @Last Modified by:   LlHai
# @Last Modified time: 2022-05-25 12:43:00
'''
使用yolov3作为pose net模型的前处理
use yolov3 as the 2d human bbox detector
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
path1 = os.path.split(os.path.realpath(__file__))[0]
path2 = os.path.join(path1, '..')
sys.path.insert(0, path1)
sys.path.insert(0, path2)
import ipdb;pdb=ipdb.set_trace
import numpy as np

import time

import torch

import _init_paths
from config import cfg
import config
from config import update_config

import torchvision.transforms as transforms
from utils.transforms import *
from lib.core.inference import get_final_preds
import cv2
import models
from lib.detector.yolo.human_detector import main as yolo_det
from scipy.signal import savgol_filter
from lib.detector.yolo.human_detector import load_model as yolo_model
sys.path.pop(0)
sys.path.pop(1)
sys.path.pop(2)



kpt_queue = []
def smooth_filter(kpts):
    if len(kpt_queue) < 6:
        kpt_queue.append(kpts)
        return kpts

    queue_length = len(kpt_queue)
    if queue_length == 50:
        kpt_queue.pop(0)
    kpt_queue.append(kpts)

    # transpose to shape (17, 2, num, 50) 关节点keypoints num、横纵坐标、每帧人数、帧数
    transKpts = np.array(kpt_queue).transpose(1,2,3,0)

    window_length = queue_length - 1 if queue_length % 2 == 0 else queue_length - 2
    # array, window_length(bigger is better), polyorder
    result = savgol_filter(transKpts, window_length, 3).transpose(3, 0, 1, 2) #shape(frame_num, human_num, 17, 2)

    # 返回倒数第几帧 return third from last frame
    return result[-3]


class get_args():
    # hrnet config
    cfg = path2 + '/experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml'
    dataDir=''
    logDir=''
    modelDir=''
    opts=[]
    prevModelDir=''

def _box2cs(box, image_width, image_height):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h, image_width, image_height)

def _xywh2cs(x, y, w, h, image_width, image_height):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    aspect_ratio = image_width * 1.0 / image_height
    pixel_std = 200

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale

def PreProcess(image, bboxs, scores, cfg, thred_score=0.8):

    if type(image) == str:
        data_numpy = cv2.imread(image, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    else:
        data_numpy = image

    inputs = []
    centers = []
    scales = []

    score_num = np.sum(scores>thred_score)
    max_box = min(5, score_num)
    for bbox in bboxs[:max_box]:
        x1,y1,x2,y2 = bbox
        box = [x1, y1, x2-x1, y2-y1]

        # 截取 box fron image  --> return center, scale
        c, s = _box2cs(box, data_numpy.shape[0], data_numpy.shape[1])
        centers.append(c)
        scales.append(s)
        r = 0

        trans = get_affine_transform(c, s, r, cfg.MODEL.IMAGE_SIZE)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        input = transform(input).unsqueeze(0)
        inputs.append(input)
    

    if(not inputs):
        print('The human body is not detected')
    else:
        inputs = torch.cat(inputs)
    return inputs, data_numpy, centers, scales

##### load model
def model_load(config):
    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, is_train=False
    )
    model_file_name  = path2 + '/models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth'
    state_dict = torch.load(model_file_name)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k # remove module.
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    return model



def ckpt_time(t0=None, display=None):
    if not t0:
        return time.time()
    else:
        t1 = time.time()
        if display:
            print('consume {:2f} second'.format(t1-t0))
        return t1-t0, t1


###### LOAD human detecotor model
args = get_args()
update_config(cfg, args)

def generate_kpts(video_name, smooth=None):
    human_model = yolo_model()
    args = get_args()
    update_config(cfg, args)
    cam = cv2.VideoCapture(video_name)
    video_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    ret_val, input_image = cam.read()
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    input_fps = cam.get(cv2.CAP_PROP_FPS)

    #### load pose-hrnet MODEL
    pose_model = model_load(cfg)
    pose_model.cuda()

    # collect keypoints coordinate
    kpts_result = []
    for i in tqdm(range(video_length-1)):

        ret_val, input_image = cam.read()

        try:
            bboxs, scores = yolo_det(input_image, human_model)
            # bbox is coordinate location
            inputs, origin_img, center, scale = PreProcess(input_image, bboxs, scores, cfg)
        except Exception as e:
            print(e)
            continue

        with torch.no_grad():
            # compute output heatmap
            inputs = inputs[:,[2,1,0]]
            output = pose_model(inputs.cuda())
            # compute coordinate
            preds, maxvals = get_final_preds(
                cfg, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))

        if smooth:
            # smooth and fine-tune coordinates
            preds = smooth_filter(preds)

        # 3D video pose (only support single human)
        kpts_result.append(preds[0])

    result = np.array(kpts_result)
    return result

def getTwoModel():
    #  args = get_args()
    #  update_config(cfg, args)
    #### load pose-hrnet MODEL
    pose_model = model_load(cfg)
    pose_model.cuda()

    # load YoloV3 Model
    bbox_model = yolo_model()
    bbox_model.cuda()

    return bbox_model, pose_model


def getKptsFromImage(human_model, pose_model, image, smooth=None):
    #t0 = time.time()
    #print("-----------bboxs-------------")
    bboxs, scores = yolo_det(image, human_model)
    # print(bboxs)
    #print("-----------score-----------")
    # print(scores)
    # t1 = time.time()
    # print("box location consume %f"%(t1-t0))
    # bbox is coordinate location
    inputs, origin_img, center, scale = PreProcess(image, bboxs, scores, cfg)
    if not torch.is_tensor(inputs):
        return []
    #print("----------inputs--------------")
    #print(inputs)
    #cv2.imshow("test",origin_img)
    #cv2.waitKey()
    # t2 = time.time()
    # print("PreProcess consume %f"%(t2-t1))
    with torch.no_grad():
        # compute output heatmap
        inputs = inputs[:,[2,1,0]]
        output = pose_model(inputs.cuda())
        # compute coordinate
        preds, maxvals = get_final_preds(
            cfg, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))
            #cfg, output.clone().numpy(), np.asarray(center), np.asarray(scale))
    
    # t3 = time.time()
    # print("compute heatmap consume %f"%(t3-t2))
    # 3D video pose (only support single human)
    #print("-------------result-------------")
    result = []
    for i in range(preds.shape[0]):
        result.append(np.concatenate((preds[i], maxvals[i]), 1))
    #print(result)
    # t4 = time.time()
    # print("concatenate consume %f"%(t4-t3))
    return result

if __name__ == '__main__':
    main()
