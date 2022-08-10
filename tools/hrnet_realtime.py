import os
import cv2
import numpy as np
from argparse import ArgumentParser
import ipdb;pdb=ipdb.set_trace
import time
import matplotlib.pyplot as plt


from joints_detectors.hrnet.pose_estimation.video import getTwoModel, getKptsFromImage
bboxModel, poseModel = getTwoModel()
interface2D = getKptsFromImage
from tools.utils import videopose_model_load as Model3Dload
model3D = Model3Dload()
from tools.utils import interface as VideoPoseInterface
interface3D = VideoPoseInterface
from tools.utils import draw_3Dimg, draw_2Dimg, videoInfo, resize_img


def main(VideoName):
    plt.ion()
    fig = plt.figure(figsize=(12,6))

    cap, cap_length = videoInfo(VideoName)
    #cap = cv2.VideoCapture(0)
    #cap_length = 300
    kpt2Ds = []
    for i in tqdm(range(cap_length)):
        t0 = time.time()

        _, frame = cap.read()
        frame, W, H = resize_img(frame)

        try:
            joint2D = interface2D(bboxModel, poseModel, frame)
            #print('2D comsume {:0.3f} s'.format(time.time() - t0))
        except Exception as e:
            print(e)
            continue

        if i == 0:
            for _ in range(30):
                kpt2Ds.append(joint2D)
        elif i < 30:
            kpt2Ds.append(joint2D)
            kpt2Ds.pop(0)
        else:
            kpt2Ds.append(joint2D)

        joint3D = interface3D(model3D, np.array(kpt2Ds), W, H)
        joint3D_item = joint3D[-1] #(17, 3)
        print('comsume {:0.3f} s'.format(time.time() - t0))
        #draw_3Dimg(plt,fig,joint3D_item, frame, display=1, kpt2D=joint2D)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-video", "--video_input", help="input video file name", default="videos/dance.mp4")
    args = parser.parse_args()
    VideoName = args.video_input
    print('Input Video Name is ', VideoName)
    main(VideoName)

