# -*- coding: utf-8 -*-
# @Author: LlHai
# @Date:   2022-05-25 10:37:50
# @Last Modified by:   LlHai
# @Last Modified time: 2022-05-27 21:40:34

import json
from action_recognition import ActionRecognition

class ActionManager():

    idx = -1
    file_nums = 0
    file_list = []
    action_recognition_list = []
    score_weight_list = []
    #video_path_list = []

    def __init__(self, file_path):
        self.parseFile(file_path)
        return
    
    def parseFile(self, file_path):
        json_file = open(file_path, 'r')
        json_str = json_file.read()
        # print(json_str)

        json_obj = json.loads(json_str)

        files = json_obj['files']
        for file in files:
            file_name = file['file_name']
            score_weight = file['score_weight']
            #video_path = file['video_path']
            self.action_recognition_list.append(ActionRecognition(file_name))
            self.score_weight_list.append(float(score_weight))
            #self.video_path_list.append(video_path)
        
        self.file_nums = len(self.action_recognition_list)
    
    def getNextActionRecognition(self):
        self.idx = self.idx + 1
        return self.action_recognition_list[self.idx], self.score_weight_list[self.idx]
    
    def getFileNums(self):
        return self.file_nums
    

        

