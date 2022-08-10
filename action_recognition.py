# -*- coding: utf-8 -*-
# @Author: LlHai
# @Date:   2022-04-29 14:39:31
# @Last Modified by:   LlHai
# @Last Modified time: 2022-05-25 12:27:54
import json
import sys
import numpy as np
# HrNet关节定义
# 0. Nose
# 1. Right-Eye
# 2. left-Eye
# 3. right-Ear
# 4. left-Ear
# 5. Right-shoulder
# 6. left-shoulder
# 7. right-arm
# 8. left-arm
# 9. right-hand
# 10. left-hand
# 11. right-leg
# 12. left-leg
# 13. right-knee
# 14. left-knee
# 15. right-foot
# 16. left-foot

class ActionRecognition():
    
    idx = 0
    action_list = []
    action_scores = []
    action_name = ''
    action_nums = 0
    frame_rate = 1
    current_score = 0
    def __init__(self, file_path):
        self.parseFile(file_path)
        self.idx = 0
        self.action_scores.clear()
        return 
    
    def parseFile(self, file_path):
        json_file = open(file_path, 'r')
        json_str = json_file.read()
        # print(json_str)

        json_obj = json.loads(json_str)
        # print(json_obj['actions'][1]['id'])
        # print(type(json_obj['actions'][1]['id']))

        self.action_list = json_obj['actions']
        self.action_name = json_obj['name']
        self.action_nums = int(json_obj['actionNums'])
        return

    def setFrameRate(self, rate):
        self.frame_rate = rate
        return

    # 生成一个不在人体上的关节点，用于处理肢体和水平or竖直方向夹角的情况
    def getExtraPoint(self, points, id, pc):
        if id <= 16 :
            return points[id]
        
        if id == 17:
            return pc - [0, 100] 
        elif id == 18:
            return pc + [0, 100]
        elif id == 19:
            return pc - [100, 0]
        elif id == 20:
            return pc + [100, 0] 
        else:
            print('[getExtraPoint Error]: illegl joint id')

        return 

    def getAngleOfVector(self, pc, p1, p2):
        norm1 = p1 - pc
        # print(norm1)
        norm2 = p2 - pc
        # print(norm2)
        # print(norm1.dot(norm2))
        # print(norm1.dot(norm1))
        # print(norm2.dot(norm2))
        # print(np.sqrt(norm1.dot(norm1)) * np.sqrt(norm2.dot(norm2)))
        angle_hu = np.arccos(norm1.dot(norm2) / (np.sqrt(norm1.dot(norm1)) * np.sqrt(norm2.dot(norm2))))
        # print(angle_hu)
        
        angle_jiao = angle_hu *180 / np.pi
        return angle_jiao

    def recognize(self, points, frame_id):
        if self.idx >= self.action_nums:
            return True, self.action_scores
        
        cur_action = self.action_list[self.idx]

        # 获取下一个动作的起始时间
        next_action_frame_time = sys.maxsize
        if self.idx < self.action_nums - 1:
            next_action_frame_time = int(self.action_list[self.idx + 1]['frame_time']) * self.frame_rate
        
        # 超过动作最晚时间，切换到下一动作
        if frame_id >= next_action_frame_time:
            self.action_scores.append(0)
            self.idx += 1
            cur_action = self.action_list[self.idx]

        rules = cur_action['rules']

        # 根据规则计算分数
        cur_score = self.judgeAction(points, rules)
        if cur_score > 30:
            if cur_score > self.current_score:
                self.current_score = cur_score
            elif self.current_score - cur_score > 5:
                self.idx += 1
                self.action_scores.append(self.current_score)
                print('[ActionRecognition Success] : frame id ：{}, score : {}'.format(frame_id, self.current_score))
                self.current_score = 0
        
        if self.idx >= self.action_nums:
            print('[ActionRecognition Finish ] ')
            return True, self.action_scores
        
        return False, self.action_scores

        
    def judgeAction(self, points, rules):
        
        scores = []
        for rule in rules:
            # 读取action信息
            center_id = int(rule['centerId'])
            point1_id = int(rule['point1Id'])
            point2_id = int(rule['point2Id'])
            low_score_angle = int(rule['low_score_angle'])
            mid_score_angle = int(rule['mid_score_angle'])
            high_score_angle = int(rule['high_score_angle'])
            score_weight = float(rule['score_weight'])
    
            # 获取指定关节坐标
            point_center = points[center_id]
            point1 = points[point1_id]
            point2 = self.getExtraPoint(points, point2_id, point_center)
    
            # 计算角度
            angle = self.getAngleOfVector(point_center, point1, point2)

            cur_score = 0
            if high_score_angle >= mid_score_angle:
                if angle >= high_score_angle:
                    cur_score = 100
                elif angle >= mid_score_angle:
                    cur_score = 70
                elif angle >= low_score_angle:
                    cur_score = 40
            else:
                if angle <= high_score_angle:
                    cur_score = 100
                elif angle <= mid_score_angle:
                    cur_score = 70
                elif angle <= low_score_angle:
                    cur_score = 40

            scores.append(cur_score * score_weight)
        
        return sum(scores)
    
    def getActionNum(self):
        return self.action_nums
    
    def getActionName(self):
        return self.action_name
    
    def reset(self):
        self.action_scores.clear()
        self.idx = 0
        
        
        



if __name__ == "__main__":
    main()
