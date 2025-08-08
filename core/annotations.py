

import os
import cv2

import matplotlib.pyplot as plt
import json

class BoxInfo:
    def __init__(self, player_id, frame_id, x, y, w, h, activity):
        self.player_id = player_id
        self.frame_id = int(frame_id)
        self.x1 = int(x)
        self.y1 = int(y)
        self.x2 = self.x1 + int(w)
        self.y2 = self.y1 + int(h)
        self.activity = activity

def parse_annotation_line(line):
    group_activity_dict={}
    tokens = line.strip().split()
    frame_name = tokens[0] 
    frame_id = int(frame_name.replace(".jpg", ""))  
    group_activity = tokens[1]  
    
    box_infos = []
    player_id = 0

    for i in range(2, len(tokens), 5):
        if i + 4 < len(tokens):  
            x1 = int(tokens[i])
            y1 = int(tokens[i+1])
            w = int(tokens[i+2])
            h = int(tokens[i+3])
            activity = tokens[i+4]
            box = BoxInfo(player_id, frame_id,x1, y1, w, h,activity)
            box_infos.append(box)
            player_id += 1

    return frame_id, group_activity, box_infos

def read_file(path):
    file = open(path, "r")
    video_info = {}
    activity_set = set()

    for line in file:   # reading annotation.text
        frame_id, group_activity, box_infos = parse_annotation_line(line)
        activity_set.add(group_activity)  

        video_info[frame_id] = {
            "groupactivity": group_activity,
            "boxinfos": box_infos
        }

    file.close()

    activity_list = sorted(list(activity_set))  
    activity_to_id = {activity: idx for idx, activity in enumerate(activity_list)}
    id_to_activity = {idx: activity for activity, idx in activity_to_id.items()}

    return video_info, activity_to_id, id_to_activity


def save_label_mappings(activity_to_id, save_path="group_activity_labels.json"):
    with open(save_path, "w") as f:
        json.dump(activity_to_id, f, indent=4)




