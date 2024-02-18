import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm

import sys
import os
sys.path.append("../rene")
from rene.utils.loaders import ReneDataset
os.makedirs('./training/lighting', exist_ok=1)

def sort():
    rene = ReneDataset(input_folder="/viscam/projects/scenegen/treehacks24/data/rene_dataset/")
    
    arr = []
    objn = rene._KEYS_TO_LOAD
    for object_key in objn:
        n = len(rene[object_key])
        for light_key in tqdm(range(n-1)):
            #if light_key == 3: exit(0)
            #for light_key2 in tqdm(range(n)):
                light_key2=(light_key+1)%n
                nn = len(rene[object_key][light_key])
                for camera_key in range(nn):
                    source = rene[object_key][light_key][camera_key]
                    target = rene[object_key][light_key2][camera_key]
                    prompt = str(object_key)
                    
                    srcf = str(source["pose"].__dict__["_file_sources"][0]).replace("pose.txt", "image.png")
                    tarf = str(target["pose"].__dict__["_file_sources"][0]).replace("pose.txt", "image.png'")
#                     1/0
#                     v = np.array([0,0,0,1])
#                     res = rene[object_key][light_key2][camera_key]["light"]()@v
#                     res = np.array([res[0]/res[3], res[1]/res[3],res[2]/res[3]])
                    res = rene[object_key][light_key2][camera_key]["light"]()[:,-1]
                    if res[0]<-1:
                        prompt+= " left"
                    elif res[0]>1:
                        prompt+= " middle"
                    else:
                        prompt+= " right"
                    if res[1]<-1:
                        prompt+= " bottom"
                    elif res[1]>1:
                        prompt+= " middle"
                    else:
                        prompt+= " top"
                    if res[2]<-1:
                        prompt+= " front"
                    elif res[2]>1:
                        prompt+= " middle"
                    else:
                        prompt+= " back"
#                     1/0
                        
#                     print("AHHHH",source, target)
#                     exit(0)
                    arr.append((tarf, tarf, prompt))
    return arr
def triplet_to_json(e):
    return {
        "source": e[0],
        "target": e[1],
        "prompt": e[2]
    }
arr = sort()
arr = [triplet_to_json(e) for e in arr]
with open('./training/lighting/prompts.json', 'w+') as f:
#     print(len(arr))
    
    for e in arr:
        f.write(e)
        f.write("\n")
