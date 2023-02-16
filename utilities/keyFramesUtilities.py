from tqdm import tqdm
import cv2
import numpy as np
import os, os.path



def weighted_average(axes): 
    axes_cordinate=0
    axes_sum = 0
    for i in range (len(axes)):
        axes_cordinate = axes_cordinate + (axes[i]*i)
        axes_sum= axes_sum+axes[i]
    axes_cordinate= axes_cordinate/ axes_sum
    return axes_cordinate

def find_center_of_gravity(path):
    cordinates_x=[]
    cordinates_y=[]

    files = os.listdir(path) 
    files.sort(key = lambda x: int(x[:-4]))

    for i in tqdm(range (len(files) )):
        frame= cv2.imread(path+files[i])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = 255-frame
        x= (np.mean(frame, axis=0))
        y= (np.mean(frame, axis=1))
        cordinates_x.append(weighted_average(x))
        cordinates_y.append(weighted_average(y))
        
    cordinates_x=np.asarray(cordinates_x)
    cordinates_y=np.asarray(cordinates_y)
    
    return  cordinates_x.astype(int), cordinates_y.astype(int)

def save_key_frames(frames_path, keyFramesList, step=4, threshold=60, near=30, slowmotion=6, speedUp=10, title="final", fps=24):
    """
    input: frames: of the video
           keyFramesList: lis of keyframes index
           threshold: i save on the output video video at 3x speed  frames that are between  keyFramesList-threshold and keyFramesList+threshold
                       others are not relevant
           near: frames that are betwen keyFramesList-near and keyFramesList+near are slowdown of [slowmotion] parameter 
           in order to get more importance
    """
    slowmotion_count=0
    selected_frames=[]
    focus_frames=[]
    keyFramesList.sort()
    keyFramesList = [x * step for x in keyFramesList]
    print(keyFramesList)
    files = os.listdir(frames_path) 
    files.sort(key = lambda x: int(x[:-4]))

    image = cv2.imread(frames_path+files[0])
    h,w = image.shape[:2]

    for key in keyFramesList:
        selected_frames.extend(range(-near+key,near+key))
        focus_frames.extend(range(-threshold+key, threshold+key))
    
    image = cv2.imread(frames_path+files[0])
    w,h = image.shape[:2] 
    writer = cv2.VideoWriter(title, cv2.VideoWriter_fourcc(*'PIM1'), fps, (h,w))
    
    for i in tqdm(range(len(files))):
        image = cv2.imread(frames_path+files[i])
        if i in selected_frames:
            for i in range(slowmotion):
                writer.write(image)

        elif i in focus_frames:
            writer.write(image)            
        elif (slowmotion_count%speedUp)==0:
            writer.write(image)
        slowmotion_count+=1