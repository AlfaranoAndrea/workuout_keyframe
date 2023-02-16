import os, os.path
import cv2
from tqdm import tqdm
def extract_frames(path, temporary_path= "./temp/Frames/"):
    os.makedirs(temporary_path, exist_ok=True)
    vidcap = cv2.VideoCapture(path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    print(fps)
    count = 0
    success,image = vidcap.read()
    while success:
            cv2.imwrite(f"{temporary_path}{count}.jpg", image)
            success,image = vidcap.read()
            count += 1
    print(f'Readed {count}  frames ')
    
    return fps 

def saveVideo (path, title, fps=24):
    title = title +".avi"
    files = os.listdir(path) 
    files.sort(key = lambda x: int(x[:-4]))
    image = cv2.imread(path+files[0])
    h,w = image.shape[:2]
    writer = cv2.VideoWriter(title, cv2.VideoWriter_fourcc(*'PIM1'), fps, (w,h))

    for i in tqdm(range(len(files))):
        image = cv2.imread(path+files[i])
        writer.write(image)