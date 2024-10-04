import cv2
import numpy as np
import sys
import io
import datetime
import time
import shutil
import os
from YOLOV7_Tracking import *

# sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
# sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

enable_GPU = True
conf_thres = 0.5
iou_thres = 0.5
vel_belt = 20.0
view_img = True
fps = 30
d0 = 0.15
d1 = 0.47


def job():

    global enable_GPU 
    global conf_thres 
    global iou_thres
    global vel_belt 
    global view_img 
    global fps
    global d0
    global d1
    
    try:
        print("Opening web camera...")
        cap = cv2.VideoCapture(cv2.CAP_DSHOW+0)
    except:
        print("Failed opening web camera!")
        return

    # Set camera properties of width and height
    cap.set(3, 1280/2)
    cap.set(4, 720/2)
    
    # Get camera properties of width and height
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    # Set codec
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    
    cur_time = datetime.datetime.now()
    str_time = str(cur_time.strftime('%Y%m%d-%H%M%S'))
    out = cv2.VideoWriter(str_time+'.mp4', fourcc, fps, (width, height))
    
    n_frame = 0
    while(n_frame < fps*3):
        n_frame = n_frame + 1
        ret, frame = cap.read()

        if not ret:
            print("Error: reading video frame")
            break

        curTime = datetime.datetime.now()
        strTime = str(curTime.strftime('%Y-%m-%d %H:%M:%S'))
        cv2.putText(frame, strTime, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
        
        # Show the video frame
        cv2.imshow('video', frame)
        
        # Write the video frame
        out.write(frame)

        k= cv2.waitKey(1)
        if(k == 27):
            break
    
    # Close and destroy windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()
   
    # YOLOR_Tracking 함수
    input_video = str_time+'.mp4'
    source = f'{input_video}'
    conf_thres = 0.25
    iou_thres = 0.45

    file_name = detect(source, conf_thres, iou_thres,save_img=False)

    #Date transmit
    # os.system(f"scp -P 33470 C:/Users/MSDL-egg/pyEggCount-master/Cage_data/{file_name}.txt root@ubimasok.iptime.org:/root/data")
    
    shutil.move(str_time+".mp4", "./out_record/"+str_time+".mp4")
    return file_name

      

if __name__ == "__main__":

    if os.path.isdir("archive") == False:
        os.makedirs("archive")


    #함수 일정시간 실행
    job()
    #schedule.every().day.at("10:30").do(job)

   


    # while True:
    #     time.sleep(1)