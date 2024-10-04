import os
import cv2
import time
import torch
import argparse
from pathlib import Path
from numpy import random
from random import randint
from collections import deque
import datetime
import torch.backends.cudnn as cudnn
import math


from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, \
                check_imshow, non_max_suppression, apply_classifier, \
                scale_coords, xyxy2xywh, strip_optimizer, set_logging, \
                increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, \
                time_synchronized, TracedModel
from utils.download_weights import download

#For SORT tracking
import skimage
from sort import *

# 변수선언
list1 = []
list2 = []
idx_cage = {}

data_deque = {}
speed_four_line_queue = {}
object_time = {}
object_counter = {}
color_str1 = [167, 121, 204] # Reddish purple, BGR
color_str1 = [200, 200, 200] # Reddish purple, BGR
color_str2 = [0, 0, 0]

# Count eggs using YOLOv7+DeepSORT
source = '../sample/test_00.mp4'
enable_GPU = True
conf_thres = 0.5
iou_thres = 0.5
vel_belt = 20.0
view_img = True
d0 = 0.47
d1 = 0.15


# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def draw_border(img, pt1, pt2, color, thick, r, d):

    x1,y1 = pt1
    x2,y2 = pt2
    
    # Top left
    cv2.line(img, (x1+r, y1), (x1+r+d, y1), color, thick)
    cv2.line(img, (x1, y1+r), (x1, y1+r+d), color, thick)
    cv2.ellipse(img, (x1+r, y1+r), (r, r), 180, 0, 90, color, thick)

    # Top right
    cv2.line(img, (x2-r, y1), (x2-r-d, y1), color, thick)
    cv2.line(img, (x2, y1+r), (x2, y1+r+d), color, thick)
    cv2.ellipse(img, (x2-r, y1+r), (r, r), 270, 0, 90, color, thick)
    
    # Bottom left
    cv2.line(img, (x1+r, y2), (x1+r+d, y2), color, thick)
    cv2.line(img, (x1, y2-r), (x1, y2-r-d), color, thick)
    cv2.ellipse(img, (x1+r, y2-r), (r, r), 90, 0, 90, color, thick)
    
    # Bottom right
    cv2.line(img, (x2-r, y2), (x2-r-d, y2), color, thick)
    cv2.line(img, (x2, y2-r), (x2, y2-r-d), color, thick)
    cv2.ellipse(img, (x2-r, y2-r), (r, r), 0, 0, 90, color, thick)
    
    cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1+r), (x2, y2-r-d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1+r, y1+r), 2, color, 12)
    cv2.circle(img, (x2-r, y1+r), 2, color, 12)
    cv2.circle(img, (x1+r, y2-r), 2, color, 12)
    cv2.circle(img, (x2-r, y2-r), 2, color, 12)
    
    return img


# Plots one bounding box on image
def UI_box(box, img, color=None, label=None, line_thick=None):
    # Thickness of the line and font
    tl = line_thick or round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1
    
    # Set colors
    color = color or [random.randint(0, 255) for _ in range(3)]
    
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    
    if label:
        tf = max(tl-1, 1)   # Font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl/4, thickness=tf)[0]
        img = draw_border(img, (c1[0], c1[1]-t_size[1]-3), (c1[0]+t_size[0], c1[1]+3), color, 1, 8, 2)
        cv2.putText(img, label, (c1[0], c1[1]-2), 0, tl/4, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return


def draw_boxes(img, bbox, line, vel_belt, frame_fps, frame_pos, d0, d1, identities=None, categories=None, names=None, save_with_object_id=False, path=None,offset=(0, 0)):
    
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        center_point = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))


        # Create new buffer for new object
        if id not in data_deque:
            data_deque[id] = deque(maxlen= 64)
            speed_four_line_queue[id] = []
            object_time[id] = []

        # Add center to buffer
        data_deque[id].appendleft(center_point)

        #bbox 위 텍스트
        obj_name = names[cat]
        colors = [0, 94, 231]  # Vermiion, BGR color

        if len(data_deque[id]) >= 2:
            if intersect(data_deque[id][0], data_deque[id][1], line[0], line[1]):

                object_time[id].append(round(frame_pos/frame_fps*vel_belt*0.01, 2))

                # Cage number calculate
                distance = object_time[id]
                output = np.subtract(distance,d1)
                idx_cage = ((output//d0)+1)
                idx_cage_list = idx_cage.tolist()
                int_idx_cage_list = list(map(int,idx_cage_list))
               
                #Cage number list print
                if len(int_idx_cage_list)>0:
                    
                    for i in int_idx_cage_list: 
                        list1.extend(int_idx_cage_list)

                if obj_name not in object_counter:  
                    object_counter[obj_name] = 1
                else:
                    object_counter[obj_name] += 1

        #Cage number list
        distance = object_time[id]
        output = np.subtract(distance,d1)
        idx_cage = ((output//d0)+1)
        
        try:
            #label = label + " " + str(sum(speed_four_line_queue[id])//len(speed_four_line_queue[id]))
            if(len(object_time[id]) == 0):
                label = '%s' % (obj_name)
            else:
                label = str(sum(object_time[id])) + "m" + "| cage" +str(int(idx_cage[0]))

        except:
            pass

        # Plots one bounding box on image
        UI_box(box, img, label=label, color=colors)
        
        cv2.circle(img, center_point, 2, (0,0,255), -1)   #centroid of box

    # Count the total number of objects
    count = 0
    for idx, (key, value) in enumerate(object_counter.items()):
        cnt_str = str(key) + ": " + str(value)
        count += value

    print(idx_cage)
            
    return img, count, idx_cage, list1



def detect(source, conf_thres, iou_thres,save_img=False):
    global list1

    cur_time = datetime.datetime.now()
    str_time = str(cur_time.strftime('%Y%m%d'))
 
    weights = '../weight/yolov7_egg.pt'
    view_img = True
    save_txt = False
    imgsz = img_size = 640
    trace = False
    colored_trk = False
    save_bbox_dim = False
    save_with_object_id = False
    project = '../out_infer'
    name = f'{str_time}'
    exist_ok = True
    device = '0'
    augment = False
    agnostic_nms = False
    classes = 0
    


    save_img = not False and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    #.... Initialize SORT .... 
    #......................... 
    sort_max_age = 5 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh)
    #......................... 
    
    #........Rand Color for every trk.......
    rand_color_list = []
    for i in range(0,5005):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)
    #......................................
   
    # Directories
    save_dir = Path(increment_path(Path(project) / name, exist_ok= exist_ok))  # increment run
    (save_dir / 'labels' if save_txt or save_with_object_id else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'CUDA'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    prevTime = 0
    count = 0
    idx_cage = 0
    t0 = time.time()


    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic= agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Find video properties
        frame_fps = vid_cap.get(cv2.CAP_PROP_FPS)               # FPS
        frame_pos = vid_cap.get(cv2.CAP_PROP_POS_FRAMES)        # Current frame number
        frame_count = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)     # Total number of frames
        frame_str = str(int(frame_pos)) + "/" + str(int(frame_count))

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            
            # 라인 생성
            im0 = im0s
            height, width, _ = im0.shape
            line = [(int(0.7*width), 0), (int(0.7*width), height)]
            cv2.line(im0, line[0], line[1], [115, 158, 0], 5)

            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                #..................USE TRACK FUNCTION....................
                #pass an empty array to sort
                dets_to_sort = np.empty((0,6))
                
                # NOTE: We send in detected object class too
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, 
                                np.array([x1, y1, x2, y2, conf, detclass])))
                
                # Run SORT
                tracked_dets = sort_tracker.update(dets_to_sort)
                tracks =sort_tracker.getTrackers()

                txt_str = ""

                #loop over tracks
                for track in tracks:
                    # color = compute_color_for_labels(id)
                    #draw colored tracks
                    if colored_trk:
                        [cv2.line(im0, (int(track.centroidarr[i][0]),
                                    int(track.centroidarr[i][1])), 
                                    (int(track.centroidarr[i+1][0]),
                                    int(track.centroidarr[i+1][1])),
                                    rand_color_list[track.id], thickness=2) 
                                    for i,_ in  enumerate(track.centroidarr) 
                                      if i < len(track.centroidarr)-1 ] 
                    #draw same color tracks
                    else:
                        [cv2.line(im0, (int(track.centroidarr[i][0]),
                                    int(track.centroidarr[i][1])), 
                                    (int(track.centroidarr[i+1][0]),
                                    int(track.centroidarr[i+1][1])),
                                    (255,0,0), thickness=2) 
                                    for i,_ in  enumerate(track.centroidarr) 
                                      if i < len(track.centroidarr)-1 ] 

                    if save_txt and not save_with_object_id:
                        # Normalize coordinates
                        txt_str += "%i %i %f %f" % (track.id, track.detclass, track.centroidarr[-1][0] / im0.shape[1], track.centroidarr[-1][1] / im0.shape[0])
                        if save_bbox_dim:
                            txt_str += " %f %f" % (np.abs(track.bbox_history[-1][0] - track.bbox_history[-1][2]) / im0.shape[0], np.abs(track.bbox_history[-1][1] - track.bbox_history[-1][3]) / im0.shape[1])
                        txt_str += "\n"
                
                if save_txt and not save_with_object_id:
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(txt_str)

                # draw boxes for visualization
                if len(tracked_dets)>0:
                    bbox_xyxy = tracked_dets[:,:4]
                    identities = tracked_dets[:, 8]
                    categories = tracked_dets[:, 4]
                    im0, count, idx_cage, list1 = draw_boxes(im0, bbox_xyxy,line, vel_belt, frame_fps, frame_pos, d0, d1, identities, categories, names, save_with_object_id, txt_path)
                #........................................................
                
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            
            # Print information in the video
            cv2.line(im0, (20, 25), (170, 25), color_str1, 30)
            cv2.putText(im0, f'FRAME: {frame_str}', (20, 30), 0, 0.5, color_str2, thickness=1, lineType=cv2.LINE_AA)
            
            time_str = vid_cap.get(cv2.CAP_PROP_POS_FRAMES) / vid_cap.get(cv2.CAP_PROP_FPS)
            cv2.line(im0, (20, 25+40), (170, 25+40), color_str1, 30)
            cv2.putText(im0, 'TIME: {:.2f}s'.format(time_str), (20, 30+40), 0, 0.5, color_str2, thickness=1, lineType=cv2.LINE_AA)
            
            cnt_str = "EGGS: " + str(count)
            cv2.line(im0, (20, 25+80), (170, 25+80), color_str1, 30)
            cv2.putText(im0, cnt_str, (20, 30+80), 0, 0.5, color_str2, thickness=1, lineType=cv2.LINE_AA)

            cage_str = "CAGE NO.: " + str(idx_cage)
            cv2.line(im0, (20, 25+120), (170, 25+120), color_str1, 30)
            cv2.putText(im0, cage_str, (20, 30+120), 0, 0.5, color_str2, thickness=1, lineType=cv2.LINE_AA)

        
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                  cv2.destroyAllWindows()
                  raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img or save_with_object_id:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

    #Count Cage ->egg save txt file

    Cage_egg_counter = {}
    for value in list1:
        if value in Cage_egg_counter:
            Cage_egg_counter[value] +=1
        else:
            Cage_egg_counter[value] =1
   
    print(Cage_egg_counter)
    cur_time = datetime.datetime.now()
    str_time = str(cur_time.strftime('%Y%m%d'))
 
    with open('../'+str_time+'.txt','w',encoding='UTF-8') as f:
        for key,value in Cage_egg_counter.items():
            f.write(f'{key}  {value}\n')



if __name__ == '__main__':

    source = '../sample/test2.mp4'
    conf_thres = 0.25
    iou_thres = 0.45

  
   
    detect(source, conf_thres, iou_thres,save_img=False)
