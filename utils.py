
from PIL import Image
import numpy as np
import cv2
import psutil
import time 
import os
import ffmpeg
import json



# extract frames from upload video
def get_frames_from_video(video_input, video_state,model):
    """
    Args:
        video_path:str
        timestamp:float64
    Return 
        [[0:nearest_frame], [nearest_frame:], nearest_frame]
    """
    video_path = video_input
    frames = []
    user_name = time.time()
    operation_log = [("",""),("Upload video already. Try click the image for adding targets to track and inpaint.","Normal")]
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                current_memory_usage = psutil.virtual_memory().percent
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if current_memory_usage > 90:
                    operation_log = [("Memory usage is too high (>90%). Stop the video extraction. Please reduce the video resolution or frame rate.", "Error")]
                    print("Memory usage is too high (>90%). Please reduce the video resolution or frame rate.")
                    break
            else:
                break
    except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
        print("read_frame_source:{} error. {}\n".format(video_path, str(e)))
    image_size = (frames[0].shape[0],frames[0].shape[1]) 
    # initialize video_state
    video_state = {
        "user_name": user_name,
        "video_name": os.path.split(video_path)[-1],
        "origin_images": frames,
        "painted_images": frames.copy(),
        "masks": [np.zeros((frames[0].shape[0],frames[0].shape[1]), np.uint8)]*len(frames),
        "logits": [None]*len(frames),
        "select_frame_number": 0,
        "fps": fps
        }
    video_info = "Video Name: {}, FPS: {}, Total Frames: {}, Image Size:{}".format(video_state["video_name"], video_state["fps"], len(frames), image_size)
    model.samcontroler.sam_controler.reset_image() 
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][0])
    return video_state, video_info, video_state["origin_images"][0]

def get_prompt(click_input):
    
    with open(click_input,'r') as f:
        inputs = json.load(f)
    f.close()
    points = []
    labels = []
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    return np.array(points),np.array(labels)

def save_mask_video(video_state,args):
    if args.mask_save==True:
        if not os.path.exists('./result/mask/{}'.format(video_state["video_name"].split('.')[0])):
            os.makedirs('./result/mask/{}'.format(video_state["video_name"].split('.')[0]))
        i = 0
        print("save mask")
        for mask in video_state["masks"]:
            im = Image.fromarray(mask*255)
            im.save(os.path.join('./result/mask/{}'.format(video_state["video_name"].split('.')[0]), '{:05d}.jpeg'.format(i)))
            i+=1
        
    print("saving resultant mask video")
    fps = video_state['fps']
    name =  os.path.join('./result/mask_video/{}'.format(video_state["video_name"].split('.')[0]), 'mask.mp4')
    if not os.path.exists('./result/mask_video/{}'.format(video_state["video_name"].split('.')[0])):
            os.makedirs('./result/mask_video/{}'.format(video_state["video_name"].split('.')[0]))

    
    ffmpeg.input(os.path.join('./result/mask/{}'.format(video_state["video_name"].split('.')[0]), '*.jpeg'), pattern_type='glob', framerate=fps).output(name).run()
    print("Mask video successfully saved in {}".format(name))
    return name
