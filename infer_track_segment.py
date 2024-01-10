import argparse
import sys
import os
from track_anything import *

def parse_augment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--sam_model_type', type=str, default="vit_h")
    #parser.add_argument('--port', type=int, default=6080, help="only useful when running gradio applications")  
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--mask_save', default=True)
    parser.add_argument('--input_video', default=None)
    parser.add_argument('--input_clicks', default=None)
    parser.add_argument('--output_folder', default=None)
    args = parser.parse_args()

    if args.debug:
        print(args)
    return args 

if __name__ == "__main__":
    masks = None
    logits = None
    painted_images = None
    images = []
    args = parse_augment()
    video_path = args.input_video

    

    model = get_model(args=args)
    video_state,video_info,first_frame = get_frames_from_video(video_input=video_path,video_state=None,model=model)
    
    points,labels = get_prompt(args.input_clicks)
    mask,_,_ = model.first_frame_click(image=first_frame,points=points,labels=labels)
   
    masks, logits ,painted_images= model.generator(video_state["origin_images"], mask)
    video_state["masks"]=masks
    name = save_mask_video(video_state,args)
    
    
    
    
    
