"""
Caption a video of driving footage, according to whether
the driver is looking at road elements or not.
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from gaze_utils import select_device, draw_gaze
from PIL import Image, ImageOps

from face_detection import RetinaFace
from gaze_estimation_model import L2CS

import argparse
import numpy as np
import cv2
import time
import pickle

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='A driver monitoring system simulation.')
    parser.add_argument(
        '--gpu',dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--gaze_model',dest='gaze_model', help='Gaze estimation model .pkl', 
        default=None, type=str)
    parser.add_argument(
        '--video_source',dest='video_filename', help='Video to be captioned',
        default=None, type=str)
    parser.add_argument(
        '--video_output',dest='video_output', help='Video file output',
        default=None, type=str)
    parser.add_argument(
        '--distraction_model',dest='distraction_model_file', help='Visual distraction classifier .pkl',
        default=None, type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    batch_size = 1
    gpu = select_device(args.gpu_id, batch_size=batch_size)
    gaze_model_path = args.gaze_model 
    video_filename = args.video_filename
    video_output = args.video_output
    distraction_model_file = args.distraction_model_file

    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    gaze_estimation_model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], 90)

    print('Loading gaze estimation model...')
    saved_state_dict = torch.load(gaze_model_path)
    gaze_estimation_model.load_state_dict(saved_state_dict)
    gaze_estimation_model.cuda(gpu)
    gaze_estimation_model.eval()

    softmax = nn.Softmax(dim=1)
    detector = RetinaFace(gpu_id=0)
    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    distraction_model = pickle.load(open(distraction_model_file, "rb"))
  
    cap = cv2.VideoCapture(video_filename)
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open video")

    print('Processing video...')

    video_out = cv2.VideoWriter(video_output, fourcc, 30, (1280,720))

    with torch.no_grad():
        retval, frame = cap.read()    
        while retval:
        
            try:
                faces = detector(frame)
            except NotImplementedError:
                # video ended
                break

            area_and_face = []
            if faces: 
                for box, landmarks, score in faces:
                    if score < .95:
                        # invalid face
                        continue

                    x_min=int(box[0])
                    if x_min < 0:
                        x_min = 0
                    y_min=int(box[1])
                    if y_min < 0:
                        y_min = 0
                    x_max=int(box[2])
                    y_max=int(box[3])
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min
                    face_area = bbox_height * bbox_width
                    area_and_face.append((face_area, box))

            if area_and_face: # If a face was recognized in the picture

                # Only process largest face
                area_and_face.sort()
                box = area_and_face[-1][1]

                x_min = int(box[0])
                if x_min < 0:
                    x_min = 0
                y_min = int(box[1])
                if y_min < 0:
                    y_min = 0
                x_max = int(box[2])
                y_max = int(box[3])
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min

                # Crop image (only face)
                img = frame[y_min:y_max, x_min:x_max]
                img = cv2.resize(img, (224, 224))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)
                img = transformations(im_pil)
                img  = Variable(img).cuda(gpu)
                img  = img.unsqueeze(0) 
                
                # Gaze prediction
                gaze_yaw, gaze_pitch = gaze_estimation_model(img)
                pitch_predicted = softmax(gaze_pitch)
                yaw_predicted = softmax(gaze_yaw)
                
                # Get continuous predictions in degrees.
                pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
                yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180
                
                pitch_predicted= pitch_predicted.cpu().detach().numpy()* np.pi/180.0
                yaw_predicted= yaw_predicted.cpu().detach().numpy()* np.pi/180.0

                # Draw gaze
                
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
                draw_gaze(x_min,y_min,bbox_width, bbox_height,frame,(yaw_predicted,pitch_predicted),color=(245,179,66), thickness=10, scale=0.6)

                angle_values = np.array([np.array((yaw_predicted,pitch_predicted))])

            #### Process output:

            if not area_and_face:
                output_str = "???"
                color = (255, 255, 255)

            else:
                prediction = distraction_model.predict(angle_values)[0]
                if prediction == True:
                    output_str = f"Looking at road elements"
                    color = (0, 255, 100)
                
                else:
                    output_str = f"Visually distracted"
                    color = (0, 100, 255)

            text_size, _ = cv2.getTextSize(output_str, cv2.FONT_HERSHEY_PLAIN, 4, 4)
            text_w, text_h = text_size

            cv2.putText(frame, output_str, (50, 20 + text_h), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)
            video_out.write(frame)
            retval, frame = cap.read()    
