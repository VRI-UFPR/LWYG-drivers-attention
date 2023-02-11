"""
Script used to get labeled gaze angle data from
Driver Monitoring Dataset rgb_face videos and annotations.
Request access to the dataset at https://dmd.vicomtech.org/
"""


import argparse
import numpy as np
import cv2
import time
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
from PIL import Image
from gaze_utils import select_device, draw_gaze
from PIL import Image, ImageOps
from face_detection import RetinaFace
from gaze_estimation_model import L2CS
import json

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Prepare DMD using L2CS-Net')

    parser.add_argument(
        '--gpu',dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)

    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of L2CS-Net.', 
        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)

    parser.add_argument(
        '--video_dir',dest='video_dir', help='Folder where DMD video files are',
        default=None, type=str)

    parser.add_argument(
        '--json_dir',dest='json_dir', help='Folder where DMD annotation JSON files are',
        default=None, type=str)

    parser.add_argument(
        '--xnpys_dir',dest='xnpys_dir', help='Where to store X numpy arrays',
        default=None, type=str)

    parser.add_argument(
        '--ynpys_dir',dest='ynpys_dir', help='Where to store y numpy arrays',
        default=None, type=str)


    args = parser.parse_args()
    return args

def getArch(arch,bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS( torchvision.models.resnet.BasicBlock,[2, 2,  2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS( torchvision.models.resnet.BasicBlock,[3, 4,  6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                'The default value of ResNet50 will be used instead!')
        model = L2CS( torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], bins)
    return model

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    arch=args.arch
    batch_size = 1
    gpu = select_device(args.gpu_id, batch_size=batch_size)
    snapshot_path = args.snapshot 
    video_dir = args.video_dir
    json_dir = args.json_dir
    xnpys_dir = args.xnpys_dir
    ynpys_dir = args.ynpys_dir

    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    gaze_estimation_model = getArch(arch, 90)
    print('Loading snapshot.')
    saved_state_dict = torch.load(snapshot_path)
    gaze_estimation_model.load_state_dict(saved_state_dict)
    gaze_estimation_model.cuda(gpu)
    gaze_estimation_model.eval()

    softmax = nn.Softmax(dim=1)
    detector = RetinaFace(gpu_id=0)
    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    x=0

    video_files = []
    all_files = os.listdir(video_dir)
    for file in all_files:
        if file.endswith(".mp4"):
            video_files.append(os.path.join(video_dir, file))

    ########## GET GAZE DIRECTION FROM VIDEOS
    for video_filename in video_files:
        basename = os.path.basename(video_filename).replace(".mp4", "")
        gaze_output = os.path.join(xnpys_dir,  basename + "_gaze_data.npy")
        cap = cv2.VideoCapture(os.path.join(video_dir, video_filename))

        # Check if the video capture was opened correctly
        if not cap.isOpened():
            raise IOError(f"Cannot open video {video_filename}")

        values = []
        with torch.no_grad():
            frame_index = 1
            sucess = True
            success, frame = cap.read()    
            while sucess:
                area_and_face = []
                try:
                    faces = detector(frame)
                except NotImplementedError:

                    values = np.array(values)
                    np.save(gaze_output, values)
                    break

                for box, landmarks, score in faces:
                    if score < .95:
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

                    # Only process largest face
                    area_and_face.sort()

                if area_and_face:

                    box = area_and_face[-1][1]
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

                    # Crop image
                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.resize(img, (224, 224))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    img=transformations(im_pil)
                    img  = Variable(img).cuda(gpu)
                    img  = img.unsqueeze(0) 
                    
                    # gaze prediction
                    gaze_yaw, gaze_pitch = gaze_estimation_model(img)
                    
                    pitch_predicted = softmax(gaze_pitch)
                    yaw_predicted = softmax(gaze_yaw)
                    
                    # Get continuous predictions in degrees.
                    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
                    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180
                    
                    pitch_predicted= pitch_predicted.cpu().detach().numpy()* np.pi/180.0
                    yaw_predicted= yaw_predicted.cpu().detach().numpy()* np.pi/180.0
                    predicted_values = (yaw_predicted,pitch_predicted)
                    
                else:
                    predicted_values = (42, 42) # EXCEPTION VALUES

                value = np.array(predicted_values)
                values.append(value)
                success, frame = cap.read()    


    ######### GET VALUES FROM ANNOTATION JSONs
    basenames = os.listdir(json_dir)
    files = [(os.path.join(json_dir, file), file) for file in basenames if file.endswith(".json")]
    
    for file, basename in files:
        with open(file) as reading:
            d = json.load(reading)
            first_label = "openlabel"
            if not first_label in d:
                first_label = "vcd"
            frames = []

            try:
                actions = d[first_label]["actions"]
            except KeyError:
                actions = d[first_label]["actions"]
                print(file)
                raise KeyError

            for action in actions:
                if actions[action]["type"] == "gaze_on_road/looking_road":
                    for interval in actions[action]["frame_intervals"]:
                        frames.append((interval["frame_start"], interval["frame_end"]))

            total_frames = d[first_label]["streams"]["face_camera"]["stream_properties"]["total_frames"]
            ranges = [range(a, b + 1) for (a, b) in frames]

            res = np.empty(total_frames, dtype=bool)
            for i in range(total_frames):
                res[i] = False
                for frame_range in ranges:
                    if i in frame_range:
                        res[i] = True
                        break
            
            out_file = basename.replace("_rgb_ann_distraction.json", "_looking_road_label.npy")
            out_file = os.path.join(ynpys_dir, out_file) 
            np.save(out_file, res)

    train = [
        "gA_1_s1_2019-03-08T09;31;15+01;00",
        "gA_2_s1_2019-03-08T10;01;44+01;00",
        "gA_3_s1_2019-03-08T10;27;38+01;00",
        "gA_4_s1_2019-03-13T10;36;15+01;00",
        "gA_5_s1_2019-03-08T10;57;00+01;00",
        "gB_6_s1_2019-03-11T13;55;14+01;00",
        "gB_7_s1_2019-03-11T14;22;01+01;00",
        "gB_8_s1_2019-03-11T15;01;33+01;00",
        "gB_9_s1_2019-03-07T16;36;24+01;00",
        "gB_10_s1_2019-03-11T15;24;54+01;00",
        "gC_11_s1_2019-03-04T09;33;18+01;00",
        "gC_12_s1_2019-03-13T10;23;45+01;00",
        "gC_13_s1_2019-03-04T10;26;12+01;00",
        "gC_14_s1_2019-03-04T11;56;20+01;00",
        "gC_15_s1_2019-03-04T11;24;57+01;00",
        "gF_21_s1_2019-03-05T09;48;30+01;00",
        "gF_22_s1_2019-03-04T14;54;55+01;00",
        "gF_23_s1_2019-03-04T16;21;10+01;00",
        "gF_24_s1_2019-03-04T15;26;10+01;00",
        "gF_25_s1_2019-03-04T15;53;22+01;00",
    ]
    test = [
        "gE_26_s1_2019-03-15T09;25;24+01;00",
        "gE_27_s1_2019-03-07T13;18;37+01;00",
        "gE_28_s1_2019-03-15T10;23;30+01;00",
        "gE_29_s1_2019-03-15T13;58;00+01;00",
        "gE_30_s1_2019-03-15T10;58;06+01;00",
        "gZ_31_s1_2019-04-08T09;48;48+02;00",
        "gZ_32_s1_2019-04-08T12;01;26+02;00",
        "gZ_33_s1_2019-04-08T10;08;19+02;00",
        "gZ_34_s1_2019-04-08T12;25;28+02;00",
    ]

    ### JOIN X ARRAYS
    sufix = "_rgb_face_gaze_data.npy"

    #Train

    X_train = []
    for prefix in train:
        array_file = os.path.join(xnpys_dir, prefix + sufix) 
        X_train += list(np.load(array_file, allow_pickle=True))

    X_train = np.array(X_train)

    #Test

    X_test = []
    for prefix in test:
        array_file = os.path.join(xnpys_dir, prefix + sufix) 
        X_test += list(np.load(array_file, allow_pickle=True))

    X_test = np.array(X_test)


    ### JOIN y ARRAYS
    sufix = "_looking_road_label.npy"

    #Train

    y_train = []
    for prefix in train:
        array_file = os.path.join(ynpys_dir, prefix + sufix) 
        y_train += list(np.load(array_file, allow_pickle=True))

    y_train = np.array(y_train)

    #Test

    y_test = []
    for prefix in test:
        array_file = os.path.join(ynpys_dir, prefix + sufix) 
        y_test += list(np.load(array_file, allow_pickle=True))

    y_test = np.array(y_test)


    ##### FILTER OUT EXCEPTION VALUES

    train_indexes = [i for i in range(len(X_train)) if 42 not in X_train[i]]
    test_indexes = [i for i in range(len(X_test)) if 42 not in X_test[i]]

    X_train =  X_train[train_indexes]
    X_test =  X_test[test_indexes]

    y_train =  y_train[train_indexes]
    y_test =  y_test[test_indexes]

    np.save(os.path.join(ynpys_dir, "TRAIN_y.npy"), y_train)
    np.save(os.path.join(ynpys_dir, "TEST_y.npy"), y_test)
    np.save(os.path.join(xnpys_dir, "TRAIN_X.npy"), X_train)
    np.save(os.path.join(xnpys_dir, "TEST_X.npy"), X_test)