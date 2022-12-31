#!/usr/bin/env python
import os, sys
import cv2
import pyrealsense2 as rs
import numpy as np
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import math

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync


@torch.no_grad()
def run():

    weights='models_weights/original.pt'  # model.pt path(s)
    imgsz=640  # inference size (pixels)
    conf_thres=0.25  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=10  # maximum detections per image
    classes=None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False  # class-agnostic NMS
    augment=False  # augmented inference
    visualize=False  # visualize features
    line_thickness=3  # bounding box thickness (pixels)
    hide_labels=False  # hide labels
    hide_conf=False  # hide confidences
    half=False  # use FP16 half-precision inference
    stride = 32
    device_num=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False  # show results
    save_crop=False  # save cropped prediction boxes
    nosave=False  # do not save images/videos
    update=False  # update all models
    name='exp'  # save results to project/name

    # Initialize
    set_logging()
    device = select_device(device_num)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location = device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet50', n=2)  # initialize
        modelc.load_state_dict(torch.load('resnet50.pt', map_location = device)['model']).to(device).eval()

    # Dataloader
    view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    while(True):
        t0 = time.time()

        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame or not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())


        # check for common shapes
        s = np.stack([letterbox(x, imgsz, stride=stride)[0].shape for x in img], 0) # shapes s.shape (720,2)
        rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

        # Letterbox
        img0 = img.copy()
        img = img[np.newaxis, :, :, :]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_sync()
        pred = model(img, augment=augment,
                     visualize=increment_path(save_dir / 'features', mkdir=True) if visualize else False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_sync()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, img0)

        colorizer = rs.colorizer()
        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        array_img = np.asarray(img0)

        #find the nearest distance function
        def MinValue(A):
            temp = 0
            for i in A:
                if temp == 0 or i < temp:
                    temp = i
            return temp

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s = f'{i}: '
            s += '%gx%g ' % img.shape[2:]  # print string
            annotator = Annotator(img0, line_width=line_thickness, example=str(names))
            annotator_depth = Annotator(colorized_depth, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string


                # print(det)
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True)) #color map bounding box
                    annotator_depth.box_label(xyxy, label, color=colors(c, True)) #depth map bounding box

                    xyxy = (torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
                    x_start = int(xyxy[0])
                    y_start = int(xyxy[1])
                    x_end = int(xyxy[2])
                    y_end = int(xyxy[3])
                    width = (x_end - x_start)
                    height = (y_end - y_start)

                    #get the depth info from 1/2 bounding box
                    x_min = int(x_start + width * (1 / 4))
                    y_min = int(y_start + height * (1 / 4))
                    x_max = int(x_min + width * (1 / 2))
                    y_max = int(y_min + height * (1 / 2))

                    if c == 1:
                        bbox_center = (int((width/2) + x_start), int((height/2) + y_start))
                        cv2.circle(colorized_depth, bbox_center, 2, (0, 0, 255), -1)
                        depth_img = depth_image[x_min:x_max,y_min:y_max].astype(float)
                        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
                        depth = depth_img * depth_scale
                        dist, _, _, _ = cv2.mean(depth)

                        dist_round = round(dist, 2)
                        if (0.00 < dist_round < 0.70):
                            info = "Warning : There is {0} in 0.7m ".format(names[c])
                            cv2.putText(array_img, text=info, org=(30, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=0.7, color=(0, 0, 255), thickness=2)
                        else:
                            pass

                    elif c == 0:
                        # write xyxy position
                        bbox_center = (int((width/2) + x_start), int((height/2) + y_start))
                        cv2.circle(colorized_depth, bbox_center, 2, (0, 0, 255), -1)
                        depth_img = depth_image[x_min:x_max, y_min:y_max].astype(float)
                        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
                        depth = depth_img * depth_scale
                        dist, _, _, _ = cv2.mean(depth)

                        dist_round = round(dist, 2)
                        d = []
                        # get the distances in 2m
                        if (dist_round < 2.00):
                            dist_100 = dist_round * 100
                            d.append(dist_100)
                        else:
                            pass

                        # find the nearest distance
                        nearest_d = MinValue(d)
                        nearest_value = nearest_d/100
                        if nearest_value == 0:
                            continue
                        else:
                            info = "The nearest {0} is {1}m away from here".format(names[c], nearest_value)
                            cv2.putText(array_img, text=info, org=(30, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=0.7, color=(255, 0, 0), thickness=2)


        cv2.imshow("IMAGE", img0)
        cv2.imshow("DEPTH", colorized_depth)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break





if __name__ == '__main__':
    run()
