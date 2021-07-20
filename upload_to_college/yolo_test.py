"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression,xyxy2xywh,xywh2xyxy ##### add xywh2xyxy
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):
        boxes[:, 0].clamp_(0, img_shape[1])  # x1
        boxes[:, 1].clamp_(0, img_shape[0])  # y1
        boxes[:, 2].clamp_(0, img_shape[1])  # x2
        boxes[:, 3].clamp_(0, img_shape[0])  # y2
    else:  # np.array
        boxes[:, 0].clip(0, img_shape[1], out=boxes[:, 0])  # x1
        boxes[:, 1].clip(0, img_shape[0], out=boxes[:, 1])  # y1
        boxes[:, 2].clip(0, img_shape[1], out=boxes[:, 2])  # x2
        boxes[:, 3].clip(0, img_shape[0], out=boxes[:, 3])  # y2

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

########################################################################################
#add classifier
def load_classifier(name='resnet101', n=2):
    # Loads a pretrained model reshaped to n-class output
    model = torchvision.models.__dict__[name](pretrained=True)
    #import pdb;pdb.set_trace()
    # ResNet model properties
    # input_size = [3, 224, 224]
    # input_space = 'RGB'
    # input_range = [0, 1]
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # Reshape output to n classes
    classes = ('china','us','uk','russia','japan','france','german','italy','australia','korea','other','logo','background')
    if name.startswith('resnet'):
        filters = model.fc.weight.shape[1]
        model.fc.bias = nn.Parameter(torch.zeros(n), requires_grad=True)
        model.fc.weight = nn.Parameter(torch.zeros(n, filters), requires_grad=True)
        model.fc.out_features = n
    elif name.startswith('mobile'):
        #classes = ('china','us','uk','russia','japan','france','german','italy','australia','korea','other','logo','background')

        fc_inputs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(fc_inputs, len(classes))
    return model


def apply_classifier(x, model, img, im0):
    # Apply a second stage classifier to yolo outputs
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    #import pdb;pdb.set_trace()
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4])  # boxes
            #b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            #b[:, 2:] = b[:, 2:] * 1.1 + 10  # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            pred_cls1 = d[:, 5].long()
            ims = []
            test_transform=transforms.Compose([
                                transforms.Resize((112,112)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485,0.456,0.406],
                                                    std=[0.229,0.224,0.225])
                                                ])

            for j, a in enumerate(d):  # per item
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                #image = Image.fromarray(cv2.cvtColor(cutout,cv2.COLOR_BGR2RGB))
                image = Image.fromarray(cutout)
                #im = cv2.resize(cutout, (112, 112))  # BGR
                # cv2.imwrite('test%i.jpg' % j, cutout)
                
                #im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                #ascontiguousarray是将im使用行优先的方式以连续内存存储
                image = test_transform(image)
                image = np.ascontiguousarray(image, dtype=np.float32)  # uint8 to float32
                
                #im /= 255.0  # 0 - 255 to 0.0 - 1.0
                ims.append(image)
            ims_tensor = torch.tensor(ims)
            pred_cls2 = model(ims_tensor.to(d.device)).argmax(1)  # classifier prediction
            #print(pred_cls2)
            #x[i] = x[i][pred_cls1 == pred_cls2]  # retain matching class detections
            x[i] = x[i][pred_cls2 != 11] 

    return x
#############################################################################

def run(
        img0=None,  # file/dir/URL/glob, 0 for webcam
        weights='best.pt',  # model.pt path(s)
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        half=False,  # use FP16 half-precision inference
        classifier_weight='mobilenet_v3.pt'
        ):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    if half:
        model.half()  # to FP16

    ##########################################################################
    #add classifier
    classify = True
    if classify:
        #import pdb;pdb.set_trace()
        modelc = load_classifier(name='mobilenet_v3_large', n=2)  # initialize
        #modelc.load_state_dict(torch.load(classifier_weight, map_location=device)['model_state_dict']).to(device).eval()
        checkpoint = torch.load(classifier_weight, map_location=device)
        modelc.load_state_dict(checkpoint['model_state_dict'])
        modelc.to(device).eval()
    ##################################################################################
    
    img = letterbox(np.array(img0), imgsz, stride=stride)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    with torch.no_grad():
        pred = model(img, augment=augment)[0]
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    ###############################################################################################
    #add classifier
    if classify:
            pred = apply_classifier(pred, modelc, img, img0)
    ################################################################################################
    
    
    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]].to(device)
    print(pred)
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        print(pred)

    '''for pd in pred:
        for i in range(pd.shape[0]):
            bbox = pd[i][:4]
            cv2.rectangle(img0, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
    cv2.imwrite('test2_logo.jpg',img0)'''

    for det in pred:
        for i in range(det.shape[0]):
            #if len(det):
            #det[i, :4] = scale_coords(img.shape[2:], det[i, :4], img0.shape).round()
            det[i,:4] = xyxy2xywh(torch.tensor(det[i,:4]).view(1, 4)) / gn
    
        print(reversed(det))
    #pred[0][:, :4] = scale_coords(img.shape[2:], pred[0][:, :4], img0.shape).round()
    #print(type(pred))
    
    #print(pred)
    
    return pred

if __name__ == "__main__":
    import os
    #img_path = '/data/flag/test_dataset_result/australia/valtest/ffout-10_02471.jpg'
    img_path = 'test2.jpg'
    import pdb;pdb.set_trace()
    img = cv2.imread(img_path)
    run(img0=img)

    '''img_root = '/data/flag/test_dataset_result/australia/valtest'
    imgs = os.listdir(img_root)
    for im in imgs:
        if not im.endswith('jpg'):
            continue    
        img_path = os.path.join(img_root,im)
        print(img_path)
        #import pdb;pdb.set_trace()
        img = cv2.imread(img_path)
        run(img0=img)'''