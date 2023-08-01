# -*- coding: UTF-8 -*-
import argparse
import re
import cv2
import torch
import copy
import numpy as np
from lpr.models2.experimental import attempt_load
from lpr.utils2.datasets import letterbox
from lpr.utils2.general import check_img_size, non_max_suppression_face, scale_coords
from lpr.models2.plate_rec import get_plate_result, init_model, cv_imread
from lpr.models2.double_plate_split_merge import get_split_merge
from pathlib import Path
import sys
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
sys.path.append(str(ROOT / 'lpr'))


clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
danger = ['危', '险']


def is_license_plate(license):
    pattern = r'^[\u4e00-\u9fa5]{1}[A-Z]{1}([A-Z0-9]{5,6})$'
    match = re.match(pattern, license)
    return bool(match)


def four_point_transform(image, pts):                       #透视变换得到车牌小图
    rect = pts.astype('float32')
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def load_model(weights, device):   #加载检测模型
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):  #返回到原图坐标
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7]] -= pad[1]  # y padding
    coords[:, :8] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    return coords


def get_plate_rec_landmark(img, xyxy, conf, landmarks, class_num,device,plate_rec_model,is_color=False):  #获取车牌坐标以及四个角点坐标并获取车牌号
    result_dict={}

    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])

    landmarks_np=np.zeros((4,2))
    rect=[x1,y1,x2,y2]
    for i in range(4):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        landmarks_np[i]=np.array([point_x,point_y])

    class_label= int(class_num)  #车牌的的类型0代表单牌，1代表双层车牌
    roi_img = four_point_transform(img,landmarks_np)   #透视变换得到车牌小图
    if class_label:        #判断是否是双层车牌，是双牌的话进行分割后然后拼接
        roi_img=get_split_merge(roi_img)
    if not is_color:
        plate_number,rec_prob = get_plate_result(roi_img,device,plate_rec_model,is_color=is_color)                 #对车牌小图进行识别
    else:
        plate_number,rec_prob,plate_color,color_conf=get_plate_result(roi_img,device,plate_rec_model,is_color=is_color) 

    result_dict['rect']=rect                      #车牌roi区域
    result_dict['detect_conf']=conf              #检测区域得分
    result_dict['landmarks']=landmarks_np.tolist() #车牌角点坐标
    result_dict['plate_no']=plate_number   #车牌号
    result_dict['rec_conf']=rec_prob   #每个字符的概率
    result_dict['roi_height']=roi_img.shape[0]  #车牌高度
    result_dict['plate_color']=""
    if is_color:
        result_dict['plate_color']=plate_color   #车牌颜色
        result_dict['color_conf']=color_conf    #颜色得分
    result_dict['plate_type']=class_label   #单双层 0单层 1双层
    
    return result_dict


def detect_Recognition_plate(model, orgimg, device,plate_rec_model,img_size,is_color=True):#获取车牌信息
    conf_thres = 0.3      #得分阈值
    iou_thres = 0.5       #nms的iou值   
    dict_list=[]

    img0 = copy.deepcopy(orgimg)
    assert orgimg is not None, 'Image Not Found ' 
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size  

    img = letterbox(img0, new_shape=imgsz)[0]           #检测前处理，图片长宽变为32倍数，比如变为640X640

    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416  图片的BGR排列转为RGB,然后将图片的H,W,C排列变为C,H,W排列

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)
    # pred.shape
    # Out[4]: torch.Size([1, 20160, 15])
    # det.shape
    # Out[5]: torch.Size([4, 14])


    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            det[:, 5:13] = scale_coords_landmarks(img.shape[2:], det[:, 5:13], orgimg.shape).round()

            for j in range(det.size()[0]):
                xyxy = det[j, :4].view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = det[j, 5:13].view(-1).tolist()
                class_num = det[j, 13].cpu().numpy()
                result_dict = get_plate_rec_landmark(orgimg, xyxy, conf, landmarks, class_num, device, plate_rec_model, is_color=is_color)
                dict_list.append(result_dict)
    return dict_list




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model', nargs='+', type=str, default='weights/best.pt', help='model.pt path(s)')  #检测模型
    parser.add_argument('--rec_model', type=str, default='weights/plate_rec_color.pth', help='model.pt path(s)')#车牌识别+颜色识别模型
    parser.add_argument('--is_color',type=bool,default=True,help='plate color')      #是否识别颜色
    parser.add_argument('--image_path', type=str, default=r'C:\Users\Administrator\Desktop\yolo_tracking-8.0\runs\detect\exp\img.png', help='source')     #图片路径
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')  #网络输入图片大小
    parser.add_argument('--output', type=str, default='result', help='source')               #图片结果保存的位置
    parser.add_argument('--video', type=str, default='', help='source')                       #视频的路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                     #使用gpu还是cpu进行识别
    # device =torch.device("cpu")
    opt = parser.parse_args()


    detect_model = load_model(opt.detect_model, device)  #初始化检测模型
    plate_rec_model=init_model(device,opt.rec_model,is_color=opt.is_color)      #初始化识别模型

    img =cv_imread(opt.image_path)


    dict_list=detect_Recognition_plate(detect_model, img, device,plate_rec_model,opt.img_size,is_color=opt.is_color)
    for dict_list in  dict_list:
        print(dict_list['plate_no'], dict_list['plate_color'])

        
