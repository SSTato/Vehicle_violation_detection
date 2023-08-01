import argparse

import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import datetime
import time
import torch
import requests
import json
import setproctitle
import base64
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'lpr') not in sys.path:
    sys.path.append(str(ROOT / 'lpr'))  # add lpr ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_boxes, check_requirements, cv2,
                                  check_imshow, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync, within_time_list
from yolov5.utils.plots import Annotator, colors, save_one_box
from trackers.multi_tracker_zoo import create_tracker
from lpr.detect_plate import load_model, init_model, detect_Recognition_plate, is_license_plate
# remove duplicated stream handler to avoid duplicated logging
#logging.getLogger().removeHandler(logging.getLogger().handlers[0])


@torch.no_grad()
def run(
        source='0',
        yolo_weights=ROOT / 'yolov5s.pt',  # model.pt path(s),
        reid_weights=ROOT / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        lp_weights='lpr/weights/best.pt',
        lpr_weights=ROOT / 'lpr/weights/plate_rec_color.pth',
        tracking_method='strongsort',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=True,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        p_save_address='runs/p_save_address',  # Alarm image storage address
        a_save_address='runs/a_save_address',  # Alarm information storage address
        host="http://test.superton.cn",
        url_path="/server/superton-iot-server-zjf/export/push/alarm",
        interval_time=0.1,  # Detection frame interval time
        cycle_time=3.0,  # Cycle time of alarm
        parking_duration=600,  # Time for parking timeout alarm
        disappearance_time=60,  # Time for parking disappearance and deletion
        in_time_list=[[0, 24], [0, 24], [0, 24], [0, 24], [0, 24], [0, 24], [0, 24]],  # Allow detection time list
        # specified_points=np.array([(0.006250, 0.918750), (0.838281, 0.309375), (0.848437, 0.247917),
        #                            (0.913281, 0.260417), (0.822656, 0.412500), (0.680469, 0.521875),
        #                            (0.603906, 0.576042), (0.515625, 0.684375), (0.499219, 0.734375),
        #                            (0.463281, 0.883333), (0.443750, 0.989583), (0.012500, 0.991667)]),
        specified_points=np.array([(0.008594, 0.904167), (0.850781, 0.254167), (0.928906, 0.267708),
                                   (0.450781, 0.984375), (0.017969, 0.991667)]),

):

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + reid_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    p_save_address = increment_path(Path(p_save_address), exist_ok=exist_ok)  # increment run
    a_save_address = increment_path(Path(a_save_address), exist_ok=exist_ok)  # increment run
    # (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Load lprmodel
    detect_model = load_model(lp_weights, device)  # 初始化检测模型
    plate_rec_model = init_model(device, lpr_weights, is_color=True)  # 初始化识别模型
    # imgsss = cv2.imread(r'C:\Users\Administrator\Desktop\yolo_tracking-8.0\runs\detect\exp\img.png')
    # dict_list = detect_Recognition_plate(detect_model, imgsss, device, plate_rec_model, imgsz[0])
    # print(dict_list)

    # print(lpr(lp_weights, lpr_weights, device, imgsss, imgsz[0]))

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride, specified_points=specified_points)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(nr_sources):
        tracker = create_tracker(tracking_method, reid_weights, device, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * nr_sources
    vehicle_info = {}
    time_list = [time_sync(), time_sync()]  # 时间列表
    alarm_list = []
    # Run tracking
    #model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        # 每小时重新发送告警失败信息
        if alarm_list and time.localtime().tm_min == 0 and time.localtime().tm_sec == 0:
            response = requests.post(host + url_path, json=alarm_list, headers={"Referer": host})
            if response.status_code == 200:
                alarm_list.clear()
                print("警报信息发送成功")

        if within_time_list(in_time_list):
            current_times = time_sync()
            if current_times - time_list[0] > (interval_time + 0.001):  # 每隔1s推理一次
                time_list[0] = current_times

                t1 = time_sync()
                im = torch.from_numpy(im).to(device)
                im = im.half() if half else im.float()  # uint8 to fp16/32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                t2 = time_sync()
                dt[0] += t2 - t1

                # Inference
                visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)
                t3 = time_sync()
                dt[1] += t3 - t2

                # Apply NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                dt[2] += time_sync() - t3

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    seen += 1
                    if webcam:  # nr_sources >= 1
                        p, im0, _, im01 = path[i], im0s[i].copy(), dataset.count, im0s[i].copy()
                        p = Path(p)  # to Path
                        s += f'{i}: '
                        txt_file_name = p.name
                        save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                    else:
                        p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                        p = Path(p)  # to Path
                        # video file
                        if source.endswith(VID_FORMATS):
                            txt_file_name = p.stem
                            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                        # folder with imgs
                        else:
                            txt_file_name = p.parent.name  # get folder name containing current img
                            save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
                    curr_frames[i] = im0

                    txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
                    s += '%gx%g ' % im.shape[2:]  # print string
                    imc = im0.copy() if save_crop else im0  # for save_crop

                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                    if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                        if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                            tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # xyxy

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # pass detections to strongsort
                        t4 = time_sync()
                        outputs[i] = tracker_list[i].update(det.cpu(), im0)
                        t5 = time_sync()
                        dt[3] += t5 - t4

                        # draw boxes for visualization
                        if len(outputs[i]) > 0:
                            for j, (output, conf) in enumerate(zip(outputs[i], det[:, 4])):
                                im2 = im01.copy()
                                annotator_id = Annotator(im2, line_width=line_thickness, example=str(names))
                                bboxes = output[0:4]
                                id = output[4]
                                cls = output[5]
                                if id not in vehicle_info:
                                    vehicle_info[id] = (time_sync(), time_sync(), 0)
                                else:
                                    first_time, _, total_timeout = vehicle_info[id]
                                    vehicle_info[id] = (first_time, time_sync(), total_timeout)

                                (f_time, l_time, t_time) = vehicle_info[id]
                                elapsed_time = l_time - f_time
                                if elapsed_time > parking_duration:
                                    if t_time == 0:
                                        annotator_id.box_label(bboxes, None, color=colors(c, True))
                                        im1 = annotator_id.result()  # save detect images
                                        image_bytes = cv2.imencode('.jpg', im1)[1].tobytes()
                                        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                                        now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
                                        p_save_path = p_save_address / datetime.datetime.now().strftime("%Y-%m/%d")
                                        if not os.path.exists(p_save_path):
                                            p_save_path.mkdir(parents=True, exist_ok=True)
                                        pic_save_path = str(p_save_path / f'{now_time}.jpg')
                                        cv2.imwrite(pic_save_path, im1)

                                        dict_list = detect_Recognition_plate(detect_model, save_one_box(
                                            bboxes, imc, BGR=True, save=False), device, plate_rec_model, imgsz[0])
                                        if len(dict_list):
                                            if is_license_plate(dict_list[0]['plate_no']):
                                                license_plate = dict_list[0]['plate_no'] + dict_list[0]['plate_color']
                                            else:
                                                license_plate = '暂无车牌信息'
                                        else:
                                            license_plate = '暂无车牌信息'
                                        a_save_path = a_save_address / datetime.datetime.now().strftime("%Y-%m")
                                        if not os.path.exists(a_save_path):
                                            a_save_path.mkdir(parents=True, exist_ok=True)
                                        ala_save_path = str(a_save_path / f'{datetime.datetime.now().day}.json')
                                        data = {"timestamp": now_time, "msg": "alarm", "flag": "200",
                                                "alarm_type": "illegal_parking",
                                                "save_type": "picture", "pic_addr": pic_save_path, "license_plate": license_plate}
                                        # 以追加模式打开文件，并将数据写入
                                        with open(ala_save_path, "a") as file:
                                            json.dump(data, file, indent=4, ensure_ascii=False)
                                            file.write(",\n")  # 换行分隔每条数据
                                        data['image_base64'] = image_base64
                                        # response = requests.post(host + url_path, json=data, headers={"Referer": host})
                                        # if response.status_code == 200:
                                        #     print("警报信息发送成功")
                                        # else:
                                        #     alarm_list.append(data)
                                        #     print("警报信息发送失败")
                                        print(f"车辆 {id} 超时，持续时间：{elapsed_time}秒")
                                    t_time += elapsed_time
                                    vehicle_info[id] = (f_time, l_time, t_time)
                                if time_sync() - l_time > disappearance_time:
                                    if t_time > 0:
                                        annotator_id.box_label(bboxes, None, color=colors(c, True))
                                        im1 = annotator_id.result()  # save detect images
                                        image_bytes = cv2.imencode('.jpg', im1)[1].tobytes()
                                        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                                        now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
                                        p_save_path = p_save_address / datetime.datetime.now().strftime("%Y-%m/%d")
                                        if not os.path.exists(p_save_path):
                                            p_save_path.mkdir(parents=True, exist_ok=True)
                                        pic_save_path = str(p_save_path / f'{now_time}.jpg')
                                        cv2.imwrite(pic_save_path, im1)

                                        dict_list = detect_Recognition_plate(detect_model, save_one_box(
                                            bboxes, imc, BGR=True, save=False), device, plate_rec_model, imgsz[0])
                                        if len(dict_list):
                                            if is_license_plate(dict_list[0]['plate_no']):
                                                license_plate = dict_list[0]['plate_no'] + dict_list[0]['plate_color']
                                            else:
                                                license_plate = '暂无车牌信息'
                                        else:
                                            license_plate = '暂无车牌信息'
                                        a_save_path = a_save_address / datetime.datetime.now().strftime("%Y-%m")
                                        if not os.path.exists(a_save_path):
                                            a_save_path.mkdir(parents=True, exist_ok=True)
                                        ala_save_path = str(a_save_path / f'{datetime.datetime.now().day}.json')
                                        data = {"timestamp": now_time, "msg": "alarm", "flag": "200",
                                                "alarm_type": "illegal_parking",
                                                "save_type": "picture", "pic_addr": pic_save_path, "license_plate": license_plate}
                                        # 以追加模式打开文件，并将数据写入
                                        with open(ala_save_path, "a") as file:
                                            json.dump(data, file, indent=4, ensure_ascii=False)
                                            file.write(",\n")  # 换行分隔每条数据
                                        data['image_base64'] = image_base64
                                        # response = requests.post(host + url_path, json=data, headers={"Referer": host})
                                        # if response.status_code == 200:
                                        #     print("警报信息发送成功")
                                        # else:
                                        #     alarm_list.append(data)
                                        #     print("警报信息发送失败")
                                        print(f"车辆 {id} 消失超过 {disappearance_time} 秒，总超时时长：{t_time}秒")
                                    vehicle_info.pop(id)

                                if save_txt:
                                    # to MOT format
                                    bbox_left = output[0]
                                    bbox_top = output[1]
                                    bbox_w = output[2] - output[0]
                                    bbox_h = output[3] - output[1]
                                    # Write MOT compliant results to file
                                    with open(txt_path + '.txt', 'a') as f:
                                        f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                                       bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                                if save_vid or save_crop or show_vid:  # Add bbox to image
                                    c = int(cls)  # integer class
                                    id = int(id)  # integer id
                                    label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                        (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                                    annotator.box_label(bboxes, label, color=colors(c, True))
                                    if save_crop:
                                        txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                        save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                        LOGGER.info(f'{s}Done. yolo:({t3 - t2:.3f}s), {tracking_method}:({t5 - t4:.3f}s)')

                    else:
                        #strongsort_list[i].increment_ages()
                        LOGGER.info('No detections')

                    # Stream results
                    im0 = annotator.result()

                    if show_vid:
                        cv2.imshow(str(p), im0)
                        cv2.waitKey(1)  # 1 millisecond

                    # Save results (image with detections)
                    if save_vid:
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)

                    prev_frames[i] = curr_frames[i]

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=ROOT / 'weights/yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=Path, default=r'C:\Users\Administrator\Desktop\yolo_tracking-8.0\trackers\strong_sort\deep\checkpoint\osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking-method', type=str, default='strongsort', help='strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='rtsp://admin:aa888888@192.168.1.206//Streaming/Channels/1', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_false', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', default=2, nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_false', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=True, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
