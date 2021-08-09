import sys
import time
from pathlib import Path

import cv2
import torch

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords, increment_path
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device


def is_within_detection_area(xyxy, c1, c2):
    return xyxy[0] > c1[0] and xyxy[1] > c1[1] and xyxy[2] < c2[0] and xyxy[3] < c2[1]


@torch.no_grad()
def run(detect_area_c1,
        detect_area_c2,
        detect_display_function,
        weights,
        source,
        imgsz=640,
        conf_thres=0.25,
        iou_thres=0.45,
        project='tara'
        ):
    save_dir = increment_path(Path(project) / 'inference', exist_ok=True)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    device = select_device(0)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    vid_path, vid_writer = [None] * 1, [None] * 1
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    for path, img, im0s, vid_cap in dataset:
        detect_display_text = ""
        detect_display_color = [255, 255, 255]

        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=False, visualize=False)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=100)

        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=3)

                    if detect_display_function(c, xyxy) is not None:
                        detect_display_text, detect_display_color = detect_display_function(c, xyxy)

            cv2.putText(im0, detect_display_text, (130, 120), 0, 4, detect_display_color, thickness=5,
                        lineType=cv2.LINE_AA)
            cv2.rectangle(im0, detect_area_c1, detect_area_c2, color=(128, 128, 128), thickness=3, lineType=cv2.LINE_AA)

            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            else:  # 'video' or 'stream'
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)
