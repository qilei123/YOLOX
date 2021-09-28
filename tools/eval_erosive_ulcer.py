from demo import Predictor
from pycocotools.coco import COCO

from metric_polyp_multiclass import MetricMulticlass

import cv2
import os
import torch
import glob

from yolox.data.data_augment import preproc
from yolox.data.datasets import COCO_CLASSES,Erosive_Ulcer,Erosive_Ulcer3
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

from img_crop import *

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

def xywh2xyxy(box):
    return [int(box[0]), int(box[1]), int(box[2]+box[0]), int(box[3]+box[1])]

def anns2gtboxes(gtanns,categories=[1,2]):
    gtboxes = []
    for ann in gtanns:
        if ann['category_id'] in categories:
            xyxy = xywh2xyxy(ann['bbox'])
            xyxy.append( ann['category_id'])
            gtboxes.append(xyxy)
    return gtboxes

def get_eval_outputs(output,ratio):
    eval_outputs = []
    
    if output is None:
        return eval_outputs
    output = output.cpu()
    output = output.numpy()
    bboxes = output[:, 0:4]

    # preprocessing: resize
    bboxes /= ratio

    classes = output[:,6]
    
    for bbox,cls in zip(bboxes,classes):
        bbox = bbox.tolist()
        #print(bbox)
        bbox.append(cls+1)
        eval_outputs.append(bbox)
    #print(eval_outputs)
    return eval_outputs

def eval_erosive_ulcer(dataset_dir,confg_name = "yolox_x_erosive_ulcer_mix_512",param_file = "best_ckpt.pth",score = 0.01,vis = False):
    print(confg_name)
    exp_file = "exps/erosive_ulcer_mix3/"+confg_name+".py"
    exp = get_exp(exp_file, None)
    exp.test_conf = score
    exp.nmsthre = 0.1
    model = exp.get_model()
    model.cuda()
    model.eval()

    ckpt_file = "YOLOX_outputs/"+confg_name+"/"+param_file
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    predictor = Predictor(model, exp, device="gpu")

    #eval_m = MetricMulticlass(visualize=True,visualization_root="/data1/qilei_chen/DATA/erosive_ulcer_mix/work_dirs/retinanet_free_anchor_r50_fpn_1x_coco_512/epoch_13.pth_test.pkl_result_0.5/")
    eval_m = MetricMulticlass(classes=('erosive', 'ulcer','other'))

    coco_instance = COCO(os.path.join(dataset_dir,"annotations","test.json"))
    coco_imgs = coco_instance.imgs

    for img_id in coco_imgs:
        img_name = coco_imgs[img_id]["file_name"]
        img_dir = os.path.join(dataset_dir,"images",img_name)
        #if "00b04d25-1db7-4223-8180-8f3df2c46d05" in img_name:
        if True:
            gtannIds = coco_instance.getAnnIds(imgIds=img_id)
            gtanns = coco_instance.loadAnns(gtannIds)
            gtboxes = anns2gtboxes(gtanns)

            outputs, img_info = predictor.inference(img_dir)
            #print(outputs)
            eval_outputs = get_eval_outputs(outputs[0],img_info["ratio"])
            #print(eval_outputs)
            #eval_m.eval_add_result(gtboxes, filed_boxes,image=image,image_name=coco_instance.imgs[img_id]["file_name"])
            eval_m.eval_add_result(gtboxes, eval_outputs)
            if vis:
                #result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
                result_image = predictor.visual(None, img_info, predictor.confthre)
                #annIds = coco_instance.getAnnIds(imgIds=coco_imgs[img_id]['id'])
                #anns = coco_instance.loadAnns(annIds)

                for pre_box in eval_outputs:
                    cv2.putText(result_image, Erosive_Ulcer[int(pre_box[4])-1], (int(pre_box[0]), int(
                        pre_box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
                    cv2.rectangle(result_image, (int(pre_box[0]), int(pre_box[1])), (int(pre_box[2]),
                                                            int(pre_box[3])), (0,0,255), 1)                

                for ann in gtanns:
                    [x, y, w, h] = ann['bbox']
                    # print(ann['category_id'])
                    cv2.putText(result_image, Erosive_Ulcer[ann['category_id']-1], (int(x), int(
                        y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)
                    cv2.rectangle(result_image, (int(x), int(y)), (int(x+w),
                                                            int(y+h)), (0,255,0), 1)

                cv2.imwrite("YOLOX_outputs/"+confg_name+"/vis_results/"+img_name,result_image)
            
    category = eval_m.classes
    evaluation = eval_m.get_result()
    for key in evaluation:
        if key in ['overall', 'binary']:
            print('\n==================== {} ====================='.format(key))
        elif key == 'confusion_matrix':
            continue
        else:
            print('\n==================== {} ====================='.format(
                category[key - 1]))
        print("Precision: {:.4f}  Recall: {:.4f}  F1: {:.4f}  F2: {:.4f}  "
            "TP: {:3}  FP: {:3}  FN: {:3}  FP+FN: {:3}"
            .format(evaluation[key]['precision'],
                    evaluation[key]['recall'],
                    evaluation[key]['F1'],
                    evaluation[key]['F2'],
                    evaluation[key]['TP'],
                    evaluation[key]['FP'],
                    evaluation[key]['FN'],
                    evaluation[key]['FN'] + evaluation[key]['FP']))
    template = "{:^20}"
    out_t = ''
    out = []
    for i in range(len(category) + 1):
        out_t = out_t + template
    out.append(out_t.format('\n  gt class -->', *[i for i in category]))
    total_proposal = 0
    for i in range(1, len(category) + 1):
        cm = []
        for j in range(1, len(category) + 1):
            cm.append(evaluation['confusion_matrix'][i][j])
            total_proposal += evaluation['confusion_matrix'][i][j]
        out.append(out_t.format(category[i - 1], *cm))
    print('\n'.join(out))
    print('\nTotal proposals: {}, Accuracy: {:.4f}'.format(total_proposal,
                                                        (evaluation['confusion_matrix'][1][1] +
                                                            evaluation['confusion_matrix'][2][2]) / total_proposal))

def evaluation():
    score_list = [i*0.01 for i in range(10,30)]
    for score in score_list:
        print("----------"+str(score)+"-----------")
        #eval_erosive_ulcer("datasets/gastric_object_detection/","yolox_x_erosive_ulcer_mix_640_20_085_no_use_l1",param_file="best_ckpt496_350.pth",score=score)
        eval_erosive_ulcer("/home/qilei/DATASETS/erosive_ulcer_mix/","yolox_x_erosive_ulcer_mix3_512",param_file="best_ap50_95_ckpt.pth",score=score)


def process_videos(video_dir_list,exp_file_dir,ckpt_file_dir,thresh = 0.2):

    exp = get_exp(exp_file_dir, None)
    exp.test_conf = thresh
    exp.nmsthre = 0.1
    model = exp.get_model()
    model.cuda()
    model.eval()

    ckpt = torch.load(ckpt_file_dir, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    predictor = Predictor(model, exp, device="gpu")

    for video_dir in video_dir_list:
        
        cap = cv2.VideoCapture(video_dir)
        success, frame = cap.read()
        
        while success:

            frame,roi = crop_img(frame)

            outputs, img_info = predictor.inference(frame)
            
            print(outputs)
            print(img_info)
            print(roi)
            cv2.imwrite("/home/qilei/DATASETS/erosive_ulcer_mix/test.jpg",frame)
            success, frame = cap.read()            


def evaluation_videos():

    video_dir = "/home/qilei/DATASETS/erosive_ulcer_mix/videos/"
    video_list = glob.glob(os.path.join(video_dir,"*.avi"))
    
    exp_file_dir = "exps/erosive_ulcer_mix3/yolox_x_erosive_ulcer_mix3_512.py"
    ckpt_file_dir = "YOLOX_outputs/yolox_x_erosive_ulcer_mix3_512/best_ap50_95_ckpt.pth"

    process_videos(video_list,exp_file_dir,ckpt_file_dir,0.23)


if __name__ == "__main__":
    #eval_erosive_ulcer("datasets/gastric_object_detection/","yolox_x_erosive_ulcer_mix_412",0.15)
    #evaluation()
    evaluation_videos()

    