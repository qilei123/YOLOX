from demo import Predictor
from pycocotools.coco import COCO

from metric_polyp_multiclass import MetricMulticlass

import cv2
import os
import torch

from yolox.data.data_augment import preproc
from yolox.data.datasets import COCO_CLASSES,Erosive_Ulcer
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

def eval_erosive_ulcer(dataset_dir):
    exp_file = "exps/erosive_ulcer_mix/yolox_x_erosive_ulcer_mix_512.py"
    exp = get_exp(exp_file, None)
    exp.test_conf = 0.05
    exp.nmsthre = 0.01
    model = exp.get_model()
    model.cuda()
    model.eval()

    ckpt_file = "YOLOX_outputs/yolox_x_erosive_ulcer_mix_512/best_ckpt.pth.tar"
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    predictor = Predictor(model, exp, device="gpu")

    coco_instance = COCO(os.path.join(dataset_dir,"annotations","test_mix.json"))
    coco_imgs = coco_instance.imgs

    for img_id in coco_imgs:
        img_name = coco_imgs[img_id]["file_name"]
        img_dir = os.path.join(dataset_dir,"images",img_name)
        outputs, img_info = predictor.inference(img_dir)
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        
        annIds = coco_instance.getAnnIds(imgIds=coco_imgs[img_id]['id'])
        anns = coco_instance.loadAnns(annIds)
        for ann in anns:
            [x, y, w, h] = ann['bbox']
            # print(ann['category_id'])
            cv2.putText(result_image, Erosive_Ulcer[ann['category_id']-1], (int(x), int(
                y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)
            cv2.rectangle(result_image, (int(x), int(y)), (int(x+w),
                                                    int(y+h)), (0,255,0), 1)

        cv2.imwrite("YOLOX_outputs/yolox_x_erosive_ulcer_mix_512/vis_results/"+img_name,result_image)

if __name__ == "__main__":
    eval_erosive_ulcer("datasets/gastric_object_detection/")