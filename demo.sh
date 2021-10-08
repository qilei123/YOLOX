#python tools/demo.py image -f exps/erosive_ulcer_mix/yolox_x_erosive_ulcer_mix.py -c YOLOX_outputs/yolox_x_erosive_ulcer_mix/best_ckpt.pth.tar --path ./datasets/gastric_object_detection/images/07e40950-e41d-4fb8-9a36-3dc15b63bb75.jpg --conf 0.2 --nms 0.1 --tsize 640 --save_result --device gpu
python tools/demo.py video \
    -f exps/trans_drone/yolox_x_trans_drone_mix_960.py \
    -c YOLOX_outputs/yolox_x_trans_drone_mix_960/best_ap50_95_ckpt.pth \
    --path "/data2/qilei_chen/DATA/trans_drone/videos/whole_rounds/DJI_0601 Wide View 400.MOV" \
    --tsize 960 --save_result --device gpu

