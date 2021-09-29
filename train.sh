#export CUDA_VISIBLE_DEVICES=0,1,2,3&&python tools/train.py -f exps/erosive_ulcer_mix/yolox_x_erosive_ulcer_mix_640_15_07.py -d 4 -b 16 --fp16 -c ./YOLOX_outputs/yolox_x_erosive_ulcer_mix_640_15_07/latest_ckpt.pth --resume -e 69
#export CUDA_VISIBLE_DEVICES=0,1,2,3&&python tools/train.py -f exps/erosive_ulcer_mix/yolox_x_erosive_ulcer_mix_640_25_07.py -d 4 -b 16 --fp16 -c ./pre_weights/yolox_x.pth
#export CUDA_VISIBLE_DEVICES=0,1,2,3&&python tools/train.py -f exps/erosive_ulcer_mix/yolox_x_erosive_ulcer_mix_640_20_08.py -d 4 -b 16 --fp16 -c ./pre_weights/yolox_x.pth
#export CUDA_VISIBLE_DEVICES=0,1,2,3&&python tools/train.py -f exps/erosive_ulcer_mix/yolox_x_erosive_ulcer_mix_640_20_06.py -d 4 -b 16 --fp16 -c ./pre_weights/yolox_x.pth
#export CUDA_VISIBLE_DEVICES=0,1,2,3&&python tools/train.py -f exps/erosive_ulcer_mix/yolox_x_erosive_ulcer_mix_640_20_09.py -d 4 -b 16 --fp16 -c ./pre_weights/yolox_x.pth
#export CUDA_VISIBLE_DEVICES=0,1,2,3&&python tools/train.py -f exps/erosive_ulcer_mix/yolox_x_erosive_ulcer_mix_640_20_10.py -d 4 -b 16 --fp16 -c ./pre_weights/yolox_x.pth
#export CUDA_VISIBLE_DEVICES=0,1,2,3&&python tools/train.py -f exps/erosive_ulcer_mix/yolox_x_erosive_ulcer_mix_640_20_05.py -d 4 -b 16 --fp16 -c ./pre_weights/yolox_x.pth
#export CUDA_VISIBLE_DEVICES=0,1,2,3&&python tools/train.py -f exps/erosive_ulcer_mix/yolox_x_erosive_ulcer_mix_640_20_075.py -d 4 -b 16 --fp16 -c ./pre_weights/yolox_x.pth
#export CUDA_VISIBLE_DEVICES=0,1,2,3&&python tools/train.py -f exps/erosive_ulcer_mix/yolox_x_erosive_ulcer_mix_640_20_065.py -d 4 -b 16 --fp16 -c ./pre_weights/yolox_x.pth
#export CUDA_VISIBLE_DEVICES=0,1,2,3&&python tools/train.py -f exps/erosive_ulcer_mix/yolox_x_erosive_ulcer_mix_640_20_07.py -d 4 -b 16 --fp16 -c ./pre_weights/yolox_x.pth

#export CUDA_VISIBLE_DEVICES=0,1,2,3&&python tools/train.py -f exps/erosive_ulcer_mix/yolox_x_erosive_ulcer_mix_640_25_085.py -d 4 -b 16 --fp16 -c ./pre_weights/yolox_x.pth
#export CUDA_VISIBLE_DEVICES=0,1,2,3&&python tools/train.py -f exps/erosive_ulcer_mix/yolox_x_erosive_ulcer_mix_640_15_085.py -d 4 -b 16 --fp16 -c ./pre_weights/yolox_x.pth
#export CUDA_VISIBLE_DEVICES=0,1,2,3&&python tools/train.py -f exps/erosive_ulcer_mix/yolox_x_erosive_ulcer_mix_640_20_085.py -d 4 -b 16 --fp16 -c ./pre_weights/yolox_x.pth
#export CUDA_VISIBLE_DEVICES=0,1,2,3&&python tools/train.py -f exps/erosive_ulcer_mix/yolox_x_erosive_ulcer_mix_640_20_085_ap50_95.py -d 4 -b 16 --fp16 -c ./pre_weights/yolox_x.pth
#export CUDA_VISIBLE_DEVICES=0,1,2,3&&python tools/train.py -f exps/erosive_ulcer_mix/yolox_x_erosive_ulcer_mix_640_20_085_no_use_l1.py -d 4 -b 16 --fp16 -c ./pre_weights/yolox_x.pth

python tools/train.py -f exps/polyp/polyp_yolox_x_erosive_ulcer_mix_320_adjust.py -d 2 -b 16 --fp16 -c ./pre_weights/yolox_x.pth
python tools/train.py -f exps/polyp/polyp_yolox_x_erosive_ulcer_mix_320_adjust1.py -d 2 -b 16 --fp16 -c ./pre_weights/yolox_x.pth