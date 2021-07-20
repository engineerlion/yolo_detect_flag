python detect.py --weights runs/train/exp/weights/best.pt --source /data/flag/test_videos/w0021n2d80c.mp4 
python test.py --weights runs/train/exp/weights/best.pt --data flag.yaml --img 640 --task test --name final_test_expand_conf --conf-thres 0.25
python export.py --weights runs/train/exp/weights/best.pt --include onnx

python detect.py --weights runs/train/exp/weights/best.pt --source /data/flag/test_dataset_result/australia/valtest --name AUS_valtest

python test.py --weights runs/train/exp/weights/best.pt --data flag.yaml --img 640 --task test --name final_test_expand_conf --conf-thres 0.1 --save-json --coco-path /data/flag/flag_coco_finaltest_update.json

python test.py --weights runs/train/exp/weights/best.pt --data flag.yaml --img 640 --task test --name final_test_conf --conf-thres 0.1 --save-json --coco-path /data/flag/flag_coco_test.json

python test.py --weights runs/train/exp/weights/best.pt --data flag.yaml --img 640 --task test --name train_conf --conf-thres 0.1 --save-json --coco-path /data/flag/flag_coco_train_all.json

python test.py --weights runs/train/exp/weights/best.pt --data flag.yaml --img 640 --task test --name val_conf --conf-thres 0.1 --save-json --coco-path /data/flag/flag_coco_val_all.json

python test.py --weights runs/train/exp/weights/best.pt --data flag.yaml --img 640 --task test --name how_test_work --conf-thres 0.1 --save-json --coco-path /data/flag/flag_coco_val_all.json

nohup python -u train.py --data flag.yaml --img 640 --weights yolov5m.pt >0715_yolov5m.out 2>&1 &


python detect.py --weights runs/train/exp/weights/best.pt --source /data/flag/olympic --name olympic --hide-labels --conf-thres 0.4

python detect.py --weights runs/train/exp/weights/best.pt --source /data/flag/test_videos/c0021j4eqth.mp4 --save-txt --name c0021j4eqth

python detect.py --weights runs/train/exp/weights/best.pt --source /data/Detection_proj/mmdetection/result.mp4 --name result

python test.py --weights runs/train/exp/weights/best.pt --source /data/flag/test_dataset_result/australia/valtest --name AUS_valtest

python detect.py \
--weights runs/train/exp/weights/best.pt \
--classifier-weight /data/Detection_proj/resnet18_flag/weights/0719_mobilenet/best.pt \
--source /data/flag/olympic \
--name olympic_logo

python detect.py \
--weights runs/train/exp/weights/best.pt \
--classifier-weight /data/Detection_proj/resnet18_flag/weights/0719_mobilenet/best.pt \
--source /data/flag/olympic/11126_1625855927999603.5.jpeg \
--name olympic_logo

python detect.py --weights runs/train/exp/weights/best.pt --source /data/flag/olympic --name olympic_logo --save-txt

a0021so808b
b00219wkkjc
c0021j4eqth
c002160nqjr
e00213twop9
f00215tk67q
h0021sst65l
i00211vjemz
i09381b3eeg

