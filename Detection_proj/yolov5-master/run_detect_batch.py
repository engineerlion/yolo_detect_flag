import os
video_names = [
    #'a0021so808b',
    #'b00219wkkjc',
    #'c0021j4eqth',
    #'c002160nqjr',
    #'e00213twop9',
    #'f00215tk67q',
    #'h0021sst65l',
    'i00211vjemz',
    'i09381b3eeg'
]
save_dir = '/data/Detection_proj/yolov5-master/runs/detect'
for video in video_names:
    save_path = os.path.join(save_dir,'{}'.format(video))
    if os.path.exists(save_path):
        continue
    commd = 'python detect.py --weights runs/train/exp/weights/best.pt --source /data/flag/test_videos/{}.mp4 --save-txt --name {}'.format(video,video)
    print(commd)
    os.system(commd)


#'f00215tk67q',
#'h0021sst65l',
#'c002160nqjr',
#'i09381b3eeg'