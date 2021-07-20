#%matplotlib inline
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
#pylab.rcParams['figure.figsize'] = (10.0, 8.0)
import mmcv
import argparse

import pdb;pdb.set_trace()
annType = ['segm','bbox','keypoints']
annType = annType[1]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
print('Running demo for *%s* results.'%(annType))

#annFile = '/data/flag/flag_coco_finaltest.json'
def main(args):
    annFile = args.gt_file
    cocoGt=COCO(annFile)

    #initialize COCO detections api
    #resFile='%s/results/%s_%s_fake%s100_results.json'
    #resFile = resFile%(dataDir, prefix, dataType, annType)
    resFile = args.dt_file
    cocoDt=cocoGt.loadRes(resFile)

    # running evaluation
    cocoEval = COCOeval(cocoGt,cocoDt,annType)

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

def parse_opt():
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--gt-file', type=str, default='/data/flag/flag_coco_finaltest_update.json', help='dataset.yaml path')
    parser.add_argument('--dt-file', type=str, default='/data/Detection_proj/yolov5-master/runs/test/final_test_expand_conf5/best_predictions.json', help='dataset.yaml path')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
