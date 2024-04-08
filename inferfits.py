from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil

import _init_paths

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch
import sys
sys.path.append('/home/cjz/project/cdnkeydet/cdnkeydet/src/lib')


from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from datasets.dataset_factory import dataset_factory
from detectors.detector_factory import detector_factory
# from lib.external.nms import soft_nms

import random

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def recreate_dir(dir_path):
  if os.path.isdir(dir_path):
    shutil.rmtree(dir_path)
  os.mkdir(dir_path)


def test(opt, test_dir):
  random.seed(opt.seed)
  os.environ['PYTHONHASHSEED'] = str(opt.seed)  # 为了禁止hash随机化，使得实验可复现
  np.random.seed(opt.seed)
  torch.manual_seed(opt.seed)
  torch.cuda.manual_seed(opt.seed)
  torch.cuda.manual_seed_all(opt.seed)  # if you are using multi-GPU.
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.enabled = True


  img_name_list = os.listdir(test_dir)

  Dataset = dataset_factory[opt.dataset]
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)
  results = {}

  # from torchsummary import summary
  # summary(detector.model, input_size=(3,256,256))
  # pytorch_total_params = sum(p.numel() for p in detector.model.parameters() if p.requires_grad)

  num_iters = len(img_name_list)
  bar = Bar('{}'.format(opt.exp_id), max=num_iters)
  time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
  avg_time_stats = {t: AverageMeter() for t in time_stats}

  # st_time = time.time()
  w_max, h_max = [0,0]
  for ind, img_name in enumerate(img_name_list):
    img_path = os.path.join(test_dir, img_name)

    # img_id = dataset.images[ind]
    # img_info = dataset.coco.loadImgs(ids=[img_id])[0]
    # img_path = os.path.join(dataset.img_dir, img_info['file_name'])
    if opt.fits:
      img_path = img_path.replace(img_path.split('.')[-1], 'fits')

    ret = detector.run(img_path)
    
    results[img_name] = ret['results']
    w, h = ret['wh']
    w_max = max(w_max, w)
    h_max = max(h_max, h)

    Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
    for t in avg_time_stats:
      avg_time_stats[t].update(ret[t])
      Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
    bar.next()

  # print(time.time() - st_time)

  bar.finish()
  # generate IPD: require test-json only contain 1 big-image!
  run_eval(results, opt.save_dir, img_name_list, thres=0.4)


def run_eval(results, save_dir, img_name_list, wh=[4096,4096], thres=0.4):
  ''' notice: this code require results.json contains only 1 big-image '''
  thres_score = thres
  ipd_path = os.path.join(save_dir, "{}.IPD".format(img_name_list[0].split('__')[0]))
  ipd_file = open(ipd_path, 'w+')
  # mask_path = os.path.join(save_dir, 'predMask')
  # recreate_dir(mask_path)
  out_mask = np.zeros((wh[1],wh[0]))
  out_maskId = np.zeros((wh[1],wh[0]))  # maskId start with 0
  big_img_name = ''
  # # 1. check anns
  for subimg_name, det_img in results.items():
    filename = subimg_name.split('.')[0]
    det_img = det_img[1]
    for det_id in range(len(det_img)):
      det_item = np.squeeze(det_img[det_id, :])
      if float(det_item[-1]) < thres_score:
        break
      else:
        # 2. get bias
        filesplit = filename.split("__")
        big_img_name = filesplit[0]
        bias_x = int(filesplit[-2])
        # bias_y = int(filesplit[-1][1:].split('.')[0])
        bias_y = int(filesplit[-1])

        # 3. write ipd
        st_x, st_y, width, height = [det_item[0], det_item[1],
                                     det_item[2] - det_item[0], det_item[3] - det_item[1]]
        cen_x, cen_y = [st_x + width / 2 + bias_x, st_y + height / 2 + bias_y]
        width_int = int(round(width))
        height_int = int(round(height))
        area = width_int * height_int

        dataline = "{0} {1} {2} {3} {4} {5} {6} {7}\n".format(
          ("%.3f" % cen_x).zfill(8), ("%.3f" % cen_y).zfill(8),
          "%04d" % area,
          "%04d" % width_int, "%04d" % height_int,
          "0000000000",
          "00000.000", "00000.000"
        )
        ipd_file.write(dataline)
        st_x, st_y, end_x, end_y = [cen_x - width / 2, cen_y - height / 2,
                                    cen_x + width / 2, cen_y + height / 2]
        st_x_int = max(0, int(round(st_x)))
        st_y_int = max(0, int(round(st_y)))
        end_x_int = min(wh[0]-1, int(round(end_x)))
        end_y_int = min(wh[1]-1, int(round(end_y)))
        try:
          out_mask[st_y_int:end_y_int+1, st_x_int:end_x_int+1] = 1
          out_maskId[st_y_int:end_y_int + 1, st_x_int:end_x_int + 1] = det_id   # maskId start with 0
        except Exception as e:
          print('draw mask error: [{0}, {1}, {2}, {3}]'.format(st_x_int, st_y_int, end_x_int, end_y_int))
  ipd_file.close()
  # mask_filepath = os.path.join(mask_path, big_img_name + '.png')
  # cv2.imwrite(mask_filepath, out_mask * 255)
  # maskId_filepath = os.path.join(mask_path, big_img_name + '.npy')
  # np.save(maskId_filepath, out_maskId)

if __name__ == '__main__':
  this_dir = os.path.dirname(__file__)
  test_dir = os.path.join(this_dir, '../data/Sat/realdata_extend/test1024/images')

  opt = opts().parse()
  test(opt, test_dir=test_dir)
