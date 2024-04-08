from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import sys

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian, draw_umich_gaussian_weighted
from utils.image import draw_dense_reg
import math
from astropy.io import fits

import pycocotools.coco as coco
import datetime

class LocDataset(data.Dataset):
  def __init__(self):
    self.predHeatmapPath = './src/lib/datasets/sample/heatmap_assign_ms3_clsoff_{}'.format(str(datetime.datetime.now()).replace(' ', '_').replace(':', '_')[:19])
    self.predOffsetPath = './src/lib/datasets/sample/offset_assign_ms3_clsoff_{}'.format(str(datetime.datetime.now()).replace(' ', '_').replace(':', '_')[:19])
    recreate_dir(self.predHeatmapPath)
    recreate_dir(self.predOffsetPath)
    self.pred_hm = None
    self.pred_off = None

  def _coco_box_to_bbox(self, box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def locAwareAssign(self, centroid, ms_levels, file_name):
    score_list = np.zeros((ms_levels,))

    for ct_assign_level in range(ms_levels):
      scale_factor = pow(2, ct_assign_level)
      ct_l = centroid.astype(np.float32) / scale_factor   # current level centroid
      ct_int_l = np.floor(ct_l)    # current level centroid-index

      ''' pred location decoding '''
      # hmpath = os.path.join('./lib/datasets/sample/heatmap',
      #                       '{}____hm____{}.npy'.format(file_name.replace('.fits', ''), ct_assign_level))
      # offpath = os.path.join('./lib/datasets/sample/offset',
      #                         '{}____off____{}.npy'.format(file_name.replace('.fits', ''), ct_assign_level))
      # try:
      #   pred_hm = np.load(hmpath)
      #   pred_off = np.load(offpath)
      # except Exception as e:
      #   return np.ones((ms_levels,))

      if self.pred_off is None:
        return np.ones((ms_levels,))
      else:
        pred_hm = self.pred_hm[ct_assign_level]
        pred_off = self.pred_off[ct_assign_level]
        x_int = int(ct_int_l[0])
        y_int = int(ct_int_l[0])
        x_int = max(0,min(x_int,pred_hm.shape[1]-1))
        y_int = max(0,min(y_int,pred_hm.shape[0]-1))
        gt_off = np.array((ct_l[0] - ct_int_l[0],ct_l[1] - ct_int_l[1]))
        try:
          ''' method -1: cls*off '''
          hmvalue = float(torch.clamp(torch.sigmoid(torch.tensor(pred_hm[y_int,x_int])), min=1e-4, max=1-1e-4))
          loc_quality = hmvalue * math.exp(-np.linalg.norm((gt_off-pred_off[:,y_int,x_int])*scale_factor))

          ''' method -2: off '''
          # loc_quality = math.pow(math.exp(-np.linalg.norm((gt_off-pred_off[:,y_int,x_int])*scale_factor)),2)
        except Exception as e:
          loc_quality = 0.1
          print(e)

        score_list[ct_assign_level] = loc_quality

    score_sum = np.sum(np.array(score_list))
    score_list = list(map(lambda x: x / score_sum * ms_levels, score_list))
    return score_list

  def __getitem__(self, index):
    # # reset self.coco and self.images
    # bigimage_index = index // self.num_samples_per_image
    # subimage_index = index % self.num_samples_per_image
    # self.coco = coco.COCO(os.path.join(self.annot_dir, self.annot_jsonlist[bigimage_index]))
    # self.images = self.coco.getImgIds()
    # img_id = self.images[subimage_index]

    img_id = self.images[index]
    file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
    if self.opt.fits:
      if file_name.split('.')[-1] != 'fits':
        file_name = file_name.replace(file_name.split('.')[-1], 'fits')
    img_path = os.path.join(self.img_dir, file_name)
    file_name_mask = file_name if file_name.split('.')[-1] == 'png' else file_name.replace(file_name.split('.')[-1], 'png')
    mask_path = os.path.join(self.mask_dir, file_name_mask) if self.opt.mask else None
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = min(len(anns), self.max_objs)

    if self.opt.task=='loc' and self.opt.use_ms and self.opt.use_cwassign:
      try:
        pred_hm_list = []
        pred_off_list = []
        for ct_assign_level in range(self.opt.ms_levels):
          hmpath = os.path.join(self.predHeatmapPath,
                                '{}____hm____{}.npy'.format(file_name.replace('.fits', ''), ct_assign_level))
          offpath = os.path.join(self.predOffsetPath,
                                 '{}____off____{}.npy'.format(file_name.replace('.fits', ''), ct_assign_level))
          pred_hm = np.load(hmpath)
          pred_off = np.load(offpath)
          pred_hm_list.append(pred_hm)
          pred_off_list.append(pred_off)
        self.pred_hm = pred_hm_list
        self.pred_off = pred_off_list
      except Exception as e:
        pass

    if self.opt.fits:
      fits_img = fits.getdata(img_path)  # uint16
      img = np.expand_dims(np.empty(fits_img.shape, dtype='uint8'), 2).repeat(3, axis=2)

      import sep
      bkg = sep.Background(fits_img.astype(np.float32))
      fits_img = fits_img - np.array(bkg)
      # # manipulate bkg
      # fits_img = fits_img + 254

      fits_img[fits_img<0] = 0
      fits_img = fits_img.astype(dtype='uint16')


      img[:, :, 0] = (fits_img >> 8) & 0xff
      img[:, :, 1] = (fits_img >> 4) & 0xff
      img[:, :, 2] = fits_img & 0xff
    else:
      img = cv2.imread(img_path)
    if self.opt.mask:
      mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)[:,:,np.newaxis]

    height, width = img.shape[0], img.shape[1]
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    if self.opt.keep_res:
      input_h = (height | self.opt.pad) + 1
      input_w = (width | self.opt.pad) + 1
      s = np.array([input_w, input_h], dtype=np.float32)
    else:
      s = max(img.shape[0], img.shape[1]) * 1.0
      input_h, input_w = self.opt.input_h, self.opt.input_w
    
    flipped = False
    if self.split == 'train':
      if not self.opt.not_rand_crop:
        s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
        w_border = self._get_border(128, img.shape[1])
        h_border = self._get_border(128, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
      else:
        sf = self.opt.scale
        cf = self.opt.shift
        c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
      
      if np.random.random() < self.opt.flip:
        flipped = True
        img = img[:, ::-1, :]
        if self.opt.mask:
          mask = mask[:, ::-1, :]
        c[0] =  width - c[0] - 1
        

    if not self.opt.no_trans_input:
      trans_input = get_affine_transform(
        c, s, 0, [input_w, input_h])
      inp = cv2.warpAffine(img, trans_input,
                           (input_w, input_h),
                           flags=cv2.INTER_LINEAR)
      inp = (inp.astype(np.float32) / 255.)
      if self.opt.mask:
        maskp = cv2.warpAffine(mask, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)
        maskp = (maskp.astype(np.float32) / 255.)[:,:,np.newaxis]
    else:
      inp = (img.astype(np.float32) / 255.)
      if self.opt.mask:
        maskp = (mask.astype(np.float32) / 255.)

    if self.split == 'train' and not self.opt.no_color_aug:
      color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
    # inp = (inp - self.mean) / self.std
    inp = inp.transpose(2, 0, 1)
    if self.opt.mask:
      maskp = maskp.transpose(2, 0, 1)

    output_h = input_h // self.opt.down_ratio
    output_w = input_w // self.opt.down_ratio
    num_classes = self.num_classes
    trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

    if not self.opt.use_ms:
      hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
      wh = np.zeros((self.max_objs, 2), dtype=np.float32)
      dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
      reg = np.zeros((self.max_objs, 2), dtype=np.float32)
      ind = np.zeros((self.max_objs), dtype=np.int64)
      reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
      cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
      cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)
    else:
      hm_list = list()
      wh_list = list()
      reg_list = list()
      ind_list = list()
      reg_mask_list = list()
      k_list = list()
      if self.opt.use_cwassign:
        wt_ms_list = list()
      for tmp_level in range(self.opt.ms_levels):
        # tmp_output_h, tmp_output_w, tmp_max_objs = list(map(lambda x: int(x / pow(2, tmp_level)),
        #                                                     [output_h, output_h, self.max_objs]))
        tmp_output_h, tmp_output_w = list(map(lambda x: int(x / pow(2, tmp_level)),
                                                            [output_h, output_h]))
        tmp_max_objs = self.max_objs
        hm_list.append(np.zeros((num_classes, tmp_output_h, tmp_output_w), dtype=np.float32))
        wh_list.append(np.zeros((tmp_max_objs, 2), dtype=np.float32))
        reg_list.append(np.zeros((tmp_max_objs, 2), dtype=np.float32))
        ind_list.append(np.zeros((tmp_max_objs), dtype=np.int64))
        reg_mask_list.append(np.zeros((tmp_max_objs), dtype=np.uint8))
        k_list.append(0)
        if self.opt.use_cwassign:
          wt_ms_list.append(np.zeros((num_classes, tmp_output_h, tmp_output_w), dtype=np.float32))
      assert len(hm_list)==self.opt.ms_levels

    draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
                    draw_umich_gaussian
    # draw_gaussian = draw_umich_gaussian_weighted

    gt_det = []
    for k in range(num_objs):
      ann = anns[k]
      bbox = self._coco_box_to_bbox(ann['bbox'])
      ##########
      centroid = np.array(ann['centroid'])
      ##########
      cls_id = int(self.cat_ids[ann['category_id']])
      if flipped:
        bbox[[0, 2]] = width - bbox[[2, 0]] - 1
        # centroid[0] = width - centroid[0] - 1
        centroid[0] = width - centroid[0]
      if not self.opt.no_trans_input:
        bbox[:2] = affine_transform(bbox[:2], trans_output)
        bbox[2:] = affine_transform(bbox[2:], trans_output)
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
      else:
        bbox[[0, 2]] = bbox[[0, 2]] / self.opt.down_ratio
        bbox[[1, 3]] = bbox[[1, 3]] / self.opt.down_ratio
        centroid = centroid / self.opt.down_ratio
      h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

      if h > 0 and w > 0:
        if not self.opt.use_ms:
          radius = gaussian_radius((math.ceil(h), math.ceil(w)))
          radius = max(0, int(radius))
          radius = self.opt.hm_gauss if self.opt.mse_loss else radius
          # ct = np.array(
          #   [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
          ct = centroid.astype(np.float32)
          ct_int = ct.astype(np.int32)
          draw_gaussian(hm[cls_id], ct_int, radius)
          wh[k] = 1. * w, 1. * h
          ind[k] = ct_int[1] * output_w + ct_int[0]
          reg[k] = ct - ct_int
          assert all(ct - ct_int >= 0), 'offsets are minus value'
          reg_mask[k] = 1
          cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
          cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
          if self.opt.dense_wh:
            draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
          gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                         ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
        else:
          if self.opt.use_cwassign:
            # wt_ms = cenAwareAssign(centroid, h, w, self.opt.ms_levels)
            wt_ms = self.locAwareAssign(centroid, self.opt.ms_levels, file_name)
          for ct_assign_level in range(self.opt.ms_levels):
            h_l = h / pow(2, ct_assign_level)
            w_l = w / pow(2, ct_assign_level)
            ct_l = centroid.astype(np.float32) / pow(2, ct_assign_level)
            ct_int_l = ct_l.astype(np.int32)

            radius = gaussian_radius((math.ceil(h_l), math.ceil(w_l)))
            radius = max(0, int(radius))
            radius = self.opt.hm_gauss if self.opt.mse_loss else radius
            draw_gaussian(hm_list[ct_assign_level][cls_id], ct_int_l, radius)
            if self.opt.use_cwassign:
              wt_ms_list[ct_assign_level][0,np.clip(ct_int_l[1],0,hm_list[ct_assign_level].shape[1]-1),np.clip(ct_int_l[0],0,hm_list[ct_assign_level].shape[1]-1)] = wt_ms[ct_assign_level]

            kk = k_list[ct_assign_level]  ## kk is object's index in each level
            wh_list[ct_assign_level][kk] = 1. * w_l, 1. * h_l
            ind_list[ct_assign_level][kk] = ct_int_l[1] * int(output_w / pow(2, ct_assign_level)) + ct_int_l[0]
            reg_list[ct_assign_level][kk] = ct_l - ct_int_l
            assert all(ct_l - ct_int_l >= 0), 'offsets are minus value'
            reg_mask_list[ct_assign_level][kk] = 1
            k_list[ct_assign_level] = kk + 1

    if not self.opt.use_ms:
      ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
      if self.opt.dense_wh:
        hm_a = hm.max(axis=0, keepdims=True)
        dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
        ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
        del ret['wh']
      elif self.opt.cat_spec_wh:
        ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
        del ret['wh']
      if self.opt.reg_offset:
        ret.update({'reg': reg})
      if self.opt.debug > 0 or not self.split == 'train':
        gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                 np.zeros((1, 6), dtype=np.float32)
        meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
        ret['meta'] = meta
      if self.opt.mask:
        ret.update({'mask': maskp})
    else:
      assert (True not in np.isnan(hm_list[0]))
      assert (True not in np.isnan(hm_list[1]))
      ret = {'input': inp, 'hm': hm_list, 'reg_mask': reg_mask_list, 'ind': ind_list, 'wh': wh_list}
      if self.opt.reg_offset:
        ret.update({'reg': reg_list})
      if self.opt.use_cwassign:
        ret.update({'wt_ms': wt_ms_list})

    ret.update({'filename': file_name})

    return ret


''' statical centroid assignment '''
def cenAwareAssign(centroid, h, w, ms_levels):
  iou_list = np.zeros((ms_levels,))

  for ct_assign_level in range(ms_levels):
    h_l = h / pow(2, ct_assign_level)
    w_l = w / pow(2, ct_assign_level)
    ct_l = centroid.astype(np.float32) / pow(2, ct_assign_level)
    ct_int_l = np.floor(ct_l)

    x0, y0, x1, y1 = [ct_l[0]-w_l/2, ct_l[1]-h_l/2, ct_l[0]+w_l/2, ct_l[1]+h_l/2]
    ''' sample range definition '''
    X0, Y0, X1, Y1 = [ct_int_l[0], ct_int_l[1], ct_int_l[0]+1, ct_int_l[1]+1]
    # X0, Y0, X1, Y1 = [ct_int_l[0]+1/3, ct_int_l[1]+1/3, ct_int_l[0]+2/3, ct_int_l[1]+2/3] # sample 1/3
    ''' sample range definition '''
    inter_x0, inter_y0, inter_x1, inter_y1 = [max(x0,X0), max(y0,Y0), min(x1,X1), min(y1,Y1)]
    inter_area = (inter_x1-inter_x0) * (inter_y1-inter_y0)
    iou_list[ct_assign_level] = inter_area / (w_l*h_l + 1 - inter_area)

  iou_sum = np.sum(np.array(iou_list))
  iou_list = list(map(lambda x: x/iou_sum, iou_list))
  return iou_list


''' utils '''
def recreate_dir(dir_path):
  if os.path.isdir(dir_path):
    shutil.rmtree(dir_path)
  os.mkdir(dir_path)
