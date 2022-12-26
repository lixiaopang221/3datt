import os
import time
import numpy as np
import SimpleITK as itk
import tensorflow as tf
import subprocess as subp
from data import Load
from model import Anet
from setproctitle import setproctitle
# from skimage import measure
# from boundingbox import set_ps
from scipy import ndimage as ndi
from dice_function import np_dice_wt_tc_et

def chk_dir(fdir):
    if not os.path.isdir(fdir):
        os.makedirs(fdir)
    return fdir

def clean_contour(img):
    # Smaller areas with lower prob are very likely to be false positives
    s_input = np.zeros_like(img)
    s_input[img > 0] = 1
    wt_mor = ndi.binary_dilation(s_input, iterations = 2).astype('int8')
    labels, num_featrues = ndi.label(wt_mor)
    w_area = []
    for i in range(1, num_featrues+1):
        w_area.append(np.sum(s_input[labels == i]))
    if len(w_area) > 1:
        max_area = np.max(w_area)
        for i in range(len(w_area)):
            if w_area[i] < max_area / 3.0:
                img[labels == i + 1] = 0
    img = ndi.binary_fill_holes(img).astype(np.int8)
    return img
    
class Patch_test(object):
    def __init__(self, name = 'patch_test'):
        '默认patch_shape=[96,96,96], 取36个patch'
        self.name = name
        self.label_like = []
        self.shape = [96,96,96]
        self.points = []
        for d in [0,28,56]:
            for h in [20,72,124]:
                for w in [20,72,124]:
                    self.points.append((d,h,w))
                    
    def set_points(self, dps, hps, wps):
        'inputs are list or tuple.'
        self.points = []
        for d in dps:
            for h in hps:
                for w in wps:
                    self.points.append((d,h,w))

    def get_patch(self, img):
        'datda_format: NDHWC'
        # idx = np.nonzero(imgs)
        if not len(self.label_like):
            self.label_like = np.zeros_like(img[0,...,0])
        for point_d, point_h, point_w in self.points:
            img_patch = img[:,point_d:point_d + self.shape[0],
                              point_h:point_h + self.shape[1],
                              point_w:point_w + self.shape[2],:]
            yield img_patch


    def patch_to_label(self, predicts):
        pred_sum = np.sum((np.stack(predicts )> 0), (1, 2, 3))
        point_oder = pred_sum.argsort()
        label_like = self.label_like.copy()
        for i in point_oder:
            point_d, point_h, point_w = self.points[i]
            label_like[point_d:point_d + self.shape[0],
                       point_h:point_h + self.shape[1],
                       point_w:point_w + self.shape[2]] += predicts[i]
        label_like[label_like > 0] = 1
        return label_like

        
class PatchBox(object):
    def __init__(self, name = 'patch_test'):
        '默认patch_shape=[96,96,96], 取36个patch'
        self.name = name
        self.label_like = []
        self.shape = [96,96,96]
        self.points = []
        for d in [0,27,54]:
            for h in [0,41,84]:
                for w in [0,32,64]:
                    self.points.append((d,h,w))
                    
    def set_points(self, dps, hps, wps):
        'inputs are list or tuple.'
        self.points = []
        for d in dps:
            for h in hps:
                for w in wps:
                    self.points.append((d,h,w))

    def get_patch(self, img, box):
        'datda_format: NDHWC'
        #--- box shape [150,180,160]
        if not len(self.label_like):
            self.label_like = np.zeros_like(img[0,...,0])
        center = (np.array(box[1]) + np.array(box[0]))//2
        topleft = center - np.array([75,90,80])
        topleft = np.max([topleft, [0,0,0]], axis=0)
        self.topleft = np.min([topleft, [5,60,80]], axis=0)
        for point in self.points:
            point_abs = point + self.topleft
            point_end = point_abs + self.shape
            img_patch = img[:,point_abs[0]:point_end[0],
                              point_abs[1]:point_end[1],
                              point_abs[2]:point_end[2],:]
            yield img_patch

    def patch_to_label(self, predicts):
        pred_sum = np.sum((np.stack(predicts )> 0), (1, 2, 3))
        point_oder = pred_sum.argsort()
        label_like = self.label_like.copy()
        for i in point_oder:
            point = self.points[i]
            point_abs = point + self.topleft
            point_end = point_abs + self.shape
            label_like[point_abs[0]:point_end[0],
                       point_abs[1]:point_end[1],
                       point_abs[2]:point_end[2]] = predicts[i]
        # label_like[label_like > 0] = 1
        return label_like
        
        
def test_predict(path = None):

    with open('./netetw.conf','r') as f:
        params     = eval(f.read())['setting']
    p_shape        = params['p_shape']  #patch_shape
    net_name       = params['net_name']
    epochs         = params['epochs']
    data_root      = params['data_root']
    test_path      = params['test_path']
    w_path         = params['w_path_test']
    w_path_epoch   = params['w_path_epoch']
    post           = params['post']
    save_dir       = params['save_dir_test']
    view_direction = params['view_direction']

    chk_dir(save_dir)
    
    gpu_text = subp.getoutput('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free')
    gpu_id = np.argmax([int(x.split()[2]) for x in gpu_text.splitlines()])
    print('using gpu......:', gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    setproctitle('huanggh train')
    
    '如果有label则test，没有label则只是predict。'
    if path is None:
        path = test_path
    loader = Load(data_root, path)
    patcher = Patch_test()
    # patcher = PatchBox()
    patcher.shape = p_shape #默认是[96,96,96]

    x = tf.placeholder(tf.float32, shape=[None] + p_shape + [4], name = 'x')
    
    net_et = Anet('et')
    net_et.inference(x, None)
    prob_et = net_et.out[...,1:2]
    predict_et = tf.argmax(net_et.out, 4)
    net_tc = Anet('tc')
    net_tc.inference(x, prob_et)
    prob_tc = net_tc.out[...,1:2]
    predict_tc = tf.argmax(net_tc.out, 4)
    net_wt = Anet('wt')
    net_wt.inference(x, prob_tc)
    prob_wt = net_et.out[...,1:2]
    predict_wt = tf.argmax(net_wt.out, 4)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        file = open('./info_etw_test.txt', 'a')
        if w_path_epoch is None:
            w_epochs = range(epochs)
        else:
            w_epochs = [w_path_epoch]
        for epoch in w_epochs:
            w_path_epoch = w_path + '-' + str(epoch)
            # w_path_epoch = f'./saver/fine_tune/{epoch}'
            saver.restore(sess, w_path_epoch)
            setting_info = ('\ndata_path: {0} \nw_path_epoch: {1}'.format(path, w_path_epoch) +
                            '\npost: {0}\nview_direction: {1}'.format(post,view_direction) +
                            '\npatch_shape: {0}\nnet_type: {1}'.format(p_shape, net_name))
            print(setting_info)
            print('\n' + setting_info, file = file)

            Dice = {'et':[], 'tc':[], 'wt':[]}
            Load.sample_names.sort()
            for begin in range(0, len(Load.sample_names), 2):
                '每次读20个样本（病人）。'
                end = begin + 2 #---hgh
                samples_batch = Load.sample_names[slice(begin, end)]
                imgs, labels = loader.load(samples_batch)
                for idx in range(len(imgs)):
                    imgs_batch = imgs[idx:idx + 1]
                    sample_name = loader.samples[idx].split('/')[-1]
                    box = loader.boxes[idx]
                    test_predicts_wt = []
                    test_predicts_tc = []
                    test_predicts_et = []
                    # patcher = set_ps(patcher, sample_name)#-----
                    for img in patcher.get_patch(imgs_batch):
                    # for img in patcher.get_patch(imgs_batch, box):
                        img = view_transpose(img, view_direction)
                        test_predict_et, test_predict_tc, test_predict_wt = sess.run(
                                [predict_et, predict_tc, predict_wt], feed_dict = {x:img})
                        test_predict_et = view_transpose(test_predict_et, view_direction, True)
                        test_predict_tc = view_transpose(test_predict_tc, view_direction, True)
                        test_predict_wt = view_transpose(test_predict_wt, view_direction, True)
                        test_predicts_et.append(test_predict_et[0])
                        test_predicts_tc.append(test_predict_tc[0])
                        test_predicts_wt.append(test_predict_wt[0])

                    label_predict_et = patcher.patch_to_label(test_predicts_et)
                    label_predict_tc = patcher.patch_to_label(test_predicts_tc)
                    label_predict_wt = patcher.patch_to_label(test_predicts_wt)

                    if post == 'et800' or post == 'wtc':
                        if post == 'et800':
                            if label_predict_et.sum() < 800:
                                label_predict_et = np.zeros_like(label_predict_et)
                        label_predict_wt = clean_contour(label_predict_wt)
                        label_predict_tc = label_predict_wt*label_predict_tc
                        label_predict_et = label_predict_tc*label_predict_et
                        if post == 'wtc':    #--- for brats2015 do not require to remove et
                            if label_predict_et.sum() < 400:
                                label_predict_et = np.zeros_like(label_predict_et)
                        label_predict = label_predict_wt + label_predict_tc + label_predict_et
                    elif post == 'no-constrain':
                        label_predict = np.zeros_like(label_predict_wt, dtype = np.int16)
                        label_predict[label_predict_wt > 0] = 1
                        label_predict[label_predict_tc > 0] = 2
                        label_predict[label_predict_et > 0] = 3
                    t_local = time.localtime()
                    t_str = ('{t.tm_year}-{t.tm_mon}-{t.tm_mday}'.format(t = t_local) +
                             ' {t.tm_hour}:{t.tm_min:02d}:{t.tm_sec:02d}'.format(t = t_local))
                    if len(labels):
                        dice_wt, dice_tc, dice_et = np_dice_wt_tc_et(label_predict, labels[idx, ..., 0])
                        Dice['et'].append(dice_et)
                        Dice['tc'].append(dice_tc)
                        Dice['wt'].append(dice_wt)
                        save_path = save_dir + sample_name + '.nii.gz'
                        p_str = t_str + ('  dice_et:{0:.4f} dice_tc:{1:.4f}'.format(dice_et, dice_tc) +
                                         ' dice_wt:{0:.4f} {1}'.format(dice_wt, sample_name))
                        print(p_str)  
                        file.write('\n' + p_str)
                    else:
                        p_str = t_str + ' valuing...... {0}'.format(sample_name)
                        print(p_str)
                        save_path = save_dir + '/' + sample_name + '.nii.gz'
                    label_predict = loader.label_convert(label_predict, [1,2,3], [2,1,4])
                    img_itk = itk.GetImageFromArray(label_predict.astype('int16'))
                    itk.WriteImage(img_itk, save_path)
            if len(Dice['et']) > 0:
                Dice['et'] = np.mean(Dice['et'])
                Dice['tc'] = np.mean(Dice['tc'])
                Dice['wt'] = np.mean(Dice['wt'])
                Dice_str = "\nmean \net:{d[et]:4f} \ntc:{d[tc]:4f} \nwt:{d[wt]:4f}".format(d = Dice)
                print(Dice_str)
                print(Dice_str, file = file)
        file.close()

def view_transpose(img, view_direction, reverse=False):
    if not reverse:
        if view_direction == 'coronal':
            img = np.transpose(img, [0,2,3,1,4])
        elif view_direction == 'sagittal':
            img = np.transpose(img, [0,3,1,2,4])
        return img
    else:
        if view_direction == 'coronal':
            img_re = np.transpose(img, [0,3,1,2])
        elif view_direction == 'sagittal':
            img_re = np.transpose(img, [0,2,3,1])
        else:
            img_re = img
        return img_re

def run():
    print('testing......')
    with open('./info_etw_test.txt', 'a') as file:
        file.write('\n\ntesting......')
    test_predict()  #默认test

if __name__ == '__main__':
    run()