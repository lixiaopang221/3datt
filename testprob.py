import tensorflow as tf
from data import Load
from model import Anet
from dice_function import np_dice_wt_tc_et
import time, os
import SimpleITK as itk
import numpy as np
import nibabel as nib


class Patch_test(object):
    def __init__(self, name = 'patch_test'):
        '默认patch_shape=[96,96,96], 取27个patch'
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
        pred_sum = np.sum(predicts, (1, 2, 3))
        point_oder = pred_sum.argsort()
        label_like = self.label_like.copy()
        for i in point_oder:
            point_d, point_h, point_w = self.points[i]
            label_roi = label_like[point_d:point_d + self.shape[0],
                                   point_h:point_h + self.shape[1],
                                   point_w:point_w + self.shape[2]]
            label_roi[label_roi < predicts[i]] = predicts[i][label_roi < predicts[i]]
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
            label_roi = label_like[point_abs[0]:point_end[0],
                                   point_abs[1]:point_end[1],
                                   point_abs[2]:point_end[2]]
            label_roi[label_roi < predicts[i]] = predicts[i][label_roi < predicts[i]]
        return label_like
        
def test_predict(path = None):
    with open('./netetw.conf','r') as f:
        params = eval(f.read())['setting']
    p_shape = params['p_shape']  #patch_shape
    net_name = params['net_name']
    epochs = params['epochs']
    data_root = params['data_root']
    test_path = params['test_path']
    w_path = params['w_path']
    w_path_epoch = params['w_path_epoch']
    post = params['post']

    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmpgpuinfo')
    memory_gpu=[int(x.split()[2]) for x in open('tmpgpuinfo','r').readlines()]
    os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax(memory_gpu))
    os.system('rm tmpgpuinfo')
    '如果有label则test，没有label则只是predict。'
    if path is None:
        path = test_path
    loader = Load(data_root, path)
    patcher = Patch_test()
    patcher.shape = p_shape #默认是[96,96,96]
    
    x = tf.placeholder(tf.float32, shape=[None] + p_shape + [4], name = 'x')
    net_et = Anet('et')
    net_et.inference(x, None)
    prob_et = net_et.out[...,1:2]
    net_tc = Anet('tc')
    net_tc.inference(x, prob_et)
    prob_tc = net_tc.out[...,1:2]
    net_wt = Anet('wt')
    net_wt.inference(x, prob_tc)
    prob_wt = net_et.out[...,1:2]
    
    prob_et = prob_et[...,0]
    prob_tc = prob_tc[...,0]
    prob_wt = prob_wt[...,0]

    saver = tf.train.Saver()
    with tf.Session() as sess:
        if w_path is None:
            ckpt_file = tf.train.latest_checkpoint('./saver/fine_tune')
            saver.restore(sess, ckpt_file)
        else:
            saver.restore(sess, w_path + '-' + str(w_path_epoch))
        file = open('./info_test.txt', 'a')
        setting_info = ('\ndata_path: {0} \nw_path: {1}'.format(path, w_path) +
                        '\npost: {0}'.format(post) +
                        '\npatch_shape: {0}\nnet_type: {1}'.format(p_shape, net_name))
        print(setting_info)
        print('\n' + setting_info, file = file)
        file.close()

        Load.sample_names.sort()
        for begin in range(0, len(Load.sample_names), 20):
            '每次读20个样本（病人）。'
            end = begin + 20
            samples_batch = Load.sample_names[slice(begin, end)]
            imgs, labels = loader.load(samples_batch)
            
            for idx in range(len(imgs)):
                imgs_batch = imgs[idx:idx + 1]
                sample_name = loader.samples[idx].split('/')[-1]
                
                test_probs_wt = []
                test_probs_tc = []
                test_probs_et = []
                for img in patcher.get_patch(imgs_batch):
                    test_prob_et, test_prob_tc, test_prob_wt = sess.run(
                            [prob_et, prob_tc, prob_wt], feed_dict = {x:img})
                    test_probs_et.append(test_prob_et[0])
                    test_probs_tc.append(test_prob_tc[0])
                    test_probs_wt.append(test_prob_wt[0])

                label_prob_et = patcher.patch_to_label(test_probs_et)
                label_prob_tc = patcher.patch_to_label(test_probs_tc)
                label_prob_wt = patcher.patch_to_label(test_probs_wt)

                t_local = time.localtime()
                t_str = ('{t.tm_year}-{t.tm_mon}-{t.tm_mday}'.format(t = t_local) +
                         ' {t.tm_hour}:{t.tm_min:02d}:{t.tm_sec:02d}'.format(t = t_local))
                print(t_str, sample_name)
                save_nii(label_prob_et, sample_name, 'et.nii.gz')
                save_nii(label_prob_tc, sample_name, 'tc.nii.gz')
                save_nii(label_prob_wt, sample_name, 'wt.nii.gz')

        
def save_nii(img, img_dir, img_name):
    path = './result/prob/' + img_dir
    if not os.path.isdir(path):
        os.mkdir(path)
    f_path = os.path.join(path, img_name)
    img = (img * 10000).astype('int16')
    img_itk = itk.GetImageFromArray(img)
    itk.WriteImage(img_itk, f_path)

def run():
    print('valuing probability......')
    with open('./info_test.txt', 'a') as file:
        file.write('\n\nvaluinging probability......')
    # test_predict('../../Brats17ValidationData')  #默认test
    test_predict()

if __name__ == '__main__':
    run()