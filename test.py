import numpy as np
import time, os, sys
import SimpleITK as itk
import tensorflow as tf
from data import Load
from model import Anet
from dice_function import np_dice
from scipy import ndimage as ndi


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


    def get_patch(self, img, label, box):
        'datda_format: NDHWC'
        # idx = np.nonzero(imgs)
        if not len(self.label_like):
            self.label_like = np.zeros_like(img[0,...,0])
        for point_d, point_h, point_w in self.points:
            img_patch = img[:,point_d:point_d + self.shape[0],
                              point_h:point_h + self.shape[1],
                              point_w:point_w + self.shape[2],:]
            label_patch = label[:,point_d:point_d + self.shape[0],
                                  point_h:point_h + self.shape[1],
                                  point_w:point_w + self.shape[2],:]
            yield img_patch, label_patch


    def patch_to_label(self, predicts):
        pred_sum = np.sum(predicts, (1, 2, 3))
        point_oder = pred_sum.argsort()
        label_like = self.label_like.copy()
        for i in point_oder:
            point_d, point_h, point_w = self.points[i]
            label_like[point_d:point_d + self.shape[0],
                       point_h:point_h + self.shape[1],
                       point_w:point_w + self.shape[2]] = predicts[i]
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

    def get_patch(self, img, label, box):
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
            label_patch = label[:,point_abs[0]:point_end[0],
                                  point_abs[1]:point_end[1],
                                  point_abs[2]:point_end[2],:]
            yield img_patch, label_patch


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

        
def test_predict(net_type='tc', path = None):
    with open(f'./net{net_type}.conf','r') as file:
        params = eval(file.read())['setting']
    p_shape         = params['p_shape']  #patch_shape
    net_name        = params['net_name']
    epochs          = params['epochs']
    data_root       = params['data_root']
    w_path          = params['w_path']
    train_path      = params['train_path']
    test_path       = params['test_path']
    w_path_epoch    = params['w_path_epoch']

    type_dict = {'wt':1, 'tc':2, 'et':3}
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmpgpuinfo')
    memory_gpu=[int(x.split()[2]) for x in open('tmpgpuinfo','r').readlines()]
    os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax(memory_gpu))
    os.system('rm tmpgpuinfo')
    '如果有label则test，没有label则只是predict。'
    if path is None:
        path = test_path
    x = tf.placeholder(tf.float32, shape=[None] + p_shape + [4], name = 'x')
    y = tf.placeholder(tf.float32, shape=[None] + p_shape + [4], name = 'y')
    net = Anet(net_type, 'anet')
    if net_type == 'et':
        y_f = y
    elif net_type == 'tc':
        y_f = y[...,3:4]
    elif net_type == 'wt':
        y_f = y[...,2:3]
    net.inference(x, y_f)
    predict = tf.argmax(net.out, 4)
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        file = open(f'./info_{net_type}_test.txt', 'a')
        if w_path_epoch in range(epochs):
            w_epochs = [w_path_epoch]
        else:
            w_epochs = range(epochs)
        loader = Load(data_root, path)
        patcher = Patch_test()
        patcher.shape = p_shape
        for epoch in w_epochs:
            w_path_epoch = w_path + '-' + str(epoch)
            saver.restore(sess, w_path_epoch)
            setting_info = ('\ntrain_path: {0} \ntest_path: {1}'.format(train_path, test_path) +
                            '\npatch_shape: {0}\nnet_type: {1}'.format(p_shape, net_name) + 
                            '\nw_path: {0}'.format(w_path_epoch))
            print(setting_info)
            print('\n' + setting_info, file = file)

            Dice = {'et':[], 'tc':[], 'wt':[]}
            Load.sample_names.sort()
            for begin in range(0, len(Load.sample_names), 20):
                '每次读20个样本（病人）。'
                end = begin + 20
                samples_batch = Load.sample_names[slice(begin, end)]
                imgs, labels = loader.load(samples_batch)
                for idx in range(len(imgs)):
                    imgs_batch = imgs[idx:idx + 1]
                    labels_batch = labels[idx:idx + 1]
                    sample_name = loader.samples[idx].split('/')[-1]
                    box = loader.boxes[idx]
                    test_predicts = []
                    for img, label in patcher.get_patch(imgs_batch, labels_batch, box):
                        test_predict = sess.run(predict, feed_dict = {x:img, y:label})
                        test_predicts.append(test_predict[0].astype('int8'))
                    label_predict = patcher.patch_to_label(test_predicts)
                    t_local = time.localtime()
                    t_str = '{t.tm_year}-{t.tm_mon}-{t.tm_mday}'.format(t = t_local) + \
                            ' {t.tm_hour}:{t.tm_min:02d}:{t.tm_sec:02d}'.format(t = t_local)
                    if len(labels):
                        dice_val = np_dice(label_predict, labels[idx,...,type_dict[net_type]], 2).mean()
                        Dice[net_type].append(dice_val)
                        save_path = './result/test/' + sample_name + '.nii.gz'
                        p_str = t_str + f'  dice_{net_type}: {dice_val:.4f}  {sample_name}'
                        print(p_str)
                        file.write('\n' + p_str)
                    else:
                        p_str = t_str + ' validation: {0}'.format(sample_name)
                        print(p_str)
                        save_path = './result/validation/' + sample_name + '.nii.gz'
                    img_itk = itk.GetImageFromArray(label_predict)
                    itk.WriteImage(img_itk, save_path)
            if len(Dice[net_type]) > 0:
                dice_mean = np.mean(Dice[net_type])
                Dice_str = f"\nmean {net_type}:{dice_mean:4f}"
                print(Dice_str)
                print(Dice_str, file = file)
        file.close()
    
        
def run(net_type=None):
    if net_type is None:
        with open('net_type.conf') as file:
            net_type = file.read().splitlines()[0][-2:]
        if len(sys.argv) >= 2:
            net_type = sys.argv[1]
    print('testing......')
    test_predict(net_type)  #默认test

if __name__ == '__main__':
    run()
