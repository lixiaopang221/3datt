import tensorflow as tf
import numpy as np


def mean_var(inputs, mask):
    maskfore = tf.where(mask > 0)
    maskback = tf.where(mask < 1)
    fore = tf.gather_nd(inputs, maskfore)
    back = tf.gather_nd(inputs, maskback)
    mean_fore, variance_fore = tf.nn.moments(fore, [0])
    mean_back, variance_back = tf.nn.moments(back, [0])
    loss = variance_fore + variance_back + 1 - (mean_fore - mean_back)  
    return loss

#分别计算wt，tc，et的dice
def dice_wt_tc_et(predict, gt, num_class = 4):
    predict = tf.cast(predict, tf.float32)
    gt = tf.cast(gt, tf.float32)
    dices = []
    for i in range(num_class - 1):
        predict_i = tf.tanh(tf.nn.relu(predict - i) * 10)
        gt_i = tf.tanh(tf.nn.relu(gt - i) * 10)
        dices.append(tf.reduce_mean(dice_test(predict_i, gt_i, 2)))
    return dices


def dice_test(predict, gt, num_class):
    predict = tf.cast(predict, tf.int32)
    predict = tf.layers.flatten(predict)
    predict = tf.one_hot(predict, num_class)
    gt = tf.cast(gt, tf.int32)
    gt = tf.layers.flatten(gt)
    gt = tf.one_hot(gt, num_class)
    gt = gt[...,1:]
    predict = predict[...,1:]
    intersection = tf.reduce_sum(gt * predict, axis = 1)
    #当predict和gt中某类为0时，dice = 1
    dice = (2.*intersection + 1e-7) / (tf.reduce_sum(gt, axis = 1) +
                                       tf.reduce_sum(predict, axis = 1) + 1e-7)
    # return tf.reduce_sum(dice, axis = 1)/(num_class - 1)
    return dice

def dice_train(predict, gt, axis = [1,2,3], smooth = 1e-5):
    'data: NHWDC'
    predict = predict[...,1:]
    gt = gt[...,1:]
    dice = dice_coe(predict, gt, axis, smooth)
    return tf.reduce_mean(dice)


def dice_coe(predict, gt, axis = [1,2,3], smooth = 1e-5):
    inse = tf.reduce_sum(predict * gt, axis = axis)
    l = tf.reduce_sum(predict * predict, axis = axis)
    r = tf.reduce_sum(gt * gt, axis = axis)
    dice = (2. * inse + smooth) / (l + r + smooth)
    # dice = (2. * inse) / (l + r + smooth)
    # dice = tf.reduce_mean(dice)
    return dice

    
def np_dice_wt_tc_et(predict, gt, num_class = 4):
    dices = []
    for i in range(num_class - 1):
        predict_i = np.zeros_like(predict)
        predict_i[predict > i] = 1
        gt_i = np.zeros_like(gt)
        gt_i[gt > i] = 1
        dices.append(np_dice(predict_i, gt_i, 2).mean())
    return dices
    
def np_dice(predict, gt, num_class):
    predict = predict.astype('int')
    predict = predict.flatten()
    predict = np_one_hot(predict, num_class)
    predict = predict[...,1:]
    gt = gt.astype('int')
    gt = gt.flatten()
    gt = np_one_hot(gt, num_class)
    gt = gt[...,1:]
    intersection = np.sum(gt * predict, axis = 0)
    #当predict和gt中某类为0时，dice = 1
    dice = (2.*intersection + 1e-7)/(np.sum(gt, axis = 0) + np.sum(predict, axis = 0) + 1e-7)
    return dice

def np_one_hot(input, num_class):
    outputs = []
    for i in range(num_class):
        temp = np.zeros_like(input)
        temp[input == i] = 1
        outputs.append(temp)
    return np.stack(outputs, axis = -1)
    
    
if __name__ == '__main__':
    p1 = tf.constant([[[1,1],[0,2]],[[1,2],[0,1]]]) #shape = [2,2,2]
    g1 = tf.constant([[[1,0],[0,2]],[[1,2],[0,1]]])
    p2 = tf.random_normal((2,24,24,15), dtype = tf.float32)
    g2 = tf.ones((2,24,24,15), dtype = tf.float32)
    l1 = tf.reduce_mean(dice_test(p1, g1, 4), 1)
    wt, tc, et = dice_wt_tc_et(p2, g2)
    
    with tf.Session() as sess:
        loss_1 = sess.run(l1)
        print(loss_1)
        lwt, ltc, let = sess.run([wt, tc, et])
        print(lwt, ltc, let )