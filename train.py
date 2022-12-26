import tensorflow as tf
import time, os, sys, random
import send2trash
import numpy as np
import SimpleITK as itk
from data import Load
from model import Anet
from dice_function import np_dice, dice_coe


def train(net_type='tc'):
    with open(f'./net{net_type}.conf','r') as f:
        params = eval(f.read())['setting']
    p_shape         = params['p_shape']  #patch_shape
    net_name        = params['net_name']
    epochs          = params['epochs']
    data_root       = params['data_root']
    w_path          = params['w_path']
    sampling_times  = params['sampling_times']
    fine_tune       = params['fine_tune']
    batch_size      = params['batch_size']
    learning_rate   = params['learning_rate']
    train_path      = params['train_path']
    
    type_dict = {'wt':1, 'tc':2, 'et':3}
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmpgpuinfo')
    memory_gpu=[int(x.split()[2]) for x in open('tmpgpuinfo','r').readlines()]
    os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax(memory_gpu))
    os.system('rm tmpgpuinfo')
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
    prob = net.out[...,1]
    gt = y[...,type_dict[net_type]]
    dice_coe_val = tf.reduce_mean(dice_coe(prob, gt))
    loss_dice = 1 - dice_coe_val
    wb_collect = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = loss_dice + tf.add_n(wb_collect)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    saver = tf.train.Saver(max_to_keep = 20)
    init = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        sess.run(init)
        file =  open(f'info_{net_type}_train.txt', 'a')
        if fine_tune:
            print('Fine-tuning......')
            print('Fine-tuning......', file = file)
            ckpt_file = tf.train.latest_checkpoint('./saver/')
            saver.restore(sess, ckpt_file)
        else:
            remove_files('./train_predict')
            print('training......')
            print('\ntraining......', file = file)
        setting_info = ('epochs: {0}'.format(epochs) +
                        '\ndata_path: {0}\nbatch_size: {1}'.format(train_path, batch_size) +
                        '\ninput_shape: {0}\nnet_type: {1}'.format(p_shape, net_name))
        print(setting_info)
        print('\n' + setting_info, file = file)

        loader = Load(data_root, train_path)
        steps = 0
        for epoch in range(epochs):
            print('trining epoch: {0}'.format(epoch))
            print('trining epoch: {0}'.format(epoch), file=file)
            random.shuffle(Load.sample_names)
            for idx_sample in range(0, len(Load.sample_names), 48):
                sample_names = Load.sample_names[idx_sample:idx_sample+48]
                loader.load(sample_names)
                dice_dict = dict(wt=[],tc=[],et=[])
                for i in range(sampling_times):
                    for img, label in loader.get_batch_of_patch(batch_size, p_shape):
                        label_type = np.array(label)[...,type_dict[net_type]]
                        if label_type.sum() < 10:
                            continue
                        _train_step, train_loss, train_predict, train_dice = sess.run(
                                [train_step, loss, predict, dice_coe_val], feed_dict={x:img, y:label})
                        steps += 1
                        if steps%20 == 0:
                            t_local = time.localtime()
                            t_str = ('{t.tm_year}-{t.tm_mon}-{t.tm_mday} '
                                     '{t.tm_hour:>2}:{t.tm_min:02d}:{t.tm_sec:02d}').format(t = t_local)
                            info_str = t_str + ' steps:{0:<5} train_loss:{1:.4f}'.format(steps, train_loss)
                            t_dice = np_dice(train_predict, label_type, 2).mean()
                            dice_dict[net_type].append(t_dice)
                            info_str_type = f' train_dice_{net_type}:{train_dice:.4f}  t_dice_{net_type}:{t_dice:.4f}'
                            print(info_str, info_str_type)
                            file.write('\n' + info_str + info_str_type)
                        if steps%200 == 0:
                            saver.save(sess, w_path, global_step = epoch)
                            save_zeros = np.zeros([155,240,240], dtype = np.int16)
                            save_predict = train_predict[0]
                            d = (155 - p_shape[0]) // 2
                            h = (240 - p_shape[1]) // 2
                            w = (240 - p_shape[2]) // 2
                            save_zeros[d:d + p_shape[0],
                                       h:h + p_shape[1],
                                       w:w + p_shape[2]] = save_predict.astype('int16')
                            img_itk = itk.GetImageFromArray(save_zeros)
                            step_str = '{0:05d}'.format(steps)
                            itk.WriteImage(img_itk, './train_predict/' + step_str + '.nii.gz')
                print(f't_dice_mean: {np.mean(dice_dict[net_type])}')
                print(f'\nt_dice_mean: {np.mean(dice_dict[net_type])}', file=file)
            saver.save(sess, w_path, global_step = epoch)
        file.close()


def remove_files(folder):
    print('deleting...... {0}/*'.format(folder))
    for name in os.listdir(folder):
        path_name = os.path.join(folder, name)
        if os.path.isfile(path_name):
            send2trash.send2trash(path_name)
            

def run():
    with open('net_type.conf') as file:
        net_type = file.read().splitlines()[0][-2:]
    if len(sys.argv) >= 2:
        net_type = sys.argv[1]
    t0 = time.localtime()
    train(net_type)
    t1 = time.localtime()
    str_t0 = '{t.tm_mday} {t.tm_hour:>2}:{t.tm_min:02d}:{t.tm_sec:02d}'.format(t = t0)
    str_t1 = '{t.tm_mday} {t.tm_hour:>2}:{t.tm_min:02d}:{t.tm_sec:02d}'.format(t = t1)
    print('bengin:', str_t0, '\n***end:', str_t1)
    with open(f'info_{net_type}_train.txt', 'a') as file:
        print('\nbengin:', str_t0, '\n***end:', str_t1, file = file)

    import test
    tf.reset_default_graph()
    test.run(net_type)
    
if __name__ == '__main__':
    run()