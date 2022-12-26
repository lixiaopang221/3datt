import tensorflow as tf
import time, os, random, send2trash
import numpy as np
import subprocess as subp
import SimpleITK as itk
from data import Load
from model import Anet
from setproctitle import setproctitle
from dice_function import dice_coe, np_dice, mean_var


def train():
    with open('./netetw.conf','r') as f:
        params = eval(f.read())['setting']
    p_shape         = params['p_shape']  #patch_shape
    net_name        = params['net_name']
    w_path          = params['w_path']
    data_root       = params['data_root']
    train_path      = params['train_path']
    fine_tune       = params['fine_tune']
    w_paths_train   = params['w_paths_train']
    epochs          = params['epochs']
    sampling_times  = params['sampling_times']
    batch_size      = params['batch_size']
    learning_rate   = params['learning_rate']
    ttest_weight    = params['ttest_weight']
    view_direction  = params['view_direction']
    log_file        = params['log_file']

    type_dict = {'wt':1, 'tc':2, 'et':3}
    
    gpu_text = subp.getoutput('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free')
    gpu_id = np.argmax([int(x.split()[2]) for x in gpu_text.splitlines()])
    print('using gpu......:', gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    setproctitle('huanggh train')
    
    loader = Load(data_root, train_path)
    x = tf.placeholder(tf.float32, shape=[None] + p_shape + [4], name = 'x')
    y = tf.placeholder(tf.float32, shape=[None] + p_shape + [4], name = 'y')

    net_et = Anet('et')
    net_et.inference(x, None)
    prob_et = net_et.out[...,1:2]
    gt_et = y[...,3]
    dice_coe_et = tf.reduce_mean(dice_coe(prob_et[...,0], gt_et))
    loss_et = 1 - dice_coe_et + mean_var(prob_et[...,0], gt_et)*ttest_weight
    predict_et = tf.argmax(net_et.out, 4)

    net_tc = Anet('tc')
    net_tc.inference(x, prob_et)
    prob_tc = net_tc.out[...,1:2]
    gt_tc = y[...,2]
    dice_coe_tc = tf.reduce_mean(dice_coe(prob_tc[...,0], gt_tc))
    loss_tc = 1 - dice_coe_tc + mean_var(prob_tc[...,0], gt_tc)*ttest_weight
    predict_tc = tf.argmax(net_tc.out, 4)

    net_wt = Anet('wt')
    net_wt.inference(x, prob_tc)
    prob_wt = net_wt.out[...,1]
    gt_wt = y[...,1]
    dice_coe_wt = tf.reduce_mean(dice_coe(prob_wt, gt_wt))
    loss_wt = 1 - dice_coe_wt + mean_var(prob_wt, gt_wt)*ttest_weight
    predict_wt = tf.argmax(net_wt.out, 4)

    wb_collect = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = (loss_et + loss_tc + loss_wt) / 3 + tf.add_n(wb_collect)

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'et')
    saver_et = tf.train.Saver(var_list = vars)
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'tc')
    saver_tc = tf.train.Saver(var_list = vars)
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'wt')
    saver_wt = tf.train.Saver(var_list = vars)
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        
    saver = tf.train.Saver(max_to_keep = 10, var_list = vars)
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
        file =  open(log_file, 'a')
        sess.run(init)
        if fine_tune:
            print('loading the checkpooint and fine_tune......')
            ckpt_file = tf.train.latest_checkpoint('./saver/etw0')
            saver.restore(sess, ckpt_file)
        elif w_paths_train is None:
            print('loading the pretrained et, tc, wt net params......')
            ckpt_file = tf.train.latest_checkpoint('./saver/et')
            saver_et.restore(sess, ckpt_file)
            ckpt_file = tf.train.latest_checkpoint('./saver/tc')
            saver_tc.restore(sess, ckpt_file)
            ckpt_file = tf.train.latest_checkpoint('./saver/wt')
            saver_wt.restore(sess, ckpt_file)
        else:
            print('loading the pretrained et, tc, wt net params......')
            saver_et.restore(sess, './saver/et/Anet.ckpt-' + w_paths_train['et'])
            saver_tc.restore(sess, './saver/tc/Anet.ckpt-' + w_paths_train['tc'])
            saver_wt.restore(sess, './saver/wt/Anet.ckpt-' + w_paths_train['wt'])
        
        remove_files('./train_predict')
        setting_info = ('training......\n epochs: {0}\n'.format(epochs) +
                        'sampling_times: {0}\n'.format(sampling_times) +
                        'data_path: {0}\nbatch_size: {1}\n'.format(train_path, batch_size) +
                        'patch_shape: {0}\nnet_type: {1}\n'.format(p_shape, net_name) +
                        'view_direction: {0}'.format(view_direction))
        print(setting_info)
        print('\n' + setting_info, file = file)

        steps = 0
        for epoch in range(epochs):
            print('trining epoch: {0}'.format(epoch))
            random.shuffle(Load.sample_names)
            for begin in range(0, len(Load.sample_names), 48):
                '每次读32个样本（病人）'
                end = begin + 48
                samples_batch = Load.sample_names[begin:end]
                loader.load(samples_batch)
                print('\n' + '\n'.join(samples_batch), file = file)
                for sampling_time in range(sampling_times):
                    for img, label in loader.get_batch_of_patch(batch_size, p_shape):
                        if view_direction == 'coronal':
                            img = np.transpose(img, [0,2,3,1,4])
                            label = np.transpose(label, [0,2,3,1,4])
                        elif view_direction == 'sagittal':
                            img = np.transpose(img, [0,3,1,2,4])
                            label = np.transpose(label, [0,3,1,2,4])
                        (_train_step, train_loss, train_predict_et, train_loss_et, train_dice_et,
                                                  train_predict_tc, train_loss_tc, train_dice_tc,
                                                  train_predict_wt, train_loss_wt, train_dice_wt) = sess.run(
                               [train_step, loss, predict_et, loss_et, dice_coe_et,
                                                  predict_tc, loss_tc, dice_coe_tc,
                                                  predict_wt, loss_wt, dice_coe_wt], feed_dict={x:img, y:label})
                        steps += 1

                        if steps%20 == 0:
                            t_local = time.localtime()
                            t_str = ('{t.tm_year}-{t.tm_mon}-{t.tm_mday} '.format(t = t_local) +
                                     '{t.tm_hour:>2}:{t.tm_min:02d}:{t.tm_sec:02d}'.format(t = t_local))
                            info_str = t_str + ' steps:{0:<5} train_loss:{1:.4f}'.format(steps, train_loss)
                            print(info_str)
                            file.write('\n' + info_str)
                            t_dice_et = np_dice(train_predict_et, label[...,3], 2).mean()
                            info_str_et = ('{0:>{1}}train_loss_et:{2:.4f}'.format(' ', 13 + len(t_str), train_loss_et) +
                                           '  train_dice_et:{0:.4f}  t_dice_et:{1:.4f};'.format(train_dice_et, t_dice_et))
                            print(info_str_et)
                            file.write('\n' + info_str_et)
                            t_dice_tc = np_dice(train_predict_tc, label[...,2], 2).mean()
                            info_str_tc = ('{0:>{1}}train_loss_tc:{2:.4f}'.format(' ', 13 + len(t_str), train_loss_tc) +
                                           '  train_dice_tc:{0:.4f}  t_dice_tc:{1:.4f}'.format(train_dice_tc, t_dice_tc))
                            print(info_str_tc)
                            file.write('\n' + info_str_tc)
                            t_dice_wt = np_dice(train_predict_wt, label[...,1], 2).mean()
                            info_str_wt = ('{0:>{1}}train_loss_wt:{2:.4f}'.format(' ', 13 + len(t_str), train_loss_wt) +
                                           '  train_dice_wt:{0:.4f}  t_dice_wt:{1:.4f}'.format(train_dice_wt, t_dice_wt))
                            print(info_str_wt)
                            file.write('\n' + info_str_wt)
                        if steps%200 == 0:
                            saver.save(sess, w_path, global_step = epoch)
                            save_zeros = np.zeros([155,240,240], dtype = np.int16)
                            train_predict = (train_predict_wt + train_predict_wt*train_predict_tc +
                                             train_predict_wt*train_predict_tc*train_predict_et)
                            save_predict = loader.label_convert(train_predict[0], [1,2,3], [2,1,4])
                            d = (155 - p_shape[0]) // 2
                            h = (240 - p_shape[1]) // 2
                            w = (240 - p_shape[2]) // 2
                            save_zeros[d:d + p_shape[0], h:h + p_shape[1], w:w + p_shape[2]] = save_predict.astype('int16')
                            img_itk = itk.GetImageFromArray(save_zeros)
                            step_str = '{0:05d}'.format(steps)
                            itk.WriteImage(img_itk, './train_predict/' + step_str + '.nii.gz')

            saver.save(sess, w_path, global_step = epoch)
        file.close()


def remove_files(folder):
    print('deleting...... {0}/*'.format(folder))
    for name in os.listdir(folder):
        path_name = os.path.join(folder, name)
        if os.path.isfile(path_name):
            send2trash.send2trash(path_name)


def run():
    t0 = time.localtime()
    train()
    t1 = time.localtime()
    str_t0 = '{t.tm_mday} {t.tm_hour:>2}:{t.tm_min:02d}:{t.tm_sec:02d}'.format(t = t0)
    str_t1 = '{t.tm_mday} {t.tm_hour:>2}:{t.tm_min:02d}:{t.tm_sec:02d}'.format(t = t1)
    print('bengin:', str_t0, '\nend:', str_t1)
    with open('info_etw_train.txt', 'a') as file:
        print('\nbengin:', str_t0, '\n***end:', str_t1, file = file)

    # import etwtest
    # tf.reset_default_graph()
    # etwtest.run()
    
if __name__ == '__main__':
    run()