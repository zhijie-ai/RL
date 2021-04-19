#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/4/19 9:55                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import datetime,os
import numpy as np
import tensorflow as tf
import threading

from GAN_RL.yjp.code.options import get_options
from GAN_RL.yjp.code.data_util import Dataset
from GAN_RL.yjp.code.model import UserModelLSTM,UserModelPW

def multithread_compute_validation(out_):
    global vali_sum,vali_cnt

    vali_sum = [0.0,0.0,0.0]
    vali_cnt = 0
    threads = []
    for ii in range(cmd_args.num_thread):
        thread = threading.Thread(target=validation,args=(ii,out_))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return vali_sum[0]/vali_cnt,vali_sum[1]/vali_cnt,vali_sum[2]/vali_cnt

lock = threading.Lock()

def validation(ii,out_):
    global vali_sum,vali_cnt
    if cmd_args.user_model=='LSTM':
        vali_thread_eval = sess.run([train_loss_sum,train_prec1_sum,train_prec2_sum,train_event_cnt],
                                    feed_dict={user_model.placeholder['clicked_feature']:out_['click_feature'][ii],
                                               user_model.placeholder['ut_dispid_feature']:out_['u_t_dispid_feature'][ii],
                                               user_model.placeholder['ut_dispid_ut']:out_['u_t_dispid_split_ut'][ii],
                                               user_model.placeholder['ut_dispid']:out_['u_t_dispid'][ii],
                                               user_model.placeholder['ut_clickid']:out_['u_t_clickid'][ii] ,
                                               user_model.placeholder['ut_clickid_val']:np.ones(len(out_['u_t_clickid'][ii]), dtype=np.float32),
                                               user_model.placeholder['click_sublist_index']:np.array(out_['click_sub_index'][ii], dtype=np.int64),
                                               user_model.placeholder['ut_dense']:out_['user_time_dense'][ii],
                                               user_model.placeholder['time']:out_['max_time'][ii],
                                               user_model.placeholder['item_size']:out_['news_cnt_short_x'][ii]})
    elif cmd_args.user_model == 'PW':
        vali_thread_eval = sess.run([train_loss_sum,train_prec1_sum,train_prec2_sum,train_event_cnt],
                                    feed_dict={user_model.placeholder['disp_current_feature']: out['disp_current_feature_x'][ii],
                                               user_model.placeholder['item_size']: out['news_cnt_short_x'][ii],
                                               user_model.placeholder['section_length']: out['sec_cnt_x'][ii],
                                               user_model.placeholder['click_indices']: out['click_2d_x'][ii],
                                               user_model.placeholder['click_values']: np.ones(len(out['click_2d_x'][ii]), dtype=np.float32),
                                               user_model.placeholder['disp_indices']: np.array(out['disp_2d_x'][ii]),
                                               user_model.placeholder['cumsum_tril_indices']: out['tril_indice'][ii],
                                               user_model.placeholder['cumsum_tril_value_indices']: np.array(out['tril_value_indice'][ii], dtype=np.int64),
                                               user_model.placeholder['click_2d_subindex']: out['click_sub_index_2d'][ii],
                                               user_model.placeholder['disp_2d_split_sec_ind']: out['disp_2d_split_sec'][ii],
                                               user_model.placeholder['Xs_clicked']: out['feature_clicked_x'][ii]})

    lock.acquire()
    vali_sum[0] += vali_thread_eval[0]
    vali_sum[1] += vali_thread_eval[1]
    vali_sum[2] += vali_thread_eval[2]
    vali_cnt += vali_thread_eval[3]


if __name__ == '__main__':
    cmd_args = get_options()

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('%s,start '%log_time)
    dataset = Dataset(cmd_args)

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, load data completed" % log_time)

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, start to construct graph" % log_time)

    if cmd_args.user_model=='LSTM':
        user_model = UserModelLSTM(dataset.f_dim,cmd_args)
    elif cmd_args.user_model=='PW':
        user_model = UserModelPW(dataset.f_dim,cmd_args)
    else:
        print('using LSTM User model instead.')
        user_model = UserModelLSTM(dataset.f_dim,cmd_args)

    user_model.construct_placeholder()
    train_opt,train_loss,train_prec1,train_prec2,train_loss_sum,train_prec1_sum,train_prec2_sum,train_event_cnt = \
        user_model.construct_model(is_training=True,reuse=False)
    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, graph completed" % log_time)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    #prepare validation data
    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, start prepare vali data" % log_time)

    out_vali =  dataset.prepare_validation_data(cmd_args.num_thread,dataset.vali_user)

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, prepare validation data, completed" % log_time)

    out_test =  dataset.prepare_validation_data(cmd_args.num_thread,dataset.test_user)
    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, prepare test data, completed" % log_time)

    best_metric = [100000.0, 0.0, 0.0]
    vali_path =cmd_args.save_dir+'/'
    if not os.path.exists(vali_path):
        os.makedirs(vali_path)

    saver = tf.train.Saver(max_to_keep=None)

    for i in range(cmd_args.num_iters):
        # training_start_point = (i * cmd_args.batch_size) % (len(dataset.train_user))
        # training_user = dataset.train_user[training_start_point: min(training_start_point + cmd_args.batch_size, len(dataset.train_user))]
        training_user = np.random.choice(dataset.train_user,cmd_args.batch_size,replace=False)
        out =  dataset.data_process_for_placeholder(training_user)
        if cmd_args.user_model == 'LSTM':
            sess.run(train_opt,feed_dict={user_model.placeholder['clicked_feature']:out['click_feature'],
                                          user_model.placeholder['ut_dispid_feature']:out['u_t_dispid_feature'],
                                          user_model.placeholder['ut_dispid_ut']:out['u_t_dispid_split_ut'],
                                          user_model.placeholder['ut_dispid']:out['u_t_dispid'],
                                          user_model.placeholder['ut_clickid']:out['u_t_clickid'] ,
                                          user_model.placeholder['ut_clickid_val']:np.ones(len(out['u_t_clickid']), dtype=np.float32),
                                          user_model.placeholder['click_sublist_index']:np.array(out['click_sub_index'], dtype=np.int64),
                                          user_model.placeholder['ut_dense']:out['user_time_dense'],
                                          user_model.placeholder['time']:out['max_time'],
                                          user_model.placeholder['item_size']:out['news_cnt_short_x']})

        elif cmd_args.user_model=='PW':
            sess.run(train_opt, feed_dict={user_model.placeholder['disp_current_feature']: out['disp_current_feature_x'],
                                           user_model.placeholder['item_size']: out['news_cnt_short_x'],
                                           user_model.placeholder['section_length']: out['sec_cnt_x'],
                                           user_model.placeholder['click_indices']: out['click_2d_x'],
                                           user_model.placeholder['click_values']: np.ones(len(out['click_2d_x']), dtype=np.float32),
                                           user_model.placeholder['disp_indices']: np.array(out['disp_2d_x']),
                                           user_model.placeholder['cumsum_tril_indices']: out['tril_indice'],
                                           user_model.placeholder['cumsum_tril_value_indices']: np.array(out['tril_value_indice'], dtype=np.int64),
                                           user_model.placeholder['click_2d_subindex']: out['click_sub_index_2d'],
                                           user_model.placeholder['disp_2d_split_sec_ind']: out['disp_2d_split_sec'],
                                           user_model.placeholder['Xs_clicked']: out['feature_clicked_x']})

        if np.mod(i,10)==0:
            if i==0:
                log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s, start first iteration validation" % log_time)
            vali_loss_prc = multithread_compute_validation(out_vali)
            if i == 0:
                log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s, first iteration validation complete" % log_time)

            log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s: itr%d, vali: %.5f, %.5f, %.5f" %
                  (log_time, i, vali_loss_prc[0], vali_loss_prc[1], vali_loss_prc[2]))

            if vali_loss_prc[0]<best_metric[0]:
                best_metric[0] = vali_loss_prc[0]
                best_save_path = os.path.join(vali_path,'best-loss')
                best_save_path = saver.save(sess,best_save_path)
            if vali_loss_prc[1]>best_metric[1]:
                best_save_path = os.path.join(vali_path, 'best-pre1')
                best_save_path = saver.save(sess, best_save_path)
            if vali_loss_prc[2] > best_metric[2]:
                best_metric[2] = vali_loss_prc[2]
                best_save_path = os.path.join(vali_path, 'best-pre2')
                best_save_path = saver.save(sess, best_save_path)

        log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("%s, iteration %d train complete" % (log_time, i))

    # test
    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, start prepare test data" % log_time)
    out_test =  dataset.prepare_validation_data(cmd_args.num_thread,dataset.vali_user)

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, prepare test data end" % log_time)

    best_save_path = os.path.join(vali_path, 'best-loss')
    saver.restore(sess,best_save_path)
    test_loss_prc = multithread_compute_validation(out_test)
    vali_loss_prc = multithread_compute_validation(out_vali)
    print("test!!!loss!!!, test: %.5f, vali: %.5f" % (test_loss_prc[0], vali_loss_prc[0]))

    best_save_path = os.path.join(vali_path, 'best-pre1')
    saver.restore(sess, best_save_path)
    test_loss_prc = multithread_compute_validation(out_test)
    vali_loss_prc = multithread_compute_validation(out_vali)
    print("test!!!pre1!!!, test: %.5f, vali: %.5f" % (test_loss_prc[1], vali_loss_prc[1]))

    best_save_path = os.path.join(vali_path, 'best-pre2')
    saver.restore(sess, best_save_path)
    test_loss_prc = multithread_compute_validation(out_test)
    vali_loss_prc = multithread_compute_validation(out_vali)
    print("test!!!pre2!!!, test: %.5f, vali: %.5f" % (test_loss_prc[2], vali_loss_prc[2]))
