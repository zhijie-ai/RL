# -----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/4/19 9:55                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# -----------------------------------------------
# 根据论文的思路训练模型得到env
import datetime, os, time
import numpy as np
import tensorflow as tf
import threading
from multiprocessing import Process
from tqdm import tqdm
from utils.yjp_decorator import cost_time_def

from GAN_RL.yjp.code.options import get_options
from GAN_RL.yjp.code.data_util import Dataset
from GAN_RL.yjp.code.model import UserModelLSTM, UserModelPW

import warnings

warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow')
warnings.filterwarnings('ignore')


@cost_time_def
def multithread_compute_validation(model, out_):
    global vali_sum, vali_cnt

    vali_sum = [0.0, 0.0, 0.0]
    vali_cnt = 0
    threads = []
    for ii in range(cmd_args.num_thread):
        if cmd_args.compu_type == 'thread':
            thread = threading.Thread(target=validation, args=(model, out_, ii))
        else:
            thread = Process(target=validation, args=(model, out_, ii))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return vali_sum[0] / vali_cnt, vali_sum[1] / vali_cnt, vali_sum[2] / vali_cnt


lock = threading.Lock()


@cost_time_def
def validation(model, out_, ii):
    global vali_sum, vali_cnt

    vali_thread_eval = model.validation_on_batch_multi(out_, ii)
    lock.acquire()
    vali_sum[0] += vali_thread_eval[0]
    vali_sum[1] += vali_thread_eval[1]
    vali_sum[2] += vali_thread_eval[2]
    vali_cnt += vali_thread_eval[3]
    lock.release()


def train_on_batch(model, out_train, out_vali, i):
    loss, step, precision_1, precision_2= model.train_on_batch(out_train)
    losses.append(loss)
    prec1.append(precision_1)
    prec2.append(precision_2)

    if np.mod(step, 10) == 0:
        log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("%s, itr: %d, loss: %.5f, "
              "precision_1: %.5f, precision_2: %.5f" % (
                  log_time, step,
                  loss, precision_1,
                  precision_2,
              ))

    if np.mod(step, 100) == 0:
        if i == 0:
            log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s, start first iteration validation" % log_time)
        vali_loss_prc = model.validation_on_batch(out_vali)
        if i == 0:
            log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s, first iteration validation complete" % log_time)

        log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(">>>>>>>>%s: itr: %d, vali: %.5f, %.5f, %.5f<<<<<<<<" %
              (log_time, step, vali_loss_prc[0], vali_loss_prc[1], vali_loss_prc[2]))

        if vali_loss_prc[0] < best_metric[0]:
            best_metric[0] = vali_loss_prc[0]
            user_model.save('best-loss')
        if vali_loss_prc[1] > best_metric[1]:
            best_metric[1] = vali_loss_prc[1]
            user_model.save('best-pre1')
        if vali_loss_prc[2] > best_metric[2]:
            best_metric[2] = vali_loss_prc[2]
            user_model.save('best-pre2')


losses = []
prec1 = []
prec2 = []
if __name__ == '__main__':
    cmd_args = get_options()
    print('current args:{}'.format(cmd_args))

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    t1 = time.time()
    print('%s,start ' % log_time)
    dataset = Dataset(cmd_args)
    dataset.init_dataset()

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, load data completed" % log_time)

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, start to construct graph" % log_time)

    if cmd_args.user_model == 'LSTM':
        user_model = UserModelLSTM(dataset.f_dim, cmd_args)
    elif cmd_args.user_model == 'PW':
        user_model = UserModelPW(dataset.f_dim, cmd_args)
    else:
        print('using LSTM User model instead.')
        user_model = UserModelLSTM(dataset.f_dim, cmd_args)

    user_model.init_model()

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, graph completed" % log_time)

    # prepare validation data
    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, start prepare vali data" % log_time)
    out_vali_multi = dataset.prepare_validation_data(cmd_args.num_thread, dataset.vali_user)
    out_vali = dataset.data_process_for_placeholder(dataset.vali_user)
    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, prepare validation data, completed" % log_time)

    # prepare test data
    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, start prepare test data" % log_time)
    out_test = dataset.prepare_validation_data(cmd_args.num_thread, dataset.test_user)
    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, prepare test data end" % log_time)

    import pickle
    file = open('dataset_{}.pkl'.format(cmd_args.user_model), 'wb')
    pickle.dump(out_vali_multi, file, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(out_vali, file, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(out_test, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()

    best_metric = [100000.0, 0.0, 0.0]

    for ind in tqdm(range(0, len(dataset.train_user), cmd_args.batch_size)):
        end = ind + cmd_args.batch_size
        training_user = dataset.train_user[ind:end]
        out_train = dataset.data_process_for_placeholder(training_user)

        for epoch in range(cmd_args.num_iters):
            train_on_batch(user_model, out_train, out_vali, epoch)

        print('finish iteration!! completed user [%d/%d]' % (end, len(dataset.train_user)))

    # test
    user_model.restore('best-loss')
    test_loss_prc = multithread_compute_validation(user_model, out_test)
    vali_loss_prc = multithread_compute_validation(user_model, out_vali_multi)
    print("test!!!loss!!!, test: %.5f, vali: %.5f" % (test_loss_prc[0], vali_loss_prc[0]))

    user_model.restore('best-pre1')
    test_loss_prc = multithread_compute_validation(user_model, out_test)
    vali_loss_prc = multithread_compute_validation(user_model, out_vali_multi)
    print("test!!!pre1!!!, test: %.5f, vali: %.5f" % (test_loss_prc[1], vali_loss_prc[1]))

    user_model.restore('best-pre2')
    test_loss_prc = multithread_compute_validation(user_model, out_test)
    vali_loss_prc = multithread_compute_validation(user_model, out_vali_multi)
    print("test!!!pre2!!!, test: %.5f, vali: %.5f" % (test_loss_prc[2], vali_loss_prc[2]))

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    t2 = time.time()
    print("%s, end.\t time cost:%s m" % (log_time, (t2 - t1) / 60))

    file = open('analysis2_{}.pkl'.format(cmd_args.user_model), 'wb')
    import pickle
    pickle.dump(losses, file, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(prec1, file, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(prec2, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()