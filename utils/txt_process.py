#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2019/12/3 15:24
@Author  : tyang
@Email   : tuyang@yijiupi.com
@File    : txt_process.py
@Description: 

"""
import json
import time
import os
from utils.yjp_ml_log import log
import pickle
from utils.redis_process import Redis
redis_ = Redis(db=7).redis_con


def load_data(file_path, class_type):
    if class_type == list:
        if os.path.exists(file_path):
            return texts_to_list(file_path)
        else:
            return []
    elif class_type == dict:
        if os.path.exists(file_path) or os.path.exists(file_path+'.pkl'):
            return load_dic(file_path)
        else:
            return dict()


def save_data(data_object, file_path):
    class_type = type(data_object)
    if class_type == list:
        save_texts(data_object, file_path, "")
    elif class_type == dict:
        save_dic(data_object, file_path)


def del_model_file(path):
    if os.path.exists(path):  # 如果文件存在
        # 删除文件，可使用以下两种方法。
        os.remove(path)


def read_sql(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            sql = f.read()
    except Exception as err:
        print(err)
    return sql


def save_texts(texts, texts_path, sep=" "):
    time0 = time.time()
    del_model_file(texts_path)
    f = open(texts_path, 'w', encoding='utf-8')
    for text in texts:
        f.write(str(sep.join(text))+"\n")
    f.close()
    log.logger.info("save texts {} , cost time {}. ".format(texts_path, time.time() - time0))


def write_data(sentences, texts_path):
    time0 = time.time()
    del_model_file(texts_path)
    out = open(texts_path, 'w', encoding='utf-8')
    for sentence in sentences:
        out.write(sentence + "\n")
    out.close()
    log.logger.info("save texts {} , cost time {}. ".format(texts_path, time.time() - time0))


def load_texts(texts_path):
    f = open(texts_path, 'r', encoding='utf-8')
    result_list = f.read().split("\n")
    result = [ii.split(" ") for ii in result_list]
    f.close()
    return result


def texts_to_list(texts_path):
    f = open(texts_path, 'r', encoding='utf-8')
    result_list = f.read().split("\n")
    f.close()
    return result_list


def save_dic(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    log.logger.info('save {} 成功'.format(name + '.pkl'))

#
# def load_dic(name):
#     try:
#         with open(name + '.pkl', 'rb') as f:
#             return pickle.load(f)
#         log.logger.info('load {} 成功'.format(name + '.pkl'))
#     except Exception as e:
#         log.logger.error(e)
#         return dict()
#         log.logger.info('load {} 失败, 返回空字典'.format(name + '.pkl'))
#


def load_dic(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    log.logger.info('load {} 成功'.format(name + '.pkl'))


def save_dict_to_redis(dict_path):

    dict_ = load_dic(dict_path)
    key_ = os.path.split(dict_path)[-1].split('.')[0]
    redis_.set(key_, json.dumps(dict_))
    log.logger.info("save ' {} ' to redis successfully, and key is ' {} ' ".format(dict_path, key_))
    redis_.close()


def load_dict_from_redis(key_):
    dict_str = redis_.get(key_)
    redis_.close()
    if dict_str is None:
        log.logger.info("load ' {} ' from redis fail,because no value of {} ".format(key_, key_))
        return dict()
    else:
        data_dict = json.loads(dict_str)
        # # data_bytes = red.get(key_)
        # strs_ = bytes.decode(dict_str)
        # data_dict = eval(strs_)
        log.logger.info("load ' {} ' from redis successfully".format(key_))
        return data_dict


def dict_2_pickle(dic, dump_to_file_path):
    with open(dump_to_file_path, 'wb') as file:
        pickle.dump(dic, file)


def pickle_2_dict(load_file_path):
    with open(load_file_path, 'rb') as file:
        dic = pickle.load(file)
    return dic
