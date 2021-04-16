#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2020/11/2 16:14
@Author  : tyang
@Email   : tuyang@yijiupi.com
@File    : mongo_process.py
@Description: 

"""
import pymongo
import datetime
from pymongo import UpdateOne
import utils.config as config
import pickle

conf = config.get_value('conf')
section = 'AI_MONGODB_RELEASE'  # 切换环境只要修改对应配置文件的名称即可
mongo_url = conf.get(section, 'url')  # 产品信息-标题文档对象 字典地址


def upsert(collection, data_list, ttl_time):

    ns = []
    if ttl_time is not None:
        collection.create_index([("time", pymongo.ASCENDING)], expireAfterSeconds=ttl_time)

    for i in range(0, len(data_list)):
        i_dict = data_list[i]
        if ttl_time is not None:
            i_dict['time'] = datetime.datetime.utcnow() \
                             + datetime.timedelta(seconds=ttl_time)
        if 'id' in (i_dict.keys()):
            i_dict['_id'] = i_dict.pop('id')

        ns.append(UpdateOne({'_id': i_dict['_id']}, {'$set': i_dict}, upsert=True))

    collection.bulk_write(ns)


def get(collection, user_id):
    """
    查找单个文档的记录
    :param collection:
    :param user_id:
    :return:
    """
    result = collection.find_one({'_id': user_id}, {"_id": 0})
    return result


def get_many(collection, user_ids):
    """
    查找多个user_id的文档记录
    :param collection:
    :param user_ids:
    :return: dict的集合,可指定显示字段
    """
    result = collection.find({'_id': {'$in': user_ids}}, {"_id": 0})
    return list(result)


def save_model(collection, model_name, dict_model):
    model_serialize_data = pickle.dumps(dict_model, pickle.HIGHEST_PROTOCOL)
    ids = [{'id': model_name, 'model': model_serialize_data}]
    upsert(collection, ids, None)


def load_model(collection, model_name):
    json_data = get(collection, model_name)
    model_data = pickle.loads(json_data['model'])
    return model_data


class MongoDB:
    def __init__(self, hostname='localhost', port=27017, user=None, password=None, url=mongo_url):

        if user is not None and password is not None:
            url = "mongodb://{}:{}@{}:{}" .format(user, password, hostname, port)
            self.client = pymongo.MongoClient(url)
        elif url is not None:
            self.client = pymongo.MongoClient(url)
        else:

            self.client = pymongo.MongoClient(host=hostname, port=port)

    def close(self):
        self.client.close()


if __name__ == '__main__':
    mongo_ = MongoDB('mongodb://Mongo01.yjp.com:27017,Mongo02.yjp.com:27017,Mongo03.yjp.com:27017').client
    mydb = mongo_["ai_rec"]
    mycol = mydb["commonList.rec"]

   #  mylist = [
   #      {"_id": 1, "name": "RUNOOB", "cn_name": "菜鸟教程"},
   #      {"_id": 2, "name": "Google", "address": "Google 搜索"},
   #      {"_id": 3, "name": "Facebook", "address": "脸书"},
   #      {"_id": 4, "name": "Taobao", "address": "淘宝"},
   #      {"_id": 5, "name": "Zhihu", "address": "知乎"}
   #  ]
   #
   #  mylist = [
   #      {"_id": 1, "name": "RUNOOB", "cn_name": "2222"},
   #      {"_id": 2, "name": "Google", "address": "Google 搜索"},
   #      {"_id": 3, "name": "Facebook", "address": "脸书"},
   #      {"_id": 4, "name": "Taobao", "address": "淘宝"},
   #      {"_id": 5, "name": "Zhihu", "address": "知乎"}
   #  ]
   # # x = mycol.insert_many(mylist)
   # #  x = mycol.update_many({}, {'$set':{'num':0,"_id": 9}}, upsert=True)
   #  ids =[1,2]
   #  bulk = mycol.initialize_unordered_bulk_op()
   #
   #  for i in range (0, len(ids)):
   #      bulk.find( { '_id':  ids[i]}).update({ '$set': {  "isBad" : "N" }})
   #  print(bulk.execute())
   #
   #
   #  # 输出插入的所有文档对应的 _id 值
   #
   #  myquery = {"_id": 9}
   #  for x in mycol.find(myquery):
   #      print(x)

    ids = [{'id': 12, 'hh': 1}, {'id': 13}]
    upsert(mycol, ids, 10)
