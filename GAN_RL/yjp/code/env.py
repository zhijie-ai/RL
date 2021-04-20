#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/4/19 16:38                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
# 使用GAN训练得到的强化学习环境
import pickle
from collections import defaultdict
import numpy as np
import tensorflow as tf

class Enviroment():
    def __init__(self,args):
        self.data_folder = args.data_folder
        self.random_seed = args.random_seed

        self.iterations = args.iterations
        self.k = args.k
        self.noclick_weight = args.noclick_weight
        self.time_horizon = args.time_horizon
        self.band_size = args.pw_band_size
        self.pw_dim = args.pw_dim
        self.save_dir = args.save_dir
        self.sess=tf.Session()

        np.random.seed(self.random_seed)

    def format_feature_space(self):
        with open(self.data_folder+'data_behavior.pkl','rb') as f:
            data_behavior = pickle.load(f)

        filename = self.data_folder+'user-split.pkl'
        file = open(filename, 'rb')
        train_user = pickle.load(file)
        vali_user = pickle.load(file)
        test_user = pickle.load(file)
        self.size_user = pickle.load(file)
        self.size_item = pickle.load(file)
        file.close()

        filename =self.data_folder+'embedding.pkl'
        file = open(filename, 'rb')
        self.sku_embedding = pickle.load(file)
        self.user_embedding = pickle.load(file)
        id2key_user = pickle.load(file)
        id2key_sku = pickle.load(file)

        self.f_dim = self.sku_embedding.shape[1]
        random_emb = np.random.randn(self.f_dim)

        self.sku_emb_dict = {id2key_sku.get(ind,'UNK'):emb for ind,emb in enumerate(self.sku_embedding)}
        self.user_emb_dict = {id2key_user.get(ind,'UNK'):emb for ind,emb in enumerate(self.user_embedding)}
        file.close()

        feature_space=defaultdict(list)
        for ind in range(self.size_user):
            u = data_behavior[ind][0]
            disp_id = data_behavior[ind][1]
            for id in disp_id:
                emb = self.sku_emb_dict.get(id,random_emb)
            feature_space[u].append(emb)

        return feature_space,train_user,vali_user,test_user

    def initialize_environment(self):
        print(['_k', self.k, 'iterations', self.iterations, '_noclick_weight', self.noclick_weight])

        feature_space,train_user,vali_user,test_user = self.format_feature_space()
        return feature_space,train_user,vali_user,test_user











