#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/7/26 21:06                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import numpy as np

class Embeddings:
    def __init__(self, item_embeddings):
        self.item_embeddings = item_embeddings

    def size(self):
        return self.item_embeddings.shape[1]

    def get_embedding_vector(self):
        return self.item_embeddings

    def get_embedding(self, item_index):
        return self.item_embeddings[item_index]

    def embed(self, item_list):
        return np.array([self.get_embedding(item) for item in item_list])