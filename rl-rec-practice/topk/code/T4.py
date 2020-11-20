#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2020/10/22 16:54                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
from tensorflow import keras
import numpy as np

top_words = 10000
(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=top_words)
max_review_length = 200
print(np.min([ np.min(i) for i in X_train]))
x_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_review_length,padding='post')
print(x_train.shape)