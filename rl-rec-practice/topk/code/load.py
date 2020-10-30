#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2020/10/30 10:19                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
# 用这种自定义模型的方式保存的json 或者yaml模型文件时，有以reward命名的layer，如果直接采用model_from_yaml方式加载时报错,可以重新把模型定义一遍
# 读取模型网络结构
from tensorflow.keras.models import model_from_yaml
with open("model/model_keras.yaml", "r") as f:
    yaml_string = f.read()  # 读取本地模型的yaml文件
model = model_from_yaml(yaml_string)  # 创建一个模型
filepath = "model/weights3.best.hdf5"
model.load_weights(filepath)
print(model)