#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/5/22 11:00                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
# https://github.com/BoogieCloud/Deep-RL-Recommendation-System/blob/master/DRR.py

import tensorflow as tf
import json
# Importing some more libraries
import pandas as pd
import numpy as np
import os
import argparse
import pprint as pp
import random
from collections import deque
from sklearn.preprocessing import minmax_scale
from scipy.special import comb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import time

u_cols = ['user_id','age','sex']