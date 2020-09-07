#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2018/12/11 15:46
 =================知行合一=============
'''
import numpy as np

def f1():
    matrix = np.matrix([[0.9, 0.075, 0.025], [0.15, 0.8, 0.05], [0.25, 0.25, 0.5]], dtype=float)
    vector1 = np.matrix([[0.1, 0.7, 0.2]], dtype=float)
    for i in range(100):
        vector1 = vector1 * matrix
        print("Current round:", i + 1)
        print(vector1)

def f2():
    matrix = np.matrix([[0.9, 0.075, 0.025], [0.15, 0.8, 0.05], [0.25, 0.25, 0.5]], dtype=float)
    for i in range(100):
        matrix = matrix * matrix
        print("Current round:", i + 1)
        print(matrix)

def f3():#不是马尔科夫状态转移矩阵
    matrix = np.matrix([[0.9, 0.07, 0.025], [0.15, 0.8, 0.5], [0.5, 0.25, 0.5]], dtype=float)
    for i in range(20):
        matrix = matrix * matrix
        print("Current round:", i + 1)
        print(matrix)


def f4():
    matrix = np.matrix([[0.8, 0.1, 0.1], [0.5, 0.2, 0.3], [0.3, 0.3, 0.4]], dtype=float)
    vector1 = np.matrix([[0.1, 0.7, 0.2]], dtype=float)
    for i in range(100):
        vector1 = vector1 * matrix
        print("Current round:", i + 1)
        print(vector1)

if __name__ == '__main__':
    f1()