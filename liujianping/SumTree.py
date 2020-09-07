#-*- encoding=utf-8 -*-
'''
__author__:'xiaojie'

CreateTime:
        2019/2/23 11:27
 =================知行合一=============
'''

import numpy as np


class Tree(object):
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity  # capacity是叶子节点个数，
        self.tree = np.zeros(2 * capacity)  # 从1开始编号[1,capacity]
        self.data = np.zeros(capacity + 1, dtype=object)  # 存叶子节点对应的数据data[叶子节点编号id] = data

    def add(self, p, data):
        idx = self.write + self.capacity
        self.data[self.write + 1] = data
        self._updatetree(idx, p)
        self.write += 1
        if self.write > self.capacity:
            self.write = 0

    def _updatetree(self, idx, p):
        change = p - self.tree[idx]
        self._propagate(idx, change)
        self.tree[idx] = p

    def _propagate(self, idx, change):
        parent = idx // 2
        self.tree[parent] += change  # 更新父节点的值，是向上传播的体现
        if parent != 1:
            self._propagate(parent, change)

    def _total(self):
        return self.tree[1]

    def get(self, s):
        idx = self._retrieve(1, s)
        index_data = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[index_data])

    def _retrieve(self, idx, s):
        left = 2 * idx
        right = left + 1
        if left >= (len(self.tree) - 1):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)  # 往左孩子处查找
        else:
            return self._retrieve(right, s - self.tree[left])  # 往右孩子处查找


tree = Tree(5)
tree.add(1, 3)
tree.add(2, 4)
tree.add(3, 5)
tree.add(4, 6)
tree.add(6, 11)

print(tree.get(4))  # (8, 4.0, 6)
print(tree.tree)
print(tree.data)
print(tree._total())