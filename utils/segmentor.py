# -*- coding:utf-8 -*-
# Author：Leslie Dang
# Initial Data : 2019/11/19 10:12

"""
实现切词功能
    中文切词器
    中文切词、切进行词性标注（无法单独进行词性标注）
    拼音切词器
"""
import os
import jieba
self_lexicon = '../data/dict/brand_dict.txt'
if os.path.exists(self_lexicon):
    jieba.load_userdict(self_lexicon)
    jieba.initialize()

from pinyin.pinyin_tokenizer import PinyinTokenizer

# 中文切词器
def segmentor_chinese(str, model = 'cut'):
    """
    对输入的字符串进行切词，并返回切词结果
    :param str:待切词的字符串
    :param model:切词模式:【“精确模式” 、 “搜索引擎模式”】,默认采用“精确模式”
    :return:list ['word1', 'word2', ...]
    """
    # 导入自定义词典(已将自定义词库写入jieba的dict.txt中，
    # 并删除C:\Users\tangqiukui\AppData\Local\Temp\jieba.cache中的jieba.cache缓存文件
    # 无需再次加载。以加快处理速度)
    # resource_path = "../data/raw_data/new_words_for_jieba.txt"
    # if os.path.exists(resource_path):
    #     jieba.load_userdict(resource_path)

    # 1、判断切词模式
    if model == 'cut_for_search':
        seg_list = jieba.cut_for_search(str)
    else:
        seg_list = jieba.cut(str)

    return list(seg_list)

# 中文切词、切进行词性标注
def segmentor_chinese_and_pos_tag(str):
    import jieba.posseg as pseg
    return list(pseg.cut(str))


# 拼音切词器
def segmentor_pinyin(str):
    tokenizer = PinyinTokenizer()
    seg_lst = tokenizer.tokenize(str)
    return seg_lst

if __name__ == '__main__':

    # 中文切词器
    word_str = '皖酒王棉柔型浓香型白酒'
    # word_str = '法国CASTEL'
    # word_str = '衡水老白干10年酿'
    # word_str = '100年润发'
    # word_str = '张裕解百纳'
    # word_str = '澳大利亚墨尔本'
    # word_str = '今世缘'
    # word_str = '程序猿'
    # word_str = '卡思黛乐玛西'
    # word_str = '白云边陈酿酒'

    # seg_lst = segmentor_chinese(word_str, model = 'cut_for_search')
    # print(seg_lst)

    # 拼音切词
    pinyin_lst = 'xuexiao'
    seg_lst = segmentor_pinyin(pinyin_lst)
    print(seg_lst)

    # # 中文切词，并进行词性标注
    # word_str = '宣酒宣酒十年窖度ml白酒区域名酒'
    # seg_list = segmentor_chinese(word_str, 'cut_for_search')
    # print(seg_list)



