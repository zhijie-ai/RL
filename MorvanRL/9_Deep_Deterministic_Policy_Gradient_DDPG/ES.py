#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2019/10/15 13:52                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
file_num = 0
with open('nohup.out',encoding='utf8') as f:
    for line in f:
        if '文件数目' in line :
            file_num += int(line.split('：')[1])

print(file_num)