#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2021/4/8 15:11                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import pandas as pd
import datetime
import numpy as np
from GAN_RL.yjp.code.options import get_options
from utils.yjp_decorator import cost_time_minute
from sklearn.model_selection import train_test_split
import pickle
from collections import defaultdict
from GAN_RL.yjp.data.click import train as get_click_data
from GAN_RL.yjp.data.exposure import get_data as get_exposure_data


class Dataset():

    def __init__(self,args):
        self.model_type = args.user_model
        self.band_size = args.pw_band_size
        self.data_folder = args.data_folder
        self.embedding_path = args.embedding_path
        self.click_path = args.click_path
        self.exposure_path = args.exposure_path
        self.random_seed = args.random_seed

        np.random.seed(self.random_seed)

    @cost_time_minute
    def drop_dup_row(self,data,min_count=7):

        def _parse(ser):
            '''
                     user_id          sku_id                 time
            1938238  1543646  40300004041145  2021-04-07 19:39:27
            1664582  1543646  40300004041145  2021-04-07 19:39:45
            1196214  1543646  40300004041145  2021-04-07 19:40:06
            :param ser:
            :return:
            '''
            if 'time' in ser:# 利用modin.pandas 分组功能似乎有这个项
                return 2
            # 定义2min内如果用户对某个sku有多条点击即认定为重复数据

            delta = datetime.timedelta(seconds=8)
            end = ser+delta
            if len(ser) == 1:
                return 1

            # 如果在2min之内，则flag会小于等于0
            flag = [-1]+(end.values[:-1]-ser.values[1:]).tolist()
            flag = -np.sign(flag)#大于0保留

            # tmp = ser.tolist()[1:]+[0]
            # for ind,(t,t_next) in enumerate(zip(ser,tmp)):
            #     if ind==len(tmp)-1:
            #         break
            #     t<=t_next
            #     flag[ind+1]=0

            return flag


        #1. 过滤<min_count的数据
        tmp = data.groupby('user_id').size()
        data = data[np.in1d(data.user_id,tmp[tmp>min_count].index)]


        data['time']=data.time.apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
        data = data.sort_values(by=['user_id','sku_id','time'])
        data['flag'] = data.groupby(['user_id','sku_id'])['time'].transform(_parse)
        data.flag = data.flag.astype(int)

        # 过滤重复数据
        data = data[data.flag==1]
        data.drop('flag',axis=1,inplace=True)
        return data

    @cost_time_minute
    def filter_(self,data,min_count=4,max_count=80,total=10):
        #为了方便后面根据session_app字段过滤时速度较快
        data['session_app_']= data.session_app.apply(hash)
        # 去掉重复数据,把当前7天看成一个session,因为代码里，data_behavior每个用户然有多条记录，但最终也是合并成一条
        # data = data.drop_duplicates(['user_id','sku_id'])
        # data = data.sort_values(by=['user_id','time'])

        # 过滤用户在一个session中曝光的sku数量太少
        tmp = data.groupby('session_app_').size()
        data = data[np.in1d(data.session_app_,tmp[(tmp>=min_count) & (tmp<=max_count)].index)]
        # 过滤用户总的曝光的sku数量很小的情况
        tmp = data.groupby('user_id').size()
        data = data[np.in1d(data.user_id,tmp[tmp>=total].index)]
        data = data.drop('session_app_',axis=1)
        data = data.drop_duplicates()
        return data

    @cost_time_minute
    def split_dataset(self,data):
        data = data.sample(frac=1).reset_index(drop=True)
        user_ids = data.user_id.unique()
        train_user,test_user = train_test_split(user_ids,test_size=0.1)
        train_user,val_user = train_test_split(train_user,test_size=0.1)
        data['split_tag']=3

        train_ind = data[data.user_id.isin(train_user)].index
        data.loc[train_ind,'split_tag']=0
        val_ind = data[data.user_id.isin(val_user)].index
        data.loc[val_ind,'split_tag']=1
        test_ind = data[data.user_id.isin(test_user)].index
        data.loc[test_ind,'split_tag']=2

        data = data.sort_values(by='time')
        return data

    @cost_time_minute
    def preprocess_data(self):
        # click = self.filter_(self.click,min_count=7,max_count=20)
        # user = np.random.choice(self.exposure.user_id.unique(),10000,replace=False)
        # self.exposure = self.exposure[np.in1d(self.exposure.user_id,user)]

        click = self.click
        click['is_click'] = 1

        exposure = self.filter_(self.exposure)#结合后面的dqn，k=10，曝光的数据至少也是10
        exposure['is_click']=0
        # 过滤click 不在exposure中的数据
        click = pd.merge(click,exposure[['user_id','sku_id','session_app']].drop_duplicates(),on=['user_id','sku_id','session_app'])

        behavior_data = pd.concat([click,exposure])
        print('final shape:',click.shape,exposure.shape,'behavior_data.shape',behavior_data.shape)
        del click,exposure,self.click,self.exposure


        # 切分数据
        behavior_data = self.split_dataset(behavior_data)

        # 训练环境的时候，用户必须要有点击的数据，过滤没有点击操作的用户.
        tmp = behavior_data.groupby(['user_id'])['is_click'].sum()
        tmp = tmp[tmp>0]
        behavior_data = behavior_data[behavior_data.user_id.isin(tmp.index)]

        sizes = behavior_data.nunique()
        size_user = sizes['user_id']
        size_item = sizes['sku_id']

        print(sizes)
        data_behavior = [[] for _ in range(size_user)]

        train_user =[]
        vali_user = []
        test_user = []

        data_u_gb = behavior_data.groupby(by='user_id')
        for ind,user in enumerate(behavior_data.user_id.unique()):
            data_behavior[ind] = [[], [], []]
            data_behavior[ind][0] = user
            # data_u = behavior_data[behavior_data.user_id==user]
            data_u = data_u_gb.get_group(user)
            split_tag = list(data_u['split_tag'])[0]
            if split_tag == 0:
                train_user.append(user)
            elif split_tag == 1:
                vali_user.append(user)
            else:
                test_user.append(user)

            data_u_sess = data_u.groupby(by='session_app')
            for sess in data_u.session_app.unique():
                # data_sess = data_u[data_u.session_app==sess]
                data_sess = data_u_sess.get_group(sess)
                data_behavior[ind][1].append(data_sess[data_sess.is_click==0]['sku_id'].tolist())#曝光数据
                data_behavior[ind][2].append(data_sess[data_sess.is_click==1]['sku_id'].tolist())#同时间段点击的item，点击数据


        # new_features = np.eye(size_item)
        filename = self.data_folder+'data_behavior.pkl'
        file = open(filename, 'wb')
        pickle.dump(data_behavior, file, protocol=pickle.HIGHEST_PROTOCOL)
        # pickle.dump(new_features, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()

        filename = self.data_folder+'user-split.pkl'
        file = open(filename, 'wb')
        pickle.dump(train_user, file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(vali_user, file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_user, file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(size_user, file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(size_item, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()
        print('>>>>>function preprocess_data finished~~~~~<<<<<<<<<<<<')


    @cost_time_minute
    def gen_embedding(self,d_str=None):
        if d_str is None:
            d_str = datetime.datetime.now().strftime('%Y%m%d')

        path = '{}{}/'.format(self.embedding_path,d_str)
        print('>>>>>>>>>>>>>>>>>>>>>embedding path:{}'.format(path))
        sku_biases = pickle.load(open(path+'sku_biases.pickle','rb'))
        sku_embeddings= pickle.load(open(path+'sku_embeddings.pickle','rb'))
        user_biases= pickle.load(open(path+'user_biases.pickle','rb'))
        user_embeddings= pickle.load(open(path+'user_embeddings.pickle','rb'))

        id2key_user= pickle.load(open(path+'id2key_user.pickle','rb'))
        id2key_sku= pickle.load(open(path+'id2key_sku.pickle','rb'))
        id2key_user = {k:int(v) for k,v in id2key_user.items()}
        id2key_sku = {k:int(v) for k,v in id2key_sku.items()}


        sku_emb = np.concatenate((sku_embeddings, sku_biases.reshape(-1, 1)), axis=1)
        user_emb =np.concatenate((user_embeddings,user_biases.reshape(-1, 1)), axis=1)
        sku_emb = (sku_emb.T/np.linalg.norm(sku_emb,axis=1)).T
        user_emb = (user_emb.T/np.linalg.norm(user_emb,axis=1)).T

        filename = self.data_folder+'embedding.pkl'
        file = open(filename, 'wb')
        pickle.dump(sku_emb, file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(user_emb, file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(id2key_user, file, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(id2key_sku, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()

    @cost_time_minute
    def read_data(self):
        with open(self.data_folder+'data_behavior.pkl','rb') as f:
            self.data_behavior = pickle.load(f)
        filename = self.data_folder+'user-split.pkl'
        file = open(filename, 'rb')
        self.train_user = pickle.load(file)
        self.vali_user = pickle.load(file)
        self.test_user = pickle.load(file)
        self.size_user = pickle.load(file)
        self.size_item = pickle.load(file)

        np.random.shuffle(self.train_user)
        np.random.shuffle(self.vali_user)
        np.random.shuffle(self.test_user)
        file.close()


        filename =self.data_folder+'embedding.pkl'
        file = open(filename, 'rb')
        self.sku_embedding = pickle.load(file)
        self.user_embedding = pickle.load(file)
        self.id2key_user = pickle.load(file)
        self.id2key_sku = pickle.load(file)

        self.sku_emb_dict = {self.id2key_sku.get(ind,000000):emb.tolist() for ind,emb in enumerate(self.sku_embedding)}
        self.user_emb_dict = {self.id2key_user.get(ind,000000):emb.tolist() for ind,emb in enumerate(self.user_embedding)}
        file.close()

        self.f_dim = self.sku_embedding.shape[1]
        self.random_emb = np.random.randn(self.f_dim).tolist()

    def data_process_for_placeholder(self,user_set):
        if self.model_type=='PW':
            sec_cnt_x = 0
            news_cnt_short_x=0
            new_cnt_x=0
            click_2d_x=[]
            disp_2d_x=[]

            tril_indice = []
            tril_value_indice=[]

            disp_2d_split_sec=[]
            feature_clicked_x=[]

            disp_current_feature_x=[]
            click_sub_index_2d=[]

            u_idx = 0
            for u in user_set:
                u_idx+=1
                t_indice = []

                for kk in range(min(self.band_size-1,self.data_time[u]-1)):
                    t_indice += map(lambda x:[x + kk + 1 + sec_cnt_x,x + sec_cnt_x],np.arange(self.data_time[u]-(kk+1)))



                tril_indice+= t_indice
                tril_value_indice += map(lambda x:(x[0] - x[1]-1),t_indice)

                click_2d_tmp = map(lambda x:[x[0] + sec_cnt_x,x[1]],self.data_click[u])
                click_2d_tmp = list(click_2d_tmp)
                click_2d_x += click_2d_tmp

                disp_2d_tmp = map(lambda x:[x[0] + sec_cnt_x,x[1]],self.data_disp[u])
                disp_2d_tmp = list(disp_2d_tmp)

                click_sub_index_tmp = map(lambda x:disp_2d_tmp.index(x),click_2d_tmp)
                click_sub_index_tmp = list(click_sub_index_tmp)
                click_sub_index_2d +=map(lambda x:x + len(disp_2d_x),click_sub_index_tmp)
                disp_2d_x += disp_2d_tmp
                disp_2d_split_sec += map(lambda x:x[0] + sec_cnt_x,self.data_disp[u])

                sec_cnt_x +=self.data_time[u]
                news_cnt_short_x = max(news_cnt_short_x,self.data_news_cnt[u])
                new_cnt_x += self.data_news_cnt[u]
                disp_current_feature_x += map(lambda x:self.feature[u][x],[idd[1] for idd in self.data_disp[u]])
                feature_clicked_x += self.feature_click[u]

            out={}
            out['click_2d_x']=click_2d_x
            out['disp_2d_x'] = disp_2d_x
            out['disp_current_feature_x'] = disp_current_feature_x
            out['sec_cnt_x'] = sec_cnt_x
            out['tril_indice'] = tril_indice
            out['tril_value_indice'] = tril_value_indice
            out['disp_2d_split_sec'] = disp_2d_split_sec
            out['news_cnt_short_x'] = news_cnt_short_x
            out['click_sub_index_2d'] = click_sub_index_2d
            out['feature_clicked_x'] = feature_clicked_x
            return out

        else:
            news_cnt_short_x=0
            u_t_dispid = []
            u_t_dispid_split_ut = []
            u_t_dispid_feature = []

            u_t_clickid = []

            size_user = len(user_set)
            max_time = max([self.data_time[u] for u in user_set])

            click_sub_index = []

            user_time_dense = np.zeros([size_user,max_time],dtype=np.float32)
            click_feature = np.zeros([max_time,size_user,self.f_dim])

            for u_idx in range(size_user):
                u = user_set[u_idx]

                u_t_clickid_tmp = []
                u_t_dispid_tmp = []

                for x in self.data_click[u]:
                    t,click_id=x
                    click_feature[t][u_idx] = self.feature[u][click_id]
                    u_t_clickid_tmp.append([u_idx,t,click_id])
                    user_time_dense[u_idx,t]=1.0

                u_t_clickid = u_t_clickid+u_t_clickid_tmp

                for x in self.data_disp[u]:
                    t,disp_id = x
                    u_t_dispid_tmp.append([u_idx,t,disp_id])
                    u_t_dispid_split_ut.append([u_idx,t])
                    u_t_dispid_feature.append(self.feature[u][disp_id])

                click_sub_index_tmp = map(lambda x:u_t_dispid_tmp.index(x),u_t_clickid_tmp)
                click_sub_index += map(lambda x:x+len(u_t_dispid),click_sub_index_tmp)

                u_t_dispid = u_t_dispid+ u_t_dispid_tmp
                news_cnt_short_x = max(news_cnt_short_x,self.data_news_cnt[u])

            if self.model_type !='LSTM':
                print('model type not supported.using LSTM')

            out={}
            out['size_user']=size_user
            out['max_time']= max_time
            out['news_cnt_short_x'] = news_cnt_short_x
            out['u_t_dispid'] = u_t_dispid
            out['u_t_dispid_split_ut'] = u_t_dispid_split_ut
            out['u_t_dispid_feature'] = np.array(u_t_dispid_feature)
            out['click_feature'] = click_feature
            out['click_sub_index'] = click_sub_index
            out['u_t_clickid'] = u_t_clickid
            out['user_time_dense'] = user_time_dense
            return out

    def data_process_for_placeholder_L2(self,user_set):
        news_cnt_short_x=0
        u_t_dispid=[]
        u_t_dispid_split_ut=[]
        u_t_dispid_feature=[]

        u_t_clickid=[]

        size_user=len(user_set)

        click_sub_index=[]

        max_time = max([self.data_time[u] for u in user_set])

        user_time_dense = np.zeros([size_user,max_time],dtype=np.float32)
        click_feature=np.zeros([max_time,size_user,self.f_dim])

        for u_idx in range(size_user):
            u=user_set[u_idx]

            #其实可以不用采用这个相对的位置。用原始的数据。原始的id就是每个用户各自的相对序号的id。这里再次采用了每次点击对应的曝光的sku相对
            #   此次点击再做了一个相对位置的处理
            item_cnt = [{} for _ in range(self.data_time[u])]

            u_t_clickid_tmp=[]
            u_t_dispid_tmp=[]
            for x in self.data_disp[u]:
                t,disp_id=x
                u_t_dispid_split_ut.append([u_idx,t])
                u_t_dispid_feature.append(self.feature[u][disp_id])
                if disp_id not in item_cnt[t]:
                    item_cnt[t][disp_id]=len(item_cnt[t])
                u_t_dispid_tmp.append([u_idx,t,item_cnt[t][disp_id]])

            for x in self.data_click[u]:
                t,click_id=x
                click_feature[t][u_idx]=self.feature[u][click_id]
                u_t_clickid_tmp.append([u_idx,t,item_cnt[t][click_id]])
                user_time_dense[u_idx,t]=1.0

            u_t_clickid = u_t_clickid+u_t_clickid_tmp

            click_sub_index_tmp = map(lambda x:u_t_dispid_tmp.index(x),u_t_clickid_tmp)
            click_sub_index += map(lambda x:x+len(u_t_dispid),click_sub_index_tmp)

            u_t_dispid = u_t_dispid+u_t_dispid_tmp
            # news_cnt_short_x = max(news_cnt_short_x,data_news_cnt[u])
            news_cnt_short_x = self.max_disp_size

        out={}

        out['size_user']=size_user
        out['max_time']=max_time
        out['news_cnt_short_x']=news_cnt_short_x
        out['u_t_dispid']=u_t_dispid
        out['u_t_disp_split_ut']=u_t_dispid_split_ut
        out['u_t_dispid_feature']=np.array(u_t_dispid_feature)
        out['click_feature']=click_feature
        out['click_sub_index']=click_sub_index
        out['u_t_clickid']=u_t_clickid
        out['user_time_dense']=user_time_dense
        return out

    @cost_time_minute
    def format_data(self):
        # self.data_click = [[] for _ in range(self.size_user)]
        # self.data_disp = [[] for _ in range(self.size_user)]
        # self.data_time = np.zeros(self.size_user,dtype=np.int)
        # self.data_news_cnt = [[] for _ in range(self.size_user)]
        # self.feature = [[] for _ in range(self.size_user)]
        # self.feature_click = [[] for _ in range(self.size_user)]

        self.data_click = defaultdict(list)
        self.data_disp = defaultdict(list)
        self.data_time = defaultdict(list)
        self.data_news_cnt = defaultdict(list)
        self.feature = defaultdict(list)
        self.feature_click = defaultdict(list)

        for ind in range(self.size_user):
            u = self.data_behavior[ind][0]
            # (1) count number of clicks
            click_t = np.sum([len(t) for t in self.data_behavior[ind][2]])

            self.data_time[u] = click_t

            news_dict = {}
            self.feature_click[u] = np.zeros([click_t,self.f_dim])
            click_t = 0

            for event in range(len(self.data_behavior[ind][1])):
                disp_list = self.data_behavior[ind][1][event]
                pick_list = self.data_behavior[ind][2][event]

                for id in disp_list:
                    if id not in news_dict:
                        news_dict[id] = len(news_dict)

                for id in pick_list:
                    self.data_click[u].append([click_t,news_dict[id]])
                    self.feature_click[u][click_t] = self.sku_emb_dict.get(id,self.random_emb)
                    for idd in disp_list:
                        self.data_disp[u].append([click_t,news_dict[idd]])
                    click_t += 1



            self.data_news_cnt[u]=len(news_dict)
            self.feature[u]=np.zeros([self.data_news_cnt[u],self.f_dim])

            for id in news_dict:
                self.feature[u][news_dict[id]]=self.sku_emb_dict.get(id,self.random_emb)

            self.feature[u]=self.feature[u].tolist()
            self.feature_click[u] = self.feature_click[u].tolist()

    def prepare_validation_data_L2(self,num_sets,v_user):
        vali_thread_u = [[] for _ in range(num_sets)]
        size_user_v = [[] for _ in range(num_sets)]
        max_time_v = [[] for _ in range(num_sets)]
        news_cnt_short_v = [[] for _ in range(num_sets)]
        u_t_dispid_v = [[] for _ in range(num_sets)]
        u_t_dispid_split_ut_v = [[] for _ in range(num_sets)]
        u_t_dispid_feature_v = [[] for _ in range(num_sets)]
        click_feature_v = [[] for _ in range(num_sets)]
        click_sub_index_v = [[] for _ in range(num_sets)]
        u_t_clickid_v = [[] for _ in range(num_sets)]
        ut_dense_v = [[] for _ in range(num_sets)]
        for ii in range(len(v_user)):
            vali_thread_u[ii%num_sets].append(v_user[ii])

        for ii in range(num_sets):
            out = self.data_process_for_placeholder_L2(vali_thread_u[ii])
            size_user_v[ii],max_time_v[ii],news_cnt_short_v[ii],u_t_dispid_v[ii],\
            u_t_dispid_split_ut_v[ii],u_t_dispid_feature_v[ii],click_feature_v[ii],\
            click_sub_index_v[ii],u_t_clickid_v[ii],ut_dense_v[ii] = out['size_user'],\
                                                                     out['max_time'], \
                                                                     out['news_cnt_short_x'], \
                                                                     out['u_t_dispid'], \
                                                                     out['u_t_dispid_split_ut'], \
                                                                     out['u_t_dispid_feature'], \
                                                                     out['click_feature'], \
                                                                     out['click_sub_index'], \
                                                                     out['u_t_clickid'], \
                                                                     out['user_time_dense']

        out2={}
        out2['vali_thread_u']=vali_thread_u
        out2['size_user_v']=size_user_v
        out2['max_time_v']=max_time_v
        out2['news_cnt_short_v'] =news_cnt_short_v
        out2['u_t_dispid_v'] =u_t_dispid_v
        out2['u_t_dispid_split_ut_v']=u_t_dispid_split_ut_v
        out2['u_t_dispid_feature_v']=u_t_dispid_feature_v
        out2['click_feature_v']=click_feature_v
        out2['click_sub_index_v']=click_sub_index_v
        out2['u_t_clickid_v']=u_t_clickid_v
        out2['ut_dense_v']=ut_dense_v

        return out2

    @cost_time_minute
    def prepare_validation_data(self,num_sets,v_user):
        if self.model_type == 'PW':
            vali_thread_u = [[] for _ in range(num_sets)]
            click_2d_v = [[] for _ in range(num_sets)]
            disp_2d_v = [[] for _ in range(num_sets)]
            feature_v = [[] for _ in range(num_sets)]
            sec_cnt_v = [[] for _ in range(num_sets)]
            tril_ind_v = [[] for _ in range(num_sets)]
            tril_value_ind_v = [[] for _ in range(num_sets)]
            disp_2d_split_sec_v = [[] for _ in range(num_sets)]
            feature_clicked_v = [[] for _ in range(num_sets)]
            news_cnt_short_v = [[] for _ in range(num_sets)]
            click_sub_index_2d_v = [[] for _ in range(num_sets)]

            for ii in range(len(v_user)):
                vali_thread_u[ii%num_sets].append(v_user[ii])

            for ii in range(num_sets):
                out = self.data_process_for_placeholder(vali_thread_u[ii])

                click_2d_v[ii],disp_2d_v[ii],feature_v[ii],sec_cnt_v[ii],tril_ind_v[ii],tril_value_ind_v[ii],\
                disp_2d_split_sec_v[ii],news_cnt_short_v[ii],click_sub_index_2d_v[ii],feature_clicked_v[ii] = out['click_2d_x'], \
                                                                                                           out['disp_2d_x'], \
                                                                                                           out['disp_current_feature_x'], \
                                                                                                           out['sec_cnt_x'], \
                                                                                                           out['tril_indice'], \
                                                                                                           out['tril_value_indice'], \
                                                                                                           out['disp_2d_split_sec'], \
                                                                                                           out['news_cnt_short_x'], \
                                                                                                           out['click_sub_index_2d'], \
                                                                                                           out['feature_clicked_x']
            out2={}
            out2['vali_thread_u']=vali_thread_u
            out2['click_2d_x_v']=click_2d_v
            out2['disp_2d_x_v']=disp_2d_v
            out2['disp_current_feature_x_v']=feature_v
            out2['sec_cnt_x_v']=sec_cnt_v
            out2['tril_indice_v']=tril_ind_v
            out2['tril_value_indice_v']=tril_value_ind_v
            out2['disp_2d_split_sec_v']=disp_2d_split_sec_v
            out2['news_cnt_short_x_v']=news_cnt_short_v
            out2['click_sub_index_2d_v']=click_sub_index_2d_v
            out2['feature_clicked_x_v']=feature_clicked_v

            return out2
        else:
            if self.model_type !='LSTM':
                print('model type not supported.using LSTM')

            vali_thread_u = [[] for _ in range(num_sets)]
            size_user_v = [[] for _ in range(num_sets)]
            max_time_v = [[] for _ in range(num_sets)]
            news_cnt_short_v = [[] for _ in range(num_sets)]
            u_t_dispid_v = [[] for _ in range(num_sets)]
            u_t_dispid_split_ut_v = [[] for _ in range(num_sets)]
            u_t_dispid_feature_v = [[] for _ in range(num_sets)]
            click_feature_v = [[] for _ in range(num_sets)]
            click_sub_index_v = [[] for _ in range(num_sets)]
            u_t_clickid_v = [[] for _ in range(num_sets)]
            ut_dense_v = [[] for _ in range(num_sets)]
            for ii in range(len(v_user)):
                vali_thread_u[ii%num_sets].append(v_user[ii])

            for ii in range(num_sets):
                out = self.data_process_for_placeholder(vali_thread_u[ii])

                size_user_v[ii],max_time_v[ii],news_cnt_short_v[ii],u_t_dispid_v[ii],\
                u_t_dispid_split_ut_v[ii],u_t_dispid_feature_v[ii],click_feature_v[ii],\
                click_sub_index_v[ii],u_t_clickid_v[ii],ut_dense_v[ii] = out['size_user'], \
                                                                         out['max_time'], \
                                                                         out['news_cnt_short_x'], \
                                                                         out['u_t_dispid'], \
                                                                         out['u_t_dispid_split_ut'], \
                                                                         out['u_t_dispid_feature'], \
                                                                         out['click_feature'], \
                                                                         out['click_sub_index'], \
                                                                         out['u_t_clickid'], \
                                                                         out['user_time_dense']

            out2 = {}


            out2['vali_thread_u']=vali_thread_u
            out2['size_user_v']=size_user_v
            out2['max_time_v']=max_time_v
            out2['news_cnt_short_x_v']=news_cnt_short_v
            out2['u_t_dispid_v']=u_t_dispid_v
            out2['u_t_dispid_split_ut_v']=u_t_dispid_split_ut_v
            out2['u_t_dispid_feature_v']=u_t_dispid_feature_v
            out2['click_feature_v']=click_feature_v
            out2['click_sub_index_v']=click_sub_index_v
            out2['u_t_clickid_v']=u_t_clickid_v
            out2['user_time_dense_v']=ut_dense_v
            return out2

    @cost_time_minute
    def reading_raw(self,flag=True):
        if flag:
            self.click = pd.read_csv(self.click_path)
            self.exposure = pd.read_csv(self.exposure_path)
        else:
            self.click = get_click_data(flag=True)
            self.exposure = get_exposure_data(flag=True)
        print('data shape:',self.click.shape,self.exposure.shape)

    @cost_time_minute
    def init_dataset(self):
        self.reading_raw(False)
        self.gen_embedding()#'20210412'
        self.preprocess_data()
        self.read_data()
        self.format_data()
        print('---------------------------size_user:{}\tsize_item:{}\tnum of train user:{}'
              '\tnum of vali user:{}\tnum of test user:{}'.format(self.size_user,self.size_item,len(self.train_user),
                                                                  len(self.vali_user),len(self.test_user)))

    def get_batch_user(self,batch_size):
        # user_click = np.random.choice(self.train_user_click,124,replace=False).tolist()
        # user_noclick = np.random.choice(self.train_user_noclick,900,replace=False).tolist()
        # return np.array(user_click+user_noclick)
        return np.random.choice(self.train_user,batch_size,replace=False)




if __name__ == '__main__':
    args = get_options()
    dataset = Dataset(args)

    dataset.init_dataset()
    # file = open('data/dataset.obj','wb')
    # pickle.dump(dataset,file, protocol=pickle.HIGHEST_PROTOCOL)
    # file.close()