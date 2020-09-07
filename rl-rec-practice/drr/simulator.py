#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/5/22 下午5:35                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

import numpy as np

# to predict an item’s feedback that the user never rates before
class Simulator():
    def __init__(self,item_embeded,whole_data,alpha=0.3,sigma=0.9):
        self.alpha = alpha
        self.sigma=sigma
        self.item_embed = item_embeded
        self.whole_data = whole_data
        #self.init_state=self.reset()
        # self.current_state = self.init_state
        # self.rewards, self.group_sizes, self.avg_states, self.avg_actions = self.avg_group()

    def state_module(self,user,embedding,item_index):
        item_mat=[]
        state=[]
        user_item=[]

        width = len(item_index)

        for i in item_index:
            item_mat.append(embedding[str(i)])

        for i in range(width):
            for j in range(i+1,width):
                state.append(np.multiply(item_mat[i],item_mat[j]))
            user_item.append(np.multiply(item_mat[i],user))

        state.extend(user_item)
        return state

    # item_index:[12,52,11,55,66]
    def state_module_item(self,embedding,item_index):
        item_mat=[]
        width = len(item_index)# 第四次的state_float中的5个元素,即5个itemId

        for i in item_index:
            item_mat.append(embedding[str(i)])#5*19

        for i in range(width):
            for j in range(i+1,width):
                item_mat.append(np.multiply(item_mat[i],item_mat[j]))

        return item_mat


    def reset(self,user_idx,user_embed):
        # this one is for user-item embedding
        # mat = self.state_module(user_embed, item_embed, data.loc[user_idx]['state_float'][4])
        # 把5个item的embedding构造成15*19的矩阵.
        mat = self.state_module_item(self.item_embed,self.whole_data.loc[user_idx]['state_float'][4])
        init_state = np.array(mat).reshape((15,19))
        self.current_state = init_state
        return init_state

    def step(self,action,user_idx):
        # 获取action所对应的embedding
        actions = np.array(self.item_embed[action])
        simulate_rewards = self.simulate_reward((self.current_state.reshape((1,15*19)),
                                                 action.reshape((1,1*19))),user_idx)

        actions = actions.reshape(1,1*19)
        for i ,r in enumerate(simulate_rewards):# if simulate_reward>0 ,then change the state
            if r >0:
                # self.current_state.append(action[1])

                tmp = np.append(self.current_state,actions[i].reshape((1,19)),axis=0)
                tmp = np.delete(tmp,0,axis=0)
                #self.current_state=tmp[np.newaxis, :]
                self.current_state = tmp

        return simulate_rewards,self.current_state

    #(curren_state,action),user_idx,相当于是自己定义的一种计算reward的方式
    def simulate_reward(self, pair, user_idx):

        """use the average result to calculate simulated reward.
        Args:
            pair (tuple): <state, action> pair one shape:(1,15*19),another (1,19)
        Returns:
            simulated reward for the pair.
        """

        probability = []
        denominator = 0.
        max_prob = 0.
        result = 0.
        simulate_rewards = ""
        new_data = ((self.whole_data.loc[user_idx]).to_frame()).T
        # calculate simulated reward in normal way
        #[[1, 2, 3], [4, 5, 6]]  [1, 2]  [2, 3](sf,af,rf)
        for idx, row in new_data.iterrows():
            state_values = row['state_float']
            action_values = row['action_float']
            length = len(action_values)
            for i in range(4, length):
                # 当前的是state
                item_mat = pair[0][0][0:5 * 19] #取的是s中的地一个list,15*19的一个向量,取的是前5个item_id构成的向量
                # 某一个state中的5个item_id
                curr_embed = {key: self.item_embed[str(key)] for key in state_values[i]}
                curr_state = np.array(list(curr_embed.values())).reshape(19 * 5, 1)
                curr_embed1 = {action_values[i]: self.item_embed[str(action_values[i])]}
                curr_action = np.array(list(curr_embed1.values())).reshape(19, 1)

                numerator = self.alpha * (
                        # 5*19*19*5,当前的state和历史的state
                        np.dot(item_mat, curr_state)[0] / (
                            # 当前state的2范数
                            np.linalg.norm(item_mat, 2) * np.linalg.norm(curr_state, 2))
                        # 当前的action和历史的
                ) + (1 - self.alpha) * (
                                    np.dot(pair[1], curr_action)[0] / (
                                        np.linalg.norm(pair[1], 2) * np.linalg.norm(curr_action, 2))
                            )
                probability.append(numerator)
                denominator += numerator
        probability /= denominator
        simulate_rewards = [new_data.loc[user_idx]['reward_float'][int(np.argmax(probability))]]

        return simulate_rewards