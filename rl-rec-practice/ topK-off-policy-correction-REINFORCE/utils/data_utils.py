#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2020/10/12 16:34                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import numpy as np
import torch

# 返回的是每个item的embedding
def make_items_tensor(items_embeddings_key_dict):
    keys = list(sorted(items_embeddings_key_dict.keys()))
    key_to_id = dict(zip(keys, range(len(keys))))
    id_to_key = dict(zip(range(len(keys)), keys))

    items_embeddings_id_dict = {}
    for k in items_embeddings_key_dict.keys():
        items_embeddings_id_dict[key_to_id[k]] = items_embeddings_key_dict[k]
    items_embeddings_tensor = torch.stack(
        [items_embeddings_id_dict[i] for i in range(len(items_embeddings_id_dict))]
    )
    return items_embeddings_tensor, key_to_id, id_to_key



# 假设items:[0,1,2,3,4,5],ratings=[1,2,3,4,5,3],则state为:[0,1,2,3,4]+[1,2,3,4,5]
#   action:5，然后变成onehot。reward:3,next_state:[1,2,3,4,5]+[2,3,4,5,3]
def batch_contstate_discaction(batch, item_embeddings_tensor, frame_size, num_items):
    """
    Embed Batch: continuous state discrete action
    """
    batch = batch[0]

    #batch:{"items": items, "rates": rates, "sizes": size, "users": idx},某个用户的数据 size:item的个数
    items_t, ratings_t, sizes_t, users_t = batch["items"], batch["ratings"], batch["sizes"], batch["users"]
    items_emb = item_embeddings_tensor[items_t]
    b_size = ratings_t.size(0)

    items = items_emb[:, :-1, :].view(b_size, -1)
    next_items = items_emb[:, 1:, :].view(b_size, -1)
    ratings = ratings_t[:, :-1]
    next_ratings = ratings_t[:, 1:]

    state = torch.cat([items, ratings], 1)
    next_state = torch.cat([next_items, next_ratings], 1)
    action = items_t[:, -1]
    reward = ratings_t[:, -1]

    done = torch.zeros(b_size)
    done[torch.cumsum(sizes_t - frame_size, dim=0) - 1] = 1# 判断是否为done的一种方法。

    one_hot_action = torch.zeros(b_size, num_items)
    one_hot_action.scatter_(1, action.view(-1, 1), 1)

    batch = {
        "state": state,
        "action": one_hot_action,
        "reward": reward,
        "next_state": next_state,
        "done": done,
        "meta": {"users": users_t, "sizes": sizes_t},
    }
    return batch