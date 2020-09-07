#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/3/11 21:41                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import numpy as np
import pickle


def format_feature_space():
    size_user = 50000
    feature1_filename = "data/e3_user_news_feature_v2_part1.txt"
    feature2_filename = "data/e3_user_news_feature_v2_part2.txt"
    feature3_filename = "data/e3_user_news_feature_v2_part3.txt"
    feature4_filename = "data/e3_user_news_feature_v2_part4.txt"
    splitter = '/t'

    # 3. feature
    feature_space = [[] for _ in range(size_user)]
    fd = open(feature1_filename)
    for row in fd:
        row = row.split()[0]
        row = row.split(splitter)
        # key = 'u'+row[0]+'n'+row[1]
        # user_news_feature[key] = map(float, row[2].split(','))
        feature_space[int(row[0])].append(map(float, row[2].split(',')))
    fd.close()

    fd = open(feature2_filename)
    for row in fd:
        row = row.split()[0]
        row = row.split(splitter)
        # key = 'u'+row[0]+'n'+row[1]
        # user_news_feature[key] = map(float, row[2].split(','))
        feature_space[int(row[0])].append(map(float, row[2].split(',')))
    fd.close()
    fd = open(feature3_filename)
    for row in fd:
        row = row.split()[0]
        row = row.split(splitter)
        # key = 'u'+row[0]+'n'+row[1]
        # user_news_feature[key] = map(float, row[2].split(','))
        feature_space[int(row[0])].append(map(float, row[2].split(',')))
    fd.close()
    fd = open(feature4_filename)
    for row in fd:
        row = row.split()[0]
        row = row.split(splitter)
        # key = 'u'+row[0]+'n'+row[1]
        # user_news_feature[key] = map(float, row[2].split(','))
        feature_space[int(row[0])].append(map(float, row[2].split(',')))
    fd.close()

    return feature_space


def save_results(time_horizon, sim_vali_user, sim_user_reward, user_avg_reward, mean_user_avg_reward, clk_rate, mean_clk_rate, filename):

    print(['mean, reward of all experiments:', np.mean(mean_user_avg_reward)])
    print(['std, reward of all experiments:', np.std(mean_user_avg_reward)])
    print(['mean, click rate of all experiments:', np.mean(mean_clk_rate)])
    print(['std, click rate of all experiments:', np.std(mean_clk_rate)])

    with open(filename, 'wb') as handle:
        pickle.dump(sim_vali_user, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(sim_user_reward, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(user_avg_reward, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_user_avg_reward, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(clk_rate, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(mean_clk_rate, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(time_horizon, handle, protocol=pickle.HIGHEST_PROTOCOL)


def initialize_environment(sys_arg):

    # 特征dim
    f_dim = 20

    # k：
    k = int(sys_arg[1])
    iterations = int(sys_arg[2])
    noclick_weight = float(sys_arg[3])

    print(['_k', k, 'iterations', iterations, '_noclick_weight', noclick_weight])

    band_size = 20
    weighted_dim = 4
    pkl_file = open('data/split1.pkl', 'rb')
    train_user = pickle.load(pkl_file)
    vali_user = pickle.load(pkl_file)
    test_user = pickle.load(pkl_file)
    pkl_file.close()

    feature_space = format_feature_space()

    # train_user = sorted(train_user, key=lambda x: len(feature_space[x]))
    vali_user = sorted(vali_user, key=lambda x: -len(feature_space[x]))
    users_to_test = vali_user[0:100]
    time_horizon = 100

    num_test = 50

    sim_user_reward = [{} for _ in range(num_test)]
    user_avg_reward = [[] for _ in range(num_test)]
    click_rate = [[] for _ in range(num_test)]
    mean_user_avg_reward = np.zeros(num_test)
    mean_click_rate = np.zeros(num_test)

    return f_dim, k, iterations, noclick_weight, band_size, weighted_dim, train_user, vali_user, test_user, \
           feature_space, users_to_test, time_horizon, num_test, sim_user_reward, user_avg_reward, click_rate, mean_user_avg_reward, mean_click_rate


def compute_average_reward(sim_vali_user, sim_user_reward, current_best_reward):
    user_avg_reward = []
    clk_rate = []
    for j in range(len(sim_vali_user)):
        user_j_reward = sim_user_reward[sim_vali_user[j]]
        num_clk = np.sum(np.array(user_j_reward) == 0)
        clk_rate.append(1.0 - float(num_clk)/len(user_j_reward))

        cusum_reward = np.cumsum(user_j_reward)
        # user_cusum_reward.append(cusum_reward[-1])
        avg_cumsum_reward = cusum_reward / np.arange(1, len(cusum_reward)+1)
        user_avg_reward.append(avg_cumsum_reward[-1])

    current_avg_reward = np.mean(user_avg_reward)
    current_avg_clkrate = np.mean(clk_rate)
    best_or_not = ' '
    if current_avg_reward > current_best_reward:
        current_best_reward = current_avg_reward
        best_or_not = 'new best!!!!!'
    print(['mean avg reward', current_avg_reward, 'clk_rate:', current_avg_clkrate,  best_or_not])

    return user_avg_reward, current_avg_reward, clk_rate, current_avg_clkrate, current_best_reward


def sample_new_states(sim_vali_user, states, transition_p, reward_u, sim_user_reward, feature_space,  best_action_id, _k):

    remove_set = []
    for j in range(len(sim_vali_user)):
        if len(feature_space[sim_vali_user[j]]) - len(states[j]) <= _k+1:
            remove_set.append(j)

        disp_item = best_action_id[j]
        no_click = [max(1.0 - np.sum(transition_p[j, :]), 0.0)]
        prob = np.array(transition_p[j, :].tolist()+no_click)
        prob = prob / float(prob.sum())
        rand_choice = np.random.choice(disp_item + [-100], 1, p=prob)

        if sim_vali_user[j] not in sim_user_reward:
            sim_user_reward[sim_vali_user[j]] = []

        if rand_choice[0] != -100:
            states[j] += rand_choice.tolist()
            idx = disp_item.index(rand_choice[0])
            sim_user_reward[sim_vali_user[j]].append(reward_u[j][idx])
        else:
            sim_user_reward[sim_vali_user[j]].append(0)

    previous_size = len(sim_vali_user)
    states = [states[j] for j in range(previous_size) if j not in remove_set]
    sim_vali_user = [sim_vali_user[j] for j in range(previous_size) if j not in remove_set]

    return sim_vali_user, states, sim_user_reward


def sample_new_states_v2(sim_vali_user, states, transition_p, reward_u, sim_user_reward, feature_space,  best_action_id, _k):

    remove_set = []
    sampled_reward = []
    for j in range(len(sim_vali_user)):
        if len(feature_space[sim_vali_user[j]]) - len(states[j]) <= _k+1:
            remove_set.append(j)

        disp_item = best_action_id[j]
        no_click = [max(1.0 - np.sum(transition_p[j, :]), 0.0)]
        prob = np.array(transition_p[j, :].tolist()+no_click)
        prob = prob / float(prob.sum())
        rand_choice = np.random.choice(disp_item + [-100], 1, p=prob)

        if sim_vali_user[j] not in sim_user_reward:
            sim_user_reward[sim_vali_user[j]] = []

        if rand_choice[0] != -100:
            states[j] += rand_choice.tolist()
            idx = disp_item.index(rand_choice[0])
            sim_user_reward[sim_vali_user[j]].append(reward_u[j][idx])
            sampled_reward.append(reward_u[j][idx])
        else:
            sim_user_reward[sim_vali_user[j]].append(0)
            sampled_reward.append(0)

    previous_size = len(sim_vali_user)
    states_removed = [states[j] for j in range(previous_size) if j not in remove_set]
    sim_vali_user_removed = [sim_vali_user[j] for j in range(previous_size) if j not in remove_set]

    return sim_vali_user_removed, states_removed, sim_vali_user, states, sim_user_reward, np.array(sampled_reward)


def sample_new_states_for_train(training_user, states, transition_p, reward_u, feature_space,  best_action_id, _k):

    remove_set = []
    sampled_reward = []
    for j in range(len(training_user)):
        if len(feature_space[training_user[j]]) - len(states[j]) <= _k+1:
            remove_set.append(j)

        disp_item = best_action_id[j]
        no_click = [max(1.0 - np.sum(transition_p[j, :]), 0.0)]
        prob = np.array(transition_p[j, :].tolist()+no_click)
        prob = prob / float(prob.sum())
        rand_choice = np.random.choice(disp_item + [-100], 1, p=prob)

        if rand_choice[0] != -100:
            states[j] += rand_choice.tolist()
            idx = disp_item.index(rand_choice[0])
            sampled_reward.append(reward_u[j][idx])
        else:
            sampled_reward.append(0)

    previous_size = len(training_user)
    states_removed = [states[j] for j in range(previous_size) if j not in remove_set]
    training_user_removed = [training_user[j] for j in range(previous_size) if j not in remove_set]

    return states_removed, training_user_removed, training_user, states, np.array(sampled_reward), remove_set


def sample_new_states_no_reward(training_user, states, transition_p, feature_space,  best_action_id, _k):

    remove_set = []
    for j in range(len(training_user)):
        if len(feature_space[training_user[j]]) - len(states[j]) <= _k+1:
            remove_set.append(j)

        disp_item = best_action_id[j]
        no_click = [max(1.0 - np.sum(transition_p[j, :]), 0.0)]
        prob = np.array(transition_p[j, :].tolist()+no_click)
        prob = prob / float(prob.sum())
        rand_choice = np.random.choice(disp_item + [-100], 1, p=prob)

        if rand_choice[0] != -100:
            states[j] += rand_choice.tolist()

    previous_size = len(training_user)
    states_removed = [states[j] for j in range(previous_size) if j not in remove_set]
    training_user_removed = [training_user[j] for j in range(previous_size) if j not in remove_set]

    return states_removed, training_user_removed


def sample_new_states_ucb(sim_vali_user, states, transition_p, reward_u, sim_user_reward, feature_space,  best_action_id, _k):

    remove_set = []
    sampled_choice = []
    for j in range(len(sim_vali_user)):
        if len(feature_space[sim_vali_user[j]]) - len(states[j]) <= _k+1:
            remove_set.append(j)

        disp_item = best_action_id[j]
        no_click = [max(1.0 - np.sum(transition_p[j, :]), 0.0)]
        prob = np.array(transition_p[j, :].tolist()+no_click)
        prob = prob / float(prob.sum())
        rand_choice = np.random.choice(disp_item + [-100], 1, p=prob)
        sampled_choice.append(rand_choice[0])
        if sim_vali_user[j] not in sim_user_reward:
            sim_user_reward[sim_vali_user[j]] = []

        if rand_choice[0] != -100:
            states[j] += rand_choice.tolist()
            idx = disp_item.index(rand_choice[0])
            sim_user_reward[sim_vali_user[j]].append(reward_u[j][idx])
        else:
            sim_user_reward[sim_vali_user[j]].append(0)

    previous_size = len(sim_vali_user)
    states = [states[j] for j in range(previous_size) if j not in remove_set]
    sim_vali_user = [sim_vali_user[j] for j in range(previous_size) if j not in remove_set]

    return sim_vali_user, states, sim_user_reward, sampled_choice