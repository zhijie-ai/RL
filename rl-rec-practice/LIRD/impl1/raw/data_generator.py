#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/7/26 21:03                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import csv
import pandas as pd
import random

# MovieLens 100k Generator
class DataGenerator():
    def __init__(self, datapath, itempath):
        '''
        Load data from the DB MovieLens
        List the users and the items
        List all the users historic
        '''
        self.data = self.load_datas(datapath, itempath)# 合并二者的数据。
        self.users = self.data['userId'].unique()  # list of all users
        self.items = self.data['itemId'].unique()  # list of all items
        self.histo = self.gen_histo()
        self.train = []
        self.test = []

    def load_datas(self, datapath, itempath):
        '''
        Load the data and merge the name of each movie.
        A row corresponds to a rate given by a user to a movie.

         Parameters
        ----------
        datapath :  string
                    path to the data 100k MovieLens
                    contains usersId;itemId;rating
        itempath :  string
                    path to the data 100k MovieLens
                    contains itemId;itemName
         Returns
        -------
        result :    DataFrame
                    Contains all the ratings
        '''
        data = pd.read_csv(datapath, sep='\t',
                           names=['userId', 'itemId', 'rating', 'timestamp'])
        movie_titles = pd.read_csv(itempath, sep='|', names=['itemId', 'itemName'],
                                   usecols=range(2), encoding='latin-1')
        return data.merge(movie_titles, on='itemId', how='left')

    def gen_histo(self):
        '''
        Group all rates given by users and store them from older to most recent.

        Returns
        -------
        result :    List(DataFrame)
                    List of the historic for each user
        '''
        historic_users = []
        for i, u in enumerate(self.users):
            temp = self.data[self.data['userId'] == u]
            temp = temp.sort_values('timestamp').reset_index()
            temp.drop('index', axis=1, inplace=True)
            historic_users.append(temp)
        return historic_users

    def sample_histo(self, user_histo, action_ratio=0.8, max_samp_by_user=5, max_state=100, max_action=50, nb_states=[],
                     nb_actions=[]):
        '''
        For a given historic, make one or multiple sampling.
        If no optional argument given for nb_states and nb_actions, then the sampling
        is random and each sample can have differents size for action and state.
        To normalize sampling we need to give list of the numbers of states and actions
        to be sampled.

        Parameters
        ----------
        user_histo :  DataFrame
                          historic of user
        delimiter :       string, optional
                          delimiter for the csv
        action_ratio :    float, optional
                          ratio form which movies in history will be selected
        max_samp_by_user: int, optional
                          Nulber max of sample to make by user
        max_state :       int, optional
                          Number max of movies to take for the 'state' column
        max_action :      int, optional
                          Number max of movies to take for the 'action' action
        nb_states :       array(int), optional
                          Numbers of movies to be taken for each sample made on user's historic
        nb_actions :      array(int), optional
                          Numbers of rating to be taken for each sample made on user's historic

        Returns
        -------
        states :         List(String)
                         All the states sampled, format of a sample: itemId&rating
        actions :        List(String)
                         All the actions sampled, format of a sample: itemId&rating


        Notes
        -----
        States must be before(timestamp<) the actions.
        If given, size of nb_states is the numbller of sample by user
        sizes of nb_states and nb_actions must be equals
        '''

        n = len(user_histo)
        sep = int(action_ratio * n)
        nb_sample = random.randint(1, max_samp_by_user)
        if not nb_states:
            nb_states = [min(random.randint(1, sep), max_state) for i in range(nb_sample)]
        if not nb_actions:
            nb_actions = [min(random.randint(1, n - sep), max_action) for i in range(nb_sample)]
        assert len(nb_states) == len(nb_actions), 'Given array must have the same size'

        states = []
        actions = []
        # SELECT SAMPLES IN HISTO
        for i in range(len(nb_states)):
            sample_states = user_histo.iloc[0:sep].sample(nb_states[i])
            sample_actions = user_histo.iloc[-(n - sep):].sample(nb_actions[i])

            sample_state = []
            sample_action = []
            for j in range(nb_states[i]):
                row = sample_states.iloc[j]
                # FORMAT STATE
                state = str(row.loc['itemId']) + '&' + str(row.loc['rating'])
                sample_state.append(state)

            for j in range(nb_actions[i]):
                row = sample_actions.iloc[j]
                # FORMAT ACTION
                action = str(row.loc['itemId']) + '&' + str(row.loc['rating'])
                sample_action.append(action)

            states.append(sample_state)
            actions.append(sample_action)
        return states, actions

    def gen_train_test(self, test_ratio, seed=None):
        '''
        Shuffle the historic of users and separate it in a train and a test set.
        Store the ids for each set.
        An user can't be in both set.

         Parameters
        ----------
        test_ratio :  float
                      Ratio to control the sizes of the sets
        seed       :  float
                      Seed on the shuffle
        '''
        n = len(self.histo)

        if seed is not None:
            random.Random(seed).shuffle(self.histo)
        else:
            random.shuffle(self.histo)

        self.train = self.histo[:int((test_ratio * n))]
        self.test = self.histo[int((test_ratio * n)):]
        self.user_train = [h.iloc[0, 0] for h in self.train]
        self.user_test = [h.iloc[0, 0] for h in self.test]

    def write_csv(self, filename, histo_to_write, delimiter=';', action_ratio=0.8, max_samp_by_user=5, max_state=100,
                  max_action=50, nb_states=[], nb_actions=[]):
        '''
        From  a given historic, create a csv file with the format:
        columns : state;action_reward;n_state
        rows    : itemid&rating1 | itemid&rating2 | ... ; itemid&rating3 | ... | itemid&rating4; itemid&rating1 | itemid&rating2 | itemid&rating3 | ... | item&rating4
        at filename location.

        Parameters
        ----------
        filename :        string
                          path to the file to be produced
        histo_to_write :  List(DataFrame)
                          List of the historic for each user
        delimiter :       string, optional
                          delimiter for the csv
        action_ratio :    float, optional
                          ratio form which movies in history will be selected
        max_samp_by_user: int, optional
                          Nulber max of sample to make by user
        max_state :       int, optional
                          Number max of movies to take for the 'state' column
        max_action :      int, optional
                          Number max of movies to take for the 'action' action
        nb_states :       array(int), optional
                          Numbers of movies to be taken for each sample made on user's historic
        nb_actions :      array(int), optional
                          Numbers of rating to be taken for each sample made on user's historic

        Notes
        -----
        if given, size of nb_states is the numbller of sample by user
        sizes of nb_states and nb_actions must be equals

        '''
        with open(filename, mode='w') as file:
            f_writer = csv.writer(file, delimiter=delimiter)
            f_writer.writerow(['state', 'action_reward', 'n_state'])
            for user_histo in histo_to_write:
                states, actions = self.sample_histo(user_histo, action_ratio, max_samp_by_user, max_state, max_action,
                                                    nb_states, nb_actions)
                for i in range(len(states)):
                    # FORMAT STATE
                    state_str = '|'.join(states[i])
                    # FORMAT ACTION
                    action_str = '|'.join(actions[i])
                    # FORMAT N_STATE
                    n_state_str = state_str + '|' + action_str
                    f_writer.writerow([state_str, action_str, n_state_str])