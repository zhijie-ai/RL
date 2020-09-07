import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_VOLUME = 1000e8
MAX_AMOUNT = 3e10
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000
MAX_DAY_CHANGE = 1

INITIAL_ACCOUNT_BALANCE = 10000


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):# df代表某个股票文件的数据,sh.600036.招商银行.csv,按时间排序
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(19,), dtype=np.float16)

    def _next_observation(self):
        obs = np.array([
            self.df.loc[self.current_step, 'open'] / MAX_SHARE_PRICE,# 今日开盘价格
            self.df.loc[self.current_step, 'high'] / MAX_SHARE_PRICE,#最高价
            self.df.loc[self.current_step, 'low'] / MAX_SHARE_PRICE,#最低价
            self.df.loc[self.current_step, 'close'] / MAX_SHARE_PRICE,#今日收盘价格
            self.df.loc[self.current_step, 'volume'] / MAX_VOLUME,#成交数量
            self.df.loc[self.current_step, 'amount'] / MAX_AMOUNT,#成交金额
            self.df.loc[self.current_step, 'adjustflag'] / 10,#复权状态
            self.df.loc[self.current_step, 'tradestatus'] / 1,#交易状态
            self.df.loc[self.current_step, 'pctChg'] / 100,#涨跌幅(百分比)
            self.df.loc[self.current_step, 'peTTM'] / 1e4,#滚动市盈率
            self.df.loc[self.current_step, 'pbMRQ'] / 100,#市净率
            self.df.loc[self.current_step, 'psTTM'] / 100,#滚动市销率
            self.df.loc[self.current_step, 'pctChg'] / 1e3,#
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ])
        return obs

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = random.uniform(
            self.df.loc[self.current_step, "open"], self.df.loc[self.current_step, "close"])

        action_type = action[0]#1:买入，2:卖出,3:保持
        amount = action[1]#买入或卖出百分比

        if action_type < 1:#self.balance应该是当前本金
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)#股数
            shares_bought = int(total_possible * amount)#买入多少股
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price#买入股数花的钱

            self.balance -= additional_cost
            self.cost_basis = (
                prev_cost + additional_cost) / (self.shares_held + shares_bought)
            #shares_held：持有的股数
            self.shares_held += shares_bought

        elif action_type < 2:
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        done = False

        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'open'].values) - 1:
            self.current_step = 0  # loop training
            # done = True

        delay_modifier = (self.current_step / MAX_STEPS)

        # profits
        reward = self.net_worth - INITIAL_ACCOUNT_BALANCE
        reward = 1 if reward > 0 else -100

        if self.net_worth <= 0:
            done = True

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self, new_df=None):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # pass test dataset to environment
        if new_df:
            self.df = new_df

        # Set the current step to a random point within the data frame
        # self.current_step = random.randint(
        #     0, len(self.df.loc[:, 'open'].values) - 6)
        self.current_step = 0

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        print('-'*30)
        print('Step: {}'.format(self.current_step))
        print('Balance: {}'.format(self.balance))
        print('Shares held: {} (Total sold: {})'.format(self.shares_held,self.total_shares_sold))
        print('Avg cost for held shares: {} (Total sales value: {})'.format(self.cost_basis,self.total_sales_value))
        print('Net worth: {} (Max net worth: {})'.format(self.net_worth,self.max_net_worth))
        print('Profit: {}'.format(profit))
        return profit
