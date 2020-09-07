# 对于PG算法来说，ACTOR要么输出一个动作，要么输出动作的概率分布。
# 对于随机策略，输出的是概率分布，
#   如果是离散动作，输出的是所有动作的概率，softmax
#     如actor_critic.py，policy_gradient.py
#   如果是连续动作，输出一个正太分布，然后从该正太分布中采样，如Sample-PPO.py
# 对于DPG，输出的是某个动作，如ddpg.py，因为是确定性动作，所以在探索的时候加上一个噪声，保证探索性

# 连续动作，如：Box(1,)，可以输出11维的action，只是和离散的11个action不一样。离散的
#   11个action要softmax，然后取最大。而这种做法不需要softmax，输出11维向量，然后max，或者random，只取其中一维。
#   如5.1—Double_DQN\run_Pendulum.py,因为动作是1维的，虽然输出了11维的action，但最终要转换成一维的action
#     在5.1的Double_DQN中，实际上是一维的action，传入的n_feature=11，在choose_action方法中，将输出11维的
# action 最终取最大，只得到一个值，之后又将这个值放缩到[-2,2]中



# DQN系列算法:off policy，离线策略。
#   离散：输出所有动作的概率，softmax,莫凡老师的代码的DQN部分
#         如5.2中run_MountainCar.py中离散动作
#   连续：输出多个动作，比如11维，然后在11维中取最大，或者随机探索。
#       如5.1中的run_Pendulum.py中的，虽然只有1维，但在算Q_value的时候，输出一个11维的向量，然后在选动作的时候
#       根据epsilon-greedy策略来选择动作。此时的动作是某个整数值，然后经过变化，放缩到-2~2之间。

# PG系列算法：是一个online算法。只有一个PI。选择动作的PI和target都是同一个
#       离散：输出每个动作的概率。然后根据概率选动作。7_Policy_gradient_softmax/RL_brain.py,
#       连续：# 输出的是正太分布的均值和方差。输出一个正太分布，然后从该正太分布中采样，如Sample-PPO.py，或者
#             8—Actor_Critic_Advatage中的AC_continue_Pendulum.py中
# AC算法：
#     Critic网络，有拟合V的，也有拟合Q的
#         代码中的三个AC版本都是拟合V，刘建平老师的博客中，虽然Critic的更新用的是Q的TD误差，可是在
#          代码实现用的却是V的TD误差。AC算法中，A的评估点有多种嘛。
#         拟合Q的:https://github.com/princewen/tensorflow_practice/blob/master/RL/Basic-AC-Demo/AC.py
#     在AC_CartPole.py(左右水平杆的平衡,2个动作)实现中，ACTOR:log_prob = tf.log(self.acts_prob[0,self.a])

# epsilon_greey 策略，当产生的0-1之间的随机数大于epsilon时，随机选,如dqn.py
#     但是在5.1_Double_DQN中又是相反的，感觉<epsilon时随机选更靠谱
#     还有另外一种实现思路，采用伯努利实现的方式
#       if np.random.binomial(1, EPSILON) == 1:
#             action = np.random.choice(ACTIONS)

# A2C：在AC算法中,Actor的评估点有5种，在知乎上另一篇AC的文章中有6种，其实差不多。
#   在原本的A2C算法中，Actor的估计用Advantage来估计，而Advantage的估计要Q和V，这样的话需要2个网络，
#   所以Q = R+γ*V(s^prom)来估计Q,这样的话就只需要一个网络了，进而V的loss为关于V的TD-error

# DQN是学习Q网络，然后根据贪婪策略来选择动作。记住:DQN用的Q，即动作价值函数
# 基于值函数的强化学习算法的基本思想是根据当前的状态，计算采取每个动作的价值，然后根据价值贪心的选择动作。
#   如果我们省略中间的步骤，即直接根据当前的状态来选择动作，就引出了策略梯度的方法。
#   DQN是value based。AC = PG+value based
# value based 直接拟合的V或者Q。DQN:拟合Q或者V之后，通过贪婪法选择动作。
#   另一种是AC:Critic拟合的直接是V(s)或者Q(s,a)。想象
#   一下，在PG算法找那个，Vt的计算需要一个完整的eposide。现在通过Critic，直接就得到了V或者Q。
# DDPG:DQN+AC,并且ACTOR采用的是确定性策略。适合连续动作，和DQN不同的是，DQN通过贪婪策略选择动作。而DDPG
#   因为有目标Actor,在计算目标Q值的时候，不再通过贪婪得到，而是用目标ACTOR中得到a',然后在计算代入到目标
#   Critic中计算目标Q值

# 在看了杨益民老师关于使用DRN解决新闻推荐的论文之后的一点感悟：
# 传统的RL中，比如DQN,我们用策略来选择动作，从ac的角度，有actor和critic。但是在这篇论文中提出了一种新的
#   思路来解决推荐问题，在一般的关于RL推荐的问题中，比如《强化学习在京东推荐中的探索》中将S定义为用户历史浏览的item，
#   A定义为推荐给用户的商品列表K,reward即为用户的反馈(s,a,r)通过simulator来计算。然后通过DQN来建模,模型训练完成后
#   使用actor来做推荐，相当于把Q网络丢弃掉了。而在杨益民老师的这篇文章中，提供了一种新的思路，也是采用DDQN建模，
#   用Q网络计算reward(reward的定义参考原论文),得到reward之后，直接排序推荐，由于采用DDQN的网络，一个当做
#   探索策略，另一个当做exploit策略。2个列表融合形成一个推荐列表。不仅如此，该算法还在线上实时更新参数，做到了
#   真正的动态捕捉用户的兴趣爱好。
# 总结:
#   1.之前的用RL做推荐的系统，用agent来推荐，Q网络在训练完成后基本上不用了，而本论文中提出的观点是，直接用reward
#       来做推荐
#   2.在一般用RL做的推荐系统，一般采用AC算法或者DQN来建模，而本论文中的思想是，用DQN来建模，但是策略并不是epsilon-greedy
#       策略，而是DBGD。且和AC的策略不一样，AC中的actor是自己是实现的一个神经网络，而本论文中的DBGD策略，实际上
#       就是Q网络。相当于把策略和Q网络集成到一起了，为什么可以集成呢？在原始DQN中，策略选动作，Q网络计算Q值，
#       而本论文中，Q网络输出的Q值虽然是reward，但我们可以直接用它来排序。Q(s,a)，通过Q网络计算(s,a)的值，
#   3.在一般用RL做的推荐系统中，网络训练完之后就不在训练了，直接使用actor做推荐，而在本论文中，线上使用时还是可以
#       训练的，一直在训练，动态捕捉用户的兴趣。

# Q-learning之前的算法，都是使用e-greddy策略，在DQN系列算法中，也是使用的e-greedy策略。在DQN中，拟合的是value
#     function,可以拟合V也可以拟合Q.然后就是PG算法，拟合的Actor。输出全部动作的概率或者分布(连续动作),
#   然后就是AC系列算法，拟合Actor的同时拟合Vaule(Q或者V)。在刘建平老师的actor_critic实现中，critic拟合的是V，
#   虽然Actor输出的是所有的action的概率，但在critic中，由于拟合的V，直接根据state，计算V，所以，actor输出动作的
#   多少对critic没有影响。
# 强化学习中，在model-free模型中，由于没有P，因此一般有个仿真环境，让agent与环境交互得到下一个s,但在PG算法中，
#   有种算法叫reinforce算法，需要完整episode，与此同时，可以把episode看做RNN中的时间序列，采用RNN做推荐，
#   我想，在之前看到的那篇关于offline top-K的推荐中，应该是采用一个完整的episode来训练RL的吧。
#   在得到了episode之后，可以用DNN来训练2个回归模型，拟合(s,a)->s',(s,a)->r。
#   如果state是用户的历史浏览记录的话，比如list-wise那篇论文，建立了一个仿真器来当环境。
#   top-K那篇论文，虽然没有仿真环境，单用RNN来计算s'=f(s,a),s,a都为向量。应该是有一个完整的episode


# list-wise论文:historycal ,s1,s2,s3,s4...,action：输出的是一个权重向量，采用的是AC算法，Actor输出的是
#   一个权重向量。history采用Word2vec思想embedding。由于根据Actor输出的vector来选择动作，起到类似探索的作用
#   需要一个仿真环境。输出在历史记录中没有出现的(s,a)对的reward。
# negative feedback:虽然论文的例子是输出一个，可是论文中说一个的案例extend mutiple items的情况。对召回排序。
#   采用的是DQN，embedding+GRU来计算Q-value，根据论文的描述，在state为st时，给她推荐的item即为at，
#   似乎有一个原始的策略b(st)，而at是根据历史session中的log来得到的。所以似乎不存在探索的情况。或者使用b(st)来探索。
# DRN:虽然提出的架构是一种新思路，但是模型似乎是直接在线上与用户环境交互的，实际使用中，成本太高。仅仅只有借鉴价值
#   不过线下可以先通过log日志训练一个Q函数，训练好之后，部署到线上，然后再利用论文中的思想
#   部署到线上使用Q和Q^prog来做探索

# Value-based算法，agent在训练的时候，根据e-greedy策略用Q(s,a)来选择动作，在训练完成后用max Q来使用
# PG方法，输出的都是一个概率分布(softmax,gussian),训练完成后，直接从该分布采样，离散的话，根据权重采样，
#   连续的话直接从分布中采样

# 一般在推荐系统中的强化学习中，都是有log日志当训练数据的，所以由(s,a)是知道s'的。比如在当前s状态下，在log中，
#   给用户推荐一个item，此item是在日志中出现过的，用户要么点击，要么略过。所以s'就知道了。因此，模拟器只需要模拟
#   在日志中未出现过的(s,a)得到r即可。(其实，最好的模拟器，应该可以根据log数据，模拟出，给定(s,a),返回s'和r。即
#   模拟真实的environment。一般来说，用户即environment。如果得到用户的environment，那么就好训练强化学习模型了。

# list-wise:
#     仿真器：建立一个二维矩阵，行代表用户，列代表item。比如有20个用户，就有20行，每一行代表用户历史浏览或点击或收藏
#     的item。假设用历史操作过的20个item为state，则用for循环构造(s,a,r)对。参考list-wise论文。似乎s是不变的。
#     Actor:输出一个维度为k的向量，由该向量计算出k个最相似的物品

# nagetive-feedback：GRU+DQN,s为最近一段时间点击或下单的按时间序列排序的商品。训练强化学习的DQN模型。用GRU拟合Q

# 训练仿真器:
#   1. list-wise论文中的思路。注意生成(s,a,r)的训练对
#   2. 有一篇专门采用GAN来训练仿真器的论文。s我用户最近一段时间有过操作的item，按时间序列排序。input=concat(en,Fn)
#     en为item的embedding，Fn为用户对该item的feedback转换成的embedding。将2个embedding拼接成input输入到LSTM中。
#   3.阿里的那篇用GAN训练环境的论文:
#       用日志数据训练P和r，当agent给用户推荐k个items的时候，此时的P，就等同于P(s,At)的一个分布。P的定义本来
#       由(s,a)->r,s'的过程。当agent给user推荐k个物品时，用户会发生点击行为，下一个状态根据s的定义也能知道。所以最
#       关键的是r。从这个角度，这篇论文和京东那篇关于GAN来训练仿真器的论文是差不多的。具体参考该论文的algorithm2算法

# 之前的RL算法中，是根据用户的日志数据来训练agent的。根据日志，给用户推荐某个item a，会得到该item对应的反馈。同时会
#   得到下一个状态s。而根据阿里的那篇论文思路。先用日志训练一个env，再根据此env来训练DQN模型。假设t时刻给用户随机推送
#   了k个item，从env中根据训练好的φ(st,At)选出一个at，得到st+1,同时根据reward function算出该at对应的r，存入
#   到M中。这和上面的思路，根据日志中真实的数据a而得到r不是一样的吗？
#   不一样，上面那种不存在探索的情况，先根据log训练强化学习参数，然后再扔到online中去训练，第二种存在探索的情况。
# 所以说，在用RL做推荐的系统中，最重要的是根据日志数据训练一个reward ，这样就可以算出任意一个(s,a) pair对的reward了。
#   不能仅仅训练一个reward，对未出现的item，如果仅仅训练reward模型，虽然可以得到reward，但是无法得到下一个状态。
#   之前的，比如list-wise文章，是根据reward来判断是点击还是不点击的。如果点击就知道下一个状态了。所以，用RL做推荐
#   还是得知道P和reward。

# 生成replay buffer中的数据:
#   list-wise那篇论文中，对用户的session循环。根据ACTOR生成action，通过模拟器得到对应的
#       reward，然后在根据每个action对应的reward得到下一个s，存入M中。
#   negative feedback:初始state为用户的历史(s^+,s^_),用另一个策略支持action得到rt,根据r得到下一个状态s ，然后
#       再存入到M中,用GRU来计算Q
#   Top-k那篇论文没有说明怎么训练的的步骤，用LSTM来计算下一个state。用LSTM的变种来计算下一个状态
#   大规模离散空间论文:用的是类似gym的环境。唯一的创新点是Actor处理大规模离散空间的思路。

# 如果actor会选择一个没见过的action，如list-wise论文中的actor，那么此时就需要仿真器来计算reward了。
#   如果仅仅根据日志记录来决定action，那么这种情况虽然可以训练RL agent，但似乎不存在探索的操作。比如根据log日志，
#   给用户推荐历史记录中的item，可以知道她的reward，那没见过的item呢？就计算不了对应的reward了。虽然可以根据log
#   中的数据训练Q函数，Q函数计算召回集中的items，这样似乎退化成根据log数据训练一个回归模型，似乎不属于强化学习范畴了。
#   训练一个reward仿真器，得到仿真器之后，就可以采用大规模处理离散动作论文的思路，采用ac算法来训练强化学习模型了。
#   其次是根据阿里的那篇论文来用GAN对P和R建模，得到2个模型。这样是强化学习的思路。后面的GAIL的逆强化学习？

# 1.训练仿真器
# 2.由trajectory，根据刘建平老师文章中的思路，分别训练2个模型,(S,A)->S',(S,A)->R

