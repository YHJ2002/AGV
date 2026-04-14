communication = False


############################################################
####################    environment     ####################
############################################################
# 环境基础配置
map_length = 50
# num_agents = 2
obs_radius = 4

# 奖励函数定义
reward_fn = dict(move=-0.075,
                stay_on_goal=0,
                stay_off_goal=-0.075,
                collision=-0.5,
                finish=3)

# 单个 agent 的局部观测形状: (通道数, 高, 宽)
obs_shape = (6, 2*obs_radius+1, 2*obs_radius+1)
# 动作空间大小: 原地、上、下、左、右
action_dim = 5


############################################################
####################         DQN        ####################
############################################################

# 基础训练配置
num_actors = 20
# num_actors = 1
log_interval = 10
training_times = 600000 #600000
save_interval=2000
gamma=0.99
batch_size=192
learning_starts=50000
target_network_update_freq=2000
save_path='D:\\Project\\AGVSim\\algorithm\\DHC\\models'
# 单个 episode 最长时间步
max_episode_length = 256
# RNN 回看的历史长度
seq_len = 16
load_model = None
# load_model = 'D:\\Project\\AGVSim\\algorithm\\DHC\models\\12000.pth'

max_episode_length = max_episode_length

# actor 侧同步 learner 权重的间隔
actor_update_steps = 400

# 梯度裁剪阈值
grad_norm_dqn=40

# n-step TD 的步数
forward_steps = 2

# 全局回放池按 episode 计的容量
episode_capacity = 2048

# 优先经验回放参数
prioritized_replay_alpha=0.6
prioritized_replay_beta=0.4

# 课程学习配置
init_env_settings = (1, 10)
max_num_agents = 10
max_map_lenght = 40
pass_rate = 0.9

# 网络结构配置
cnn_channel = 128
hidden_dim = 256

# 通信配置
# 包含自己在内，因此最多与 (max_comm_agents - 1) 个其他 agent 建立通信
max_comm_agents = 3

# 通信模块结构
num_comm_layers = 2
num_comm_heads = 2


# 测试配置
test_seed = 0
num_test_cases = 200
test_env_settings = ((40, 4, 0.3), (40, 8, 0.3), (40, 16, 0.3), (40, 32, 0.3), (40, 64, 0.3),
                    (80, 4, 0.3), (80, 8, 0.3), (80, 16, 0.3), (80, 32, 0.3), (80, 64, 0.3)) # map length, number of agents, density
