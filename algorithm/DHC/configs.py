communication = False


############################################################
####################    environment     ####################
############################################################
# 地图基础配置
map_length = 8
obs_radius = 4

# 奖励函数
reward_fn = dict(
    move=-0.075,
    stay_on_goal=0,
    stay_off_goal=-0.075,
    collision=-0.5,
    finish=3
)

# 单个 agent 的局部观测形状: (通道数, 高, 宽)
obs_shape = (6, 2 * obs_radius + 1, 2 * obs_radius + 1)
# 动作空间大小: 原地、上、下、左、右
action_dim = 5


############################################################
####################         DQN        ####################
############################################################

# 训练规模配置
num_actors = 1
log_interval = 10
training_times = 500
save_interval = 100
gamma = 0.99
batch_size = 8
learning_starts = 50
target_network_update_freq = 100
save_path = r'D:\chromeDownload\github\WareRover\algorithm\DHC\models'

# 轨迹与回放配置
max_episode_length = 32
seq_len = 4
load_model = None
actor_update_steps = 20
grad_norm_dqn = 40
forward_steps = 2
episode_capacity = 64

# 优先经验回放参数
prioritized_replay_alpha = 0.6
prioritized_replay_beta = 0.4

# 课程学习配置
init_env_settings = (1, 1)
max_num_agents = 16
max_map_lenght = 8
pass_rate = 0.9

# 网络结构配置
cnn_channel = 16
hidden_dim = 32

# 通信配置
# 包含自己在内，因此最多与 (max_comm_agents - 1) 个其他 agent 建立通信
max_comm_agents = 3
num_comm_layers = 2
num_comm_heads = 2


############################################################
####################         test       ####################
############################################################
test_seed = 0
num_test_cases = 5
test_env_settings = (
    (8, 1, 0.1),
)
