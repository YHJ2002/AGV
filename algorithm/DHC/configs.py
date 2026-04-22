communication = False


############################################################
####################    environment     ####################
############################################################
map_length = 25
obs_radius = 4

reward_fn = dict(
    move=-0.075,
    stay_on_goal=0,
    stay_off_goal=-0.075,
    collision=-0.5,
    finish=3,
)

obs_shape = (6, 2 * obs_radius + 1, 2 * obs_radius + 1)
action_dim = 5


############################################################
####################         DQN        ####################
############################################################
num_actors = 4
log_interval = 30
training_times = 200000
save_interval = 5000
gamma = 0.99
batch_size = 32
learning_starts = 2000
target_network_update_freq = 1000
save_path = r"D:\chromeDownload\github\WareRover\algorithm\server"

max_episode_length = 128
seq_len = 8
load_model = None
actor_update_steps = 20
grad_norm_dqn = 40
forward_steps = 2
episode_capacity = 256

prioritized_replay_alpha = 0.6
prioritized_replay_beta = 0.4

init_env_settings = (4, 15)
max_num_agents = 14
max_map_lenght = 25
pass_rate = 0.9

cnn_channel = 32
hidden_dim = 64

max_comm_agents = 3
num_comm_layers = 2
num_comm_heads = 2


############################################################
####################         test       ####################
############################################################
test_seed = 0
num_test_cases = 5
test_env_settings = (
    (14, 25, 0.1),
)
