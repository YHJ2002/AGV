import os
from pathlib import Path


communication = False
ROOT_DIR = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = ROOT_DIR / "algorithm" / "server4"


############################################################
####################    environment     ####################
############################################################
map_length = 25
obs_radius = 4

reward_fn = dict(
    # Keep movement slightly costly, punish idle blocking and collisions
    # harder, and make true order completion dominate generic task-node
    # completion.
    move=-0.03,
    stay_on_goal=0.0,
    stay_off_goal=-0.12,
    collision=-2.5,
    finish=0.35,
    order_complete=8.0,
    timeout=-4.0,
    other=-0.1,
)
distance_reward_scale = 0.08
train_randomize_order_seed = True
train_order_seed_base = 20260423

obs_shape = (6, 2 * obs_radius + 1, 2 * obs_radius + 1)
action_dim = 5


############################################################
####################         DQN        ####################
############################################################
num_actors = int(os.environ.get("DHC_NUM_ACTORS", "8"))
log_interval = 30
training_times = int(os.environ.get("DHC_TRAINING_TIMES", "50000"))
save_interval = int(os.environ.get("DHC_SAVE_INTERVAL", "2500"))
gamma = 0.99
batch_size = int(os.environ.get("DHC_BATCH_SIZE", "64"))
learning_rate = 3e-5
learning_starts = int(os.environ.get("DHC_LEARNING_STARTS", "8000"))
target_network_update_freq = 1000
_lr_milestone_ratios = (0.24, 0.48, 0.72, 0.88)
lr_milestones = tuple(
    sorted(
        {
            max(1, min(training_times - 1, int(training_times * ratio)))
            for ratio in _lr_milestone_ratios
        }
    )
)
lr_gamma = 0.5
save_path = str(CHECKPOINT_DIR)

training_stage_presets = {
    "bootstrap": {
        "total_orders_limit": 20,
        "order_processing_timeout": 120,
        "max_steps": 384,
        "max_episode_length": 192,
        "seq_len": 16,
    },
    "warehouse": {
        "total_orders_limit": 30,
        "order_processing_timeout": 90,
        "max_steps": 512,
        "max_episode_length": 256,
        "seq_len": 16,
    },
    "stress": {
        "total_orders_limit": 40,
        "order_processing_timeout": 60,
        "max_steps": 768,
        "max_episode_length": 320,
        "seq_len": 20,
    },
}
training_stage = os.environ.get("DHC_TRAIN_STAGE", "warehouse")
if training_stage not in training_stage_presets:
    raise ValueError(f"Unknown training stage: {training_stage}")
active_training_stage = training_stage_presets[training_stage]

max_episode_length = active_training_stage["max_episode_length"]
seq_len = active_training_stage["seq_len"]
load_model = None
actor_update_steps = 20
grad_norm_dqn = 40
forward_steps = 2
# Keep the replay reasonably large without making early training too slow.
episode_capacity = int(os.environ.get("DHC_EPISODE_CAPACITY", "256"))

prioritized_replay_alpha = 0.6
prioritized_replay_beta = 0.4

# Fixed-map training now runs directly on the project warehouse setup.
init_env_settings = (14, 25)
max_num_agents = 14
max_map_lenght = 25
pass_rate = 0.9

train_total_orders_limit = active_training_stage["total_orders_limit"]
train_order_processing_timeout = active_training_stage["order_processing_timeout"]
train_max_steps = active_training_stage["max_steps"]

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
