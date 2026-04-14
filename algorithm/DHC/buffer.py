import numpy as np
from . import configs

class SumTree:
    '''used for prioritized experience replay'''
    def __init__(self, capacity: int):
        layer = 1
        while 2**(layer-1) < capacity:
            layer += 1
        assert 2**(layer-1) == capacity, 'capacity only allow n**2 size'
        self.layer = layer
        # 用数组存整棵满二叉树：
        # 根节点下标为 0，节点 i 的左右孩子分别是 2*i+1 和 2*i+2，
        # 最后 `capacity` 个节点是叶子节点，用来存每条样本的优先级。
        self.tree = np.zeros(2**layer-1, dtype=np.float64)
        self.capacity = capacity
        self.size = 0

    def sum(self):
        assert np.sum(self.tree[-self.capacity:])-self.tree[0] < 0.1, 'sum is {} but root is {}'.format(np.sum(self.tree[-self.capacity:]), self.tree[0])
        return self.tree[0]

    def __getitem__(self, idx: int):
        assert 0 <= idx < self.capacity

        # 叶子节点直接保存 replay buffer 中每个位置对应的优先级。
        return self.tree[self.capacity-1+idx]

    def batch_sample(self, batch_size: int):
        p_sum = self.tree[0]
        interval = p_sum/batch_size

        # 分层采样：把总优先级区间均分，每段采一个前缀和，降低采样方差。
        prefixsums = np.arange(0, p_sum, interval, dtype=np.float64) + np.random.uniform(0, interval, batch_size)

        # 从根节点开始向下查找，直到落到叶子节点。
        idxes = np.zeros(batch_size, dtype=int)
        for _ in range(self.layer-1):
            nodes = self.tree[idxes*2+1]
            # 如果目标前缀和落在左子树，就往左走；
            # 否则往右走，并减去左子树累计的优先级。
            idxes = np.where(prefixsums<nodes, idxes*2+1, idxes*2+2)
            prefixsums = np.where(idxes%2==0, prefixsums-self.tree[idxes-1], prefixsums)
        
        priorities = self.tree[idxes]
        idxes -= self.capacity-1

        assert np.all(priorities>0), 'idx: {}, priority: {}'.format(idxes, priorities)
        assert np.all(idxes>=0) and np.all(idxes<self.capacity)

        return idxes, priorities

    def batch_update(self, idxes: np.ndarray, priorities: np.ndarray):
        # 把样本下标转换成叶子节点下标，并写入新的优先级。
        idxes += self.capacity-1
        self.tree[idxes] = priorities

        # 从叶子向上回溯，逐层更新父节点的优先级和。
        for _ in range(self.layer-1):
            idxes = (idxes-1) // 2
            idxes = np.unique(idxes)
            self.tree[idxes] = self.tree[2*idxes+1] + self.tree[2*idxes+2]
        
        # check
        assert np.sum(self.tree[-self.capacity:])-self.tree[0] < 0.1, 'sum is {} but root is {}'.format(np.sum(self.tree[-self.capacity:]), self.tree[0])


class LocalBuffer:
    __slots__ = ('actor_id', 'map_len', 'num_agents', 'obs_buf', 'act_buf', 'rew_buf', 'hid_buf', 
                'comm_mask_buf', 'q_buf', 'capacity', 'size', 'done')
    def __init__(self, actor_id: int, num_agents: int, map_len: int, init_obs: np.ndarray,
                capacity: int = configs.max_episode_length, 
                obs_shape=configs.obs_shape, hidden_dim=configs.hidden_dim, action_dim=configs.action_dim):
        """
        buffer for each episode
        """
        self.actor_id = actor_id
        self.num_agents = num_agents
        self.map_len = map_len

        self.obs_buf = np.zeros((capacity+1, num_agents, *obs_shape), dtype=bool)
        self.act_buf = np.zeros((capacity), dtype=np.uint8)
        self.rew_buf = np.zeros((capacity), dtype=np.float16)
        self.hid_buf = np.zeros((capacity, num_agents, hidden_dim), dtype=np.float16)
        self.comm_mask_buf = np.zeros((capacity+1, num_agents, num_agents), dtype=bool)
        self.q_buf = np.zeros((capacity+1, action_dim), dtype=np.float32)

        self.capacity = capacity
        self.size = 0

        # 第 t 步的状态存在下标 t，对应的 next_obs 存在 t+1。
        self.obs_buf[0] = init_obs
    
    def __len__(self):
        return self.size

    def add(self, q_val: np.ndarray, action: int, reward: float, next_obs: np.ndarray, hidden: np.ndarray, comm_mask: np.ndarray):
        assert self.size < self.capacity

        # 保存当前时刻的一条 transition 数据。
        self.act_buf[self.size] = action
        self.rew_buf[self.size] = reward
        self.obs_buf[self.size+1] = next_obs
        self.q_buf[self.size] = q_val
        self.hid_buf[self.size] = hidden
        self.comm_mask_buf[self.size] = comm_mask

        self.size += 1

    def finish(self, last_q_val=None, last_comm_mask=None):
        # 只有 episode 自然结束时，最后一个状态才没有 bootstrap 的 Q 值。
        if last_q_val is None:
            done = True
        else:
            done = False
            # 如果是截断结束，则补上最后一个状态的 bootstrap 值。
            self.q_buf[self.size] = last_q_val
            self.comm_mask_buf[self.size] = last_comm_mask
        
        # 返回前先把预分配数组裁剪到实际 episode 长度。
        self.obs_buf = self.obs_buf[:self.size+1]
        self.act_buf = self.act_buf[:self.size]
        self.rew_buf = self.rew_buf[:self.size]
        self.hid_buf = self.hid_buf[:self.size]
        self.q_buf = self.q_buf[:self.size+1]
        self.comm_mask_buf = self.comm_mask_buf[:self.size+1]

        
        # episode_reward_sum = float(self.rew_buf[:self.size].sum())
        # episode_length = self.size
        # print(f"\n[Episode Finish] Actor={self.actor_id} | "
        #   f"Steps={episode_length} | "
        #   f"Reward_Sum={episode_reward_sum:+8.4f} | "
        #   f"Done={done}")

        # 计算每一步的 TD 误差，作为优先经验回放的优先级信号。
        td_errors = np.zeros(self.capacity, dtype=np.float32)
        # 取每个状态下最大的 Q 值作为 bootstrap 目标的一部分。
        q_max = np.max(self.q_buf[:self.size], axis=1)
        ret = self.rew_buf.tolist() + [ 0 for _ in range(configs.forward_steps-1)]
        # 用卷积形式计算 n-step 折扣回报。
        reward = np.convolve(ret, [0.99**(configs.forward_steps-1-i) for i in range(configs.forward_steps)],'valid')+q_max
        # 用目标值和实际执行动作对应的 Q 值做差，得到 TD 误差。
        q_val = self.q_buf[np.arange(self.size), self.act_buf]
        td_errors[:self.size] = np.abs(reward-q_val).clip(1e-4)

        return  self.actor_id, self.num_agents, self.map_len, self.obs_buf, self.act_buf, self.rew_buf, self.hid_buf, td_errors, done, self.size, self.comm_mask_buf
