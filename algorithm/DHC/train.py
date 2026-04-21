import os
import random
import time

import torch
import numpy as np
import ray

from .worker import GlobalBuffer, Learner, Actor
from . import configs

os.environ["OMP_NUM_THREADS"] = "1"
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def main(num_actors=configs.num_actors, log_interval=configs.log_interval):
    # 初始化 Ray，buffer / learner / actor 都作为远程对象运行。
   # ray.init()
    ray.init(
    ignore_reinit_error=True,
    _temp_dir="D:/ray_tmp",
    include_dashboard=False,
    logging_level="ERROR"
)
    # ray.init(local_mode=True)   # 本地调试时可以打开

    buffer = GlobalBuffer.remote()
    learner = Learner.remote(buffer)
    time.sleep(1)

    # 为不同 actor 分配不同的 epsilon，形成从探索到利用的梯度。
    if num_actors == 1:
        actors = [Actor.remote(0, 0.4, learner, buffer)]
    else:
        actors = [Actor.remote(i, 0.4**(1 + (i / (num_actors - 1)) * 7), learner, buffer) for i in range(num_actors)]

    for actor in actors:
        actor.run.remote()

    # 回放池预热到足够样本量之前，只进行收集和状态打印。
    while not ray.get(buffer.ready.remote()):
        time.sleep(5)
        ray.get(learner.stats.remote(5))
        ray.get(buffer.stats.remote(5))

    print('start training')
    buffer.run.remote()
    learner.run.remote()
    
    done = False
    while not done:
        time.sleep(log_interval)
        done = ray.get(learner.stats.remote(log_interval))
        ray.get(buffer.stats.remote(log_interval))
        print()


if __name__ == '__main__':
    main()
