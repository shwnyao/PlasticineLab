import taichi as ti
import numpy as np
from yacs.config import CfgNode as CN

from .optim import Optimizer, Adam, Momentum
from ..engine.taichi_env import TaichiEnv
from ..config.utils import make_cls_config
from plb.algorithms.logger import Logger

OPTIMS = {
    'Adam': Adam,
    'Momentum': Momentum
}


class Solver:
    def __init__(self, env: TaichiEnv, logger=None, cfg=None, **kwargs):
        self.cfg = make_cls_config(self, cfg, **kwargs)
        self.optim_cfg = self.cfg.optim
        self.env = env
        self.logger = logger

    def solve(self, init_actions=None, callbacks=()):
        env = self.env
        if init_actions is None:
            init_actions = self.init_actions(env, self.cfg)
        # initialize ...
        optim = OPTIMS[self.optim_cfg.type](init_actions, self.optim_cfg)
        # set softness ..
        env_state = env.get_state()
        self.total_steps = 0

        def forward(sim_state, action):
            if self.logger is not None:
                self.logger.reset()

            env.set_state(sim_state, self.cfg.softness, False)
            with ti.Tape(loss=env.loss.loss):
                for i in range(len(action)):
                    env.step(action[i])
                    self.total_steps += 1
                    loss_info = env.compute_loss()
                    if self.logger is not None:
                        self.logger.step(
                            None, None, loss_info['reward'], None, i == len(action)-1, loss_info)
            loss = env.loss.loss[None]
            return loss, env.primitives.get_grad(len(action))

        best_action = None
        best_loss = 1e10

        actions = init_actions
        for iter in range(self.cfg.n_iters):
            self.params = actions.copy()
            loss, grad = forward(env_state['state'], actions)
            if loss < best_loss:
                best_loss = loss
                best_action = actions.copy()
            actions = optim.step(grad)

            self.logger.summary_writer.writer.add_histogram('grad', grad, iter)
            for callback in callbacks:
                callback(self, optim, loss, grad)

        env.set_state(**env_state)
        return best_action

    @staticmethod
    def init_actions(env, cfg):
        action_dim = env.primitives.action_dim
        horizon = cfg.horizon
        if cfg.init_sampler == 'uniform':
            return np.random.uniform(-cfg.init_range, cfg.init_range, size=(horizon, action_dim))
        else:
            raise NotImplementedError

    @classmethod
    def default_config(cls):
        cfg = CN()
        cfg.optim = Optimizer.default_config()
        cfg.n_iters = 100
        cfg.softness = 666.
        cfg.horizon = 50

        cfg.init_range = 0.
        cfg.init_sampler = 'uniform'
        return cfg


def solve_action(env, path, logger, args):
    import os
    import cv2
    import imageio

    exp_name = f'action_{args.env_name}'
    path = f'data/{exp_name}/{exp_name}_s{args.seed}'
    os.makedirs(path, exist_ok=True)
    logger = Logger(path)
    os.makedirs(path, exist_ok=True)

    env.reset()
    taichi_env: TaichiEnv = env.unwrapped.taichi_env
    T = env._max_episode_steps
    solver = Solver(taichi_env, logger, None,
                    n_iters=(args.num_steps + T-1)//T, softness=args.softness, horizon=T,
                    **{"optim.lr": args.lr, "optim.type": args.optim, "init_range": 0.0001})

    action = solver.solve()

    with imageio.get_writer(f"{path}/output.gif", mode="I") as writer:
        for idx, act in enumerate(action):
            env.step(act)
            img = env.render(mode='rgb_array')
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            writer.append_data(img)
