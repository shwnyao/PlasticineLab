from .solver import *


class SolverTaichiNN:
    def __init__(self, env: TaichiEnv, nn, logger, cfg=None, **kwargs):
        self.cfg = make_cls_config(self, cfg, **kwargs)
        self.cfg.optim.lr *= 0.001
        self.cfg.optim.bounds = (-np.inf, np.inf)
        print(self.cfg.optim)
        self.logger = logger
        self.optim_cfg = self.cfg.optim
        self.horizon = self.cfg.horizon
        self.env = env
        self.nn = nn

    def solve(self, callbacks=()):
        env = self.env
        # assert hasattr(env, 'nn'), "nn must be an element of env .."

        # nn = env.nn  # assume that nn has been initialized.. nn.initialize
        nn = self.nn

        # initialize ...
        params = nn.get_params()
        optim = OPTIMS[self.optim_cfg.type](nn.get_params(), self.optim_cfg)

        # set softness ..
        env_state = env.get_state()
        self.total_steps = 0

        def forward(sim_state, params):
            nn.set_params(params)
            env.set_state(sim_state, self.cfg.softness, False)

            if self.logger is not None:
                self.logger.reset()

            with ti.Tape(loss=env.loss.loss):
                for i in range(self.horizon):
                    nn.set_action(i, env.simulator.substeps)
                    env.step()
                    self.total_steps += 1
                    loss_info = env.compute_loss()
                    self.logger.step(
                        None, None, loss_info['reward'], None, (i == self.horizon-1), loss_info)
            loss = env.loss.loss[None]
            # return loss, env.nn.get_grad()
            return loss, nn.get_grad()

        best_action = None
        best_loss = 1e10

        for iter in range(self.cfg.n_iters):
            self.params = params
            loss, grad = forward(env_state['state'], params)
            self.logger.summary_writer.writer.add_histogram('grad', grad, iter)
            if loss < best_loss:
                best_loss = loss
                best_action = params.copy()
            params = optim.step(grad)
            for callback in callbacks:
                callback(self, optim, loss, grad)

        env.set_state(**env_state)
        return best_action

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


def solve_taichi_nn(env, taichi_nn, args):
    import os
    import cv2
    import imageio
    import torch
    from torch import nn

    exp_name = f'taichi-nn_{args.env_name}'
    path = f'data/{exp_name}/{exp_name}_s{args.seed}'
    os.makedirs(path, exist_ok=True)
    logger = Logger(path)

    class MLP(nn.Module):
        def __init__(self, inp_dim, oup_dim):
            super(MLP, self).__init__()
            self.l1 = nn.Linear(inp_dim, 256)
            self.l2 = nn.Linear(256, 256)
            self.l3 = nn.Linear(256, oup_dim)

        def forward(self, x):
            x = torch.relu(self.l1(x))
            x = torch.relu(self.l2(x))
            return self.l3(x)

    T = env._max_episode_steps
    mlp = MLP(
        env.unwrapped.observation_space.shape[0], env.unwrapped.action_space.shape[0])

    params = []
    for i in mlp.parameters():
        print(i.shape)
        params.append(i.data.cpu().numpy().reshape(-1))
    params = np.concatenate(params)

    env.reset()

    taichi_env = env.unwrapped.taichi_env
    solver = SolverTaichiNN(taichi_env, taichi_nn, logger, None,
                            n_iters=(args.num_steps + T-1)//T, softness=args.softness, horizon=T,
                            **{"optim.lr": args.lr, "optim.type": args.optim, "init_range": 0.0001})

    nn = taichi_nn
    nn.set_params(params)
    p2 = nn.get_params()
    assert np.abs(p2 - params).max() < 1e-9
    print("Initialize", p2.sum(), params.sum())

    params = solver.solve()
    nn.set_params(params)
    os.makedirs(path, exist_ok=True)
    taichi_env.set_copy(True)

    with imageio.get_writer(f"{path}/output.gif", mode="I") as writer:
        for idx in range(50):
            nn.set_action(0, taichi_env.simulator.substeps)
            taichi_env.step(None)
            img = taichi_env.render(mode='rgb_array')
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
            writer.append_data(img)
