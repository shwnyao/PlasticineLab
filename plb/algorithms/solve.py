import argparse
import random
import numpy as np
import torch

from plb.envs import make
from plb.engine.nn.mlp import MLP
from plb.algorithms.logger import Logger

from plb.algorithms.discor.run_sac import train as train_sac
from plb.algorithms.ppo.run_ppo import train_ppo
from plb.algorithms.TD3.run_td3 import train_td3
from plb.optimizer.solver import solve_action
from plb.optimizer.solver_taichi_nn import solve_taichi_nn

RL_ALGOS = ['sac', 'td3', 'ppo']
DIFF_ALGOS = ['action', 'taichi_nn']


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default=DIFF_ALGOS + RL_ALGOS)
    parser.add_argument("--env_name", type=str, default="Move-v1")
    parser.add_argument("--path", type=str, default='./tmp')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sdf_loss", type=float, default=10)
    parser.add_argument("--density_loss", type=float, default=10)
    parser.add_argument("--contact_loss", type=float, default=1)
    parser.add_argument("--soft_contact_loss", action='store_true')

    parser.add_argument("--num_steps", type=int, default=None)

    # differentiable physics parameters
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--softness", type=float, default=666.)
    parser.add_argument("--optim", type=str, default='Adam',
                        choices=['Adam', 'Momentum'])

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    if args.num_steps is None:
        if args.algo in DIFF_ALGOS:
            args.num_steps = 50 * 200
        else:
            args.num_steps = 500000

    logger = Logger(args.path)
    set_random_seed(args.seed)

    env = make(args.env_name)

    taichi_env = env.unwrapped.taichi_env
    taichi_nn = MLP(taichi_env.simulator, taichi_env.primitives,
                    (256, 256), activation='relu')

    env.initialize(args.seed, sdf_loss=args.sdf_loss, density_loss=args.density_loss,
                   contact_loss=args.contact_loss, is_soft_contact=args.soft_contact_loss)

    if args.algo == 'sac':
        train_sac(env, args.path, logger, args)
    elif args.algo == 'action':
        solve_action(env, args.path, logger, args)
    elif args.algo == 'ppo':
        train_ppo(env, args.path, logger, args)
    elif args.algo == 'td3':
        train_td3(env, args.path, logger, args)
    elif args.algo == 'taichi_nn':
        solve_taichi_nn(env, taichi_nn, args)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
