import argparse
import random
import numpy as np
import torch
import taichi as ti

from plb.envs import make
from plb.engine.nn.mlp import MLP

from plb.optimizer.solver_taichi_nn import solve_taichi_nn

ti.init(arch=ti.gpu, debug=False, fast_math=True)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf_loss", type=float, default=10)
    parser.add_argument("--density_loss", type=float, default=10)
    parser.add_argument("--contact_loss", type=float, default=1)
    parser.add_argument("--soft_contact_loss", action='store_true')

    parser.add_argument("--num_steps", type=int, default=50 * 200)

    # differentiable physics parameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--softness", type=float, default=666.)
    parser.add_argument("--optim", type=str, default='Adam',
                        choices=['Adam', 'Momentum'])

    args = parser.parse_args()

    return args


def tune_taichi_nn():
    args = get_args()

    for env_name in ['Move-v1']:
        args.env_name = env_name

        for af in ['Tanh', 'LeakyReLU']:
            for seed in [0, 10, 20]:
                args.seed = seed
                set_random_seed(seed)

                env = make(env_name)

                taichi_env = env.unwrapped.taichi_env
                taichi_nn = MLP(taichi_env.simulator, taichi_env.primitives,
                                (256, 256), activation='relu')

                env.initialize(seed, sdf_loss=args.sdf_loss, density_loss=args.density_loss,
                               contact_loss=args.contact_loss, is_soft_contact=args.soft_contact_loss)

                solve_taichi_nn(env, taichi_nn, args)

                ti.reset()


if __name__ == '__main__':
    tune_taichi_nn()
