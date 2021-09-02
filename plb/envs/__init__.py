import gym
from .env import PlasticineEnv
from gym import register

ENVS = []
for env_name in ['Move', 'Torus', 'Rope', 'Writer', "Pinch", "Rollingpin", "Chopsticks", "Table", 'TripleMove', 'Assembly']:
    for id in range(5):
        register(
            id=f'{env_name}-v{id+1}',
            entry_point=f"plb.envs.env:PlasticineEnv",
            kwargs={'cfg_path': f"{env_name.lower()}.yml", "version": id+1},
            max_episode_steps=50
        )


def make(env_name):
    env: PlasticineEnv = gym.make(env_name)
    return env
