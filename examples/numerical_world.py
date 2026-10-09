"""A portable Python callback world whose state is an explicit numerical tree.

Run `.venv/bin/python examples/numerical_world.py`. Its importable factory accepts
the standard config/shape/agents/random/dt arguments. Such Python callbacks run on
the host; chemistry still uses the selected Quadrants backend.
"""

import numpy as np


class World:
    def __init__(self, config, shape, agents, random, dt):
        self.shape = shape
        self.agents = agents
        self.random = random

    def reset(self):
        return {"tick": 0, "target": np.zeros(self.shape, np.int32)}

    def observe(self, state):
        return {"dose": state["target"] * 8}

    def advance(self, state, actions):
        reward = (actions == state["target"]).astype(float)
        target = (self.random.uniform(state["tick"], "example-target") >= 0.5).astype(np.int32)
        return {"tick": state["tick"] + 1, "target": target}, reward, {}


def world_factory(**kwargs):
    return World(**kwargs)


if __name__ == "__main__":
    from toylife2.batch import BatchRunner
    from toylife2.protocols import EpisodeSpec, TaskSpec

    runner = BatchRunner()
    session = runner.rollout(
        [907050, 1048465, 769001],
        TaskSpec(kind="python", entrypoint="numerical_world:world_factory"),
        EpisodeSpec(horizon=20, episodes=4),
        seed=123,
    )
    session.run()
    print({"mean_reward": session.aggregate_scores().tolist(), "profile": session.profile})
    runner.close()
