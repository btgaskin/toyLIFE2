"""Run with `.venv/bin/python examples/temporal_probe.py` from the repository."""

import numpy as np

from toylife2.batch import BatchRunner
from toylife2.protocols import ControllerSpec, EpisodeSpec, TaskSpec

runner = BatchRunner()
genes = np.asarray([[907050, 1048465, 769001], [1005689, 644498, 941775]], dtype=np.int32)
session = runner.rollout(
    genes,
    task=TaskSpec(kind="temporal_probe", config={"sequence": [0, 8, 8, 0], "delay": 1}),
    protocol=EpisodeSpec(horizon=32, chemical_updates=3, episodes=4),
    controller=ControllerSpec(warmup=2),
    seed=123,
)
session.run()
print({"mean_reward": session.aggregate_scores().tolist(), "profile": session.profile})
runner.close()
