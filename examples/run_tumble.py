"""Dimensionless periodic habitat; dt=.05, horizon=1000, eight shared scenarios."""

import numpy as np

from toylife2.batch import BatchRunner
from toylife2.protocols import EpisodeSpec, TaskSpec

runner = BatchRunner()
genes = np.asarray([[907050, 1048465, 769001], [1005689, 644498, 941775]], dtype=np.int32)
session = runner.rollout(
    genes,
    task=TaskSpec(kind="run_tumble"),
    protocol=EpisodeSpec(horizon=1000, episodes=8, dt=0.05),
    seed=123,
)
session.run()
print({"mean_exposure": session.aggregate_scores().tolist(), "profile": session.profile})
runner.close()
