"""Two persistent agents, 256 balanced send/receive rounds, eight scenarios."""

import numpy as np

from toylife2.batch import BatchRunner
from toylife2.protocols import ControllerSpec, EpisodeSpec, PortSpec, TaskSpec

runner = BatchRunner()
agents = np.asarray([[907050, 1048465, 769001], [1005689, 644498, 941775]], dtype=np.int32)
session = runner.rollout(
    agents,
    agent_axis=0,
    task=TaskSpec(kind="signalling_game", config={"rounds": 256}),
    protocol=EpisodeSpec(horizon=512, episodes=8, chemical_updates=2),
    controller=ControllerSpec(ports=PortSpec(metabolites=("00000001", "00000011"))),
    seed=123,
)
session.run()
print(
    {
        "correct_fraction": session.aggregate_scores().tolist(),
        "judged_rounds": session.reward_count.tolist(),
        "profile": session.profile,
    }
)
runner.close()
