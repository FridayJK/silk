import os, sys
sys.path.append("./")
from torch.utils.tensorboard import SummaryWriter

from typing import Any

import hydra

import silk.cli
from omegaconf import DictConfig
from silk.config.resolver import init_resolvers


if __name__ == "__main__":
    init_resolvers()
    @hydra.main(config_path="../../etc", config_name="config")
    def main(cfg: DictConfig) -> Any:
        silk.cli.main(cfg)


    main()