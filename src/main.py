import logging
import socket
import warnings
from datetime import datetime

import hydra
from omegaconf import DictConfig

from utils import print_cfg, set_seed

log = logging.getLogger(__name__)
warnings.simplefilter('ignore')


@hydra.main(version_base=None, config_path='config', config_name='settings')
def main(cfg: DictConfig) -> None:
    print_cfg(
        obj={
            'Current time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Hostname': socket.gethostname(),
            'Model': cfg.model.name,
            'Input dir': cfg.io.input_dir,
        },
        title='Experimental Metadata',
        show_types=False,
        unicode_box=True,
    )
    set_seed(cfg.seed, use_gpu=cfg.use_gpu)


if __name__ == '__main__':
    main()
