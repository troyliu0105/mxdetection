import logging
import os

import coloredlogs

__all__ = ['setup_logger']


def setup_logger(filename, level='INFO',
                 fmt='%(asctime)s %(name)s[%(process)d] %(levelname)s %(message)s'):
    if os.path.isfile(filename):
        fh = logging.FileHandler(filename, mode='a+')
        logging.root.addHandler(fh)
    coloredlogs.install(level=level, fmt=fmt)
