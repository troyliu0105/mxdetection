import logging

import coloredlogs

__all__ = ['setup_logger']


def setup_logger(filename, level='INFO'):
    fh = logging.FileHandler(filename, mode='a+')
    logging.root.addHandler(fh)
    coloredlogs.install(level=level)
