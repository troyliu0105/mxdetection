import logging

import coloredlogs

__all__ = ['setup_logger']


def setup_logger(filename, level='INFO', fmt=None):
    fh = logging.FileHandler(filename, mode='a+')
    logging.root.addHandler(fh)
    if not fmt:
        fmt = '%(asctime)s %(name)s[%(process)d] %(levelname)s %(message)s'
    coloredlogs.install(level=level, fmt=fmt)
