import logging
import os

from logging.handlers import TimedRotatingFileHandler

LOGDIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')

def getLogger(name='uie'):
    # Create Folder
    os.mkdir(LOGDIR) if not os.path.exists(LOGDIR) else None

    logger = logging.getLogger(name)
    # 
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(logging.DEBUG)

    fileHandler = TimedRotatingFileHandler(
        filename = os.path.join(LOGDIR, f'{name}.log'),
        when = 'midnight',
        interval = 1,
        backupCount = 7,
        encoding = 'utf-8',
        delay = True
    )

    fileHandler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S'
    )
    fileHandler.setFormatter(formatter)

    logger.addHandler(fileHandler)

    logger.propagate = False

    return logger