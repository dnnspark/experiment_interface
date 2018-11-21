import logging

formatter = logging.Formatter(
    fmt='%(levelname)s %(asctime)s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    )

def get_train_logger(log_file=None):
    
    logger = logging.getLogger('train_logger')

    _setup_logger(logger, log_file)
    return logger


def get_test_logger(log_file=None):
    
    logger = logging.getLogger('get_test_logger')

    _setup_logger(logger, log_file)
    return logger
    
def _setup_logger(logger, log_file):

    if len(logger.handlers) == 0:
        # not set up yet.
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_file is not None:
        fh = logging.FileHandler(filename=log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
