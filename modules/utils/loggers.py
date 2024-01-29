import logging

def create_logger(name="general", level = logging.DEBUG):
    """
    This function configures and creates a logger.

    Args:
        name (str, optional): name of the logger. Defaults to "general".
        level (logging level, optional): logging level. Defaults to logging.DEBUG.

    Returns:
        logger: created logger.
    """
    # create logger
    logger=logging.getLogger(name)
    logger.setLevel(level)

    # create console handler and set level to debug
    if len(logger.handlers) == 0:
        ch = logging.StreamHandler()
    else:
        ch = logger.handlers[0]
    ch.setLevel(level)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    if len(logger.handlers) == 0:
        logger.addHandler(ch)

    return logger
