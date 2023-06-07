# -*- coding: utf-8 -*-
import logging

def setup_logger(logger_name='logger'):
    '''
    logging.info("info")
    logging.debug("debug")
    logging.warning("warning")
    logging.error("error")
    logging.critical("critical")
    '''
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    logger = logging.getLogger(logger_name)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG) 
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(f"./logs/{logger_name}.log")
    file_handler.setLevel(logging.DEBUG) 
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger