#-*- coding: UTF-8 -*-
import os
import logging.handlers


# log config
log_level = logging.DEBUG
log_file_dir = "log/"
log_file_path = log_file_dir + "pylog"


def check_dirs():
    if not os.path.exists(log_file_dir):
        os.makedirs(log_file_dir)


# log init
check_dirs()
logger = logging.getLogger()
formatter = logging.Formatter('[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] %(message)s')
# store 10 days' log
file_handler = logging.handlers.TimedRotatingFileHandler(log_file_path, when='MIDNIGHT', backupCount=10)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(log_level)
