"""
a singleton static logger to be shared by all scripts
"""
import logging

class LoggerGenerator():
    logger_dict = {}

    @staticmethod
    def get_logger(log_file_path):
        if (log_file_path not in LoggerGenerator.logger_dict.keys() ):
            print("Creating a logger that writes to {}".format(log_file_path))
            logger = logging.getLogger('myapp')
            hdlr = logging.FileHandler(log_file_path)
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
            hdlr.setFormatter(formatter)
            logger.addHandler(hdlr)
            logger.addHandler(logging.StreamHandler())
            logger.setLevel(logging.INFO)
            LoggerGenerator.logger_dict[log_file_path] = logger
            return logger
        else:
            # logger already created
            return LoggerGenerator.logger_dict[log_file_path]