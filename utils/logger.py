import os
import logging
from time import time, strftime, localtime
from datetime import datetime

levels = [logging.NOTSET,
          logging.DEBUG,
          logging.INFO,
          logging.WARNING,
          logging.ERROR,
          logging.CRITICAL]

class MyFormatter(logging.Formatter):
    converter = datetime.fromtimestamp

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            t = ct.strftime("%Y-%m-%d %H:%M:%S")
            s = "%s.%03d" % (t, record.msecs)
        return s

def create_logger(log_file=None, file_=True, console=True,
                  withtime=False, file_level=2, console_level=2,
                  propagate=True, name=None):
    """ Create a logger to write info to console and file.
    
    Params
    ------
    `log_file`: string, path to the logging file  
    `file_`: write info to file or not  
    `console`: write info to console or not  
    `withtime`: if set, log_file will be add a time prefix  
    `file_level`: file info level  
    `console_level`: console info level  
    `propagate`: if set, then message will be propagate to root logger  
    `name`: logger name, if None, then root logger will be used  

    Note: don't set propagate flag and give a name to logger is the way
    to avoid logging dublication.
    
    Returns
    -------
    A logger object of class getLogger()
    """
    if file_:
        prefix = strftime('%Y%m%d%H%M%S', localtime(time()))
        if log_file is None:
            log_file = os.path.join(os.path.dirname(__file__), prefix)
        elif withtime:
            log_file = os.path.join(os.path.dirname(log_file), prefix + "_" + os.path.basename(log_file))

        if os.path.exists(log_file):
            os.remove(log_file)

    logger = logging.getLogger(name)
    logger.setLevel(levels[2])
    logger.propagate = propagate

    formatter = MyFormatter("%(asctime)s: %(levelname).1s %(message)s")

    if file_:
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(levels[file_level])
        file_handler.setFormatter(formatter)
        # Register handler
        logger.addHandler(file_handler)

    if console:
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(levels[console_level])
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

if __name__ == "__main__":
    logger1 = create_logger(file_=False)
    logger2 = create_logger(file_=False, name="A", propagate=False)
    logger3 = create_logger(file_=False, console=False, name="A", propagate=False)

    logger3.info("abc")