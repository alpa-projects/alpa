"""Adapted from Metaseq."""
import socket
import logging
import logging.handlers
import sys
import os
import datetime

from examples.opt_serving.service.constants import LOGDIR


def normalize_newlines(s: str):
    """
    normalizes new lines, i.e. '\r\n' to '\n'
    """
    # note that web browsers send \r\n but our training data uses \n.
    return s.replace("\r\n", "\n").replace("\r", "\n")


def get_my_ip():
    """
    returns ip / hostname of current host
    """
    return socket.gethostbyname(socket.gethostname())


def encode_fn(generator, x):
    """tokenization"""
    assert generator.tokenizer is not None
    return generator.tokenizer.encode(normalize_newlines(x))


handler = None


def build_logger():
    formatter = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    logging.basicConfig(
        format=formatter,
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )
    print(os.environ.get("LOGLEVEL", "INFO").upper())
    logging.getLogger("absl").setLevel("WARNING")
    logger = logging.getLogger("alpa.opt_serving")
    global handler
    os.makedirs(LOGDIR, exist_ok=True)
    logfile_path = os.path.join(
        LOGDIR,
        f"alpa.opt_serving.log.{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    if handler == None:
        handler = logging.handlers.RotatingFileHandler(logfile_path,
                                                       maxBytes=1024 * 1024,
                                                       backupCount=100000)
        handler.setFormatter(logging.Formatter(formatter))
    logger.addHandler(handler)

    # Set the webserver logger
    logging.getLogger("werkzeug").addHandler(handler)
    return logger
