"""Adapted from Metaseq."""
import datetime
import logging
import logging.handlers
import os
import sys

from opt_serving.service.constants import LOGDIR


handler = None


def build_logger():
    formatter = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    logging.basicConfig(
        format=formatter,
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )
    logging.getLogger("absl").setLevel("WARNING")
    logging.getLogger("werkzeug").setLevel("WARNING")
    logger = logging.getLogger("alpa.opt_serving")

    stdout_logger = logging.getLogger('stdout')
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl
    sys.stdout.fileno = lambda: False

    stderr_logger = logging.getLogger('stderr')
    sl = StreamToLogger(stderr_logger, logging.INFO)
    sys.stderr = sl
    sys.stderr.fileno = lambda: False

    global handler
    os.makedirs(LOGDIR, exist_ok=True)
    logfile_path = os.path.join(
        LOGDIR,
        f"alpa.opt_serving.log.{datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )
    if handler is None:
        handler = logging.handlers.RotatingFileHandler(logfile_path,
                                                       maxBytes=1024 * 1024,
                                                       backupCount=100000)
        handler.setFormatter(logging.Formatter(formatter))

    for name, item in logging.root.manager.loggerDict.items():
        if isinstance(item, logging.Logger):
            item.addHandler(handler)
    # logger.addHandler(handler)
    # Set the webserver logger
    # logging.getLogger("werkzeug").addHandler(handler)
    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''
