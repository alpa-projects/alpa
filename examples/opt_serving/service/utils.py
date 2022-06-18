"""Adapted from Metaseq."""
import socket
import logging
import sys
import os


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


def build_logger():
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )
    logger = logging.getLogger("metaseq_cli.interactive")
    return logger
