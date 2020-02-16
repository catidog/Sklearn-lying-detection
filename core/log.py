import functools
import logging
import os
import sys
from termcolor import colored

class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")

        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."

        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)

        if record.levelno == logging.WARNING:
            prefix = colored("[WARNING]", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR :
            prefix = colored("[ERROR]", "red", attrs=["blink", "underline"])
        elif record.levelno == logging.DEBUG:
            prefix = colored("[DEBUG]", "blue", attrs=["blink"])
        elif record.levelno == logging.INFO:
            prefix = colored("[INFO]", "yellow", attrs=["blink"])
        elif record.levelno == logging.CRITICAL:
            prefix = colored("[CRITICAL]", "cyan", attrs=["blink","underline"])
        else:
            return log
        return prefix + " " + log


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def setup_logger(
    output=None, distributed_rank=0, *, color=True, name="baseline", abbrev_name=None
):
    """
    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger
        abbrev_name (str): an abbreviation of the module, to avoid long names in logs.
            Set to "" to not log the root module in logs.
            By default, will abbreviate "detectron2" to "d2" and leave other
            modules unchanged.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = "Sun&Moon Tutorial" if name == "baseline" else name

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    # stdout logging: master only
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = filename + ".rank{}".format(distributed_rank)
        os.mkdir(os.path.dirname(filename))

        fh = logging.StreamHandler(open(filename,'a'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)
    return logger
