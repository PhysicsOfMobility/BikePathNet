"""
This module includes all necessary functions for the logging functionality.
"""
import time
from datetime import datetime


def log_to_file(
    file,
    txt,
    stamptime=time.localtime(),
    start=None,
    end=None,
    stamp=True,
    difference=False,
):
    """
    Logs to file with stamp time, duration and message.
    :param file: Path of the logfile
    :type file: str
    :param txt: Message to log
    :type txt: str
    :param stamptime: Time of log
    :type stamptime: time.struct_time
    :param start: Start time to calculate a duration
    :type start: float or time.struct_time
    :param end: End time to calculate a duration
    :type end: float or time.struct_time
    :param stamp: Log stamp time
    :type stamp: bool
    :param difference: Log duration
    :type difference: bool
    :return: None
    """
    if difference and not stamp:
        dur = _get_duration(start, end)
        print(f"{txt} after " f"{dur[0]:d}d{dur[1]:02d}h{dur[2]:02d}m{dur[3]:02d}s.")
        with open(file, "a+") as logfile:
            logfile.write(
                f"{txt:} after "
                f"{dur[0]:d}d{dur[1]:02d}h"
                f"{dur[2]:02d}m{dur[3]:02d}s.\n"
            )
    elif not difference and stamp:
        print(f'{time.strftime("%d %b %Y %H:%M:%S", stamptime)}: {txt}.')
        with open(file, "a+") as logfile:
            logfile.write(
                f'{time.strftime("%d %b %Y %H:%M:%S", stamptime)}:' f" {txt}.\n"
            )
    elif difference and stamp:
        dur = _get_duration(start, end)
        print(
            f'{time.strftime("%d %b %Y %H:%M:%S", stamptime)}: {txt:} after '
            f"{dur[0]:d}d{dur[1]:02d}h{dur[2]:02d}m{dur[3]:02d}s."
        )
        with open(file, "a+") as logfile:
            logfile.write(
                f'{time.strftime("%d %b %Y %H:%M:%S", stamptime)}: '
                f"{txt:} after {dur[0]:d}d"
                f"{dur[1]:02d}h{dur[2]:02d}m{dur[3]:02d}s.\n"
            )
    else:
        print(f'{time.strftime("%d %b %Y %H:%M:%S", start)}: {txt}.')
        with open(file, "a+") as logfile:
            logfile.write(f'{time.strftime("%d %b %Y %H:%M:%S", start)}: ' f"{txt}.\n")


def _get_duration(st, et):
    """
    Returns the duration between start time and end time.
    :param st: start time
    :type st: timestamp
    :param et: end time
    :type et: timestamp
    :return: days, hours, minutes, seconds
    :rtype: int, int, int, int
    """
    sd = datetime.fromtimestamp(st)
    ed = datetime.fromtimestamp(et)
    td = abs(ed - sd)
    return int(td.days), td.seconds // 3600, td.seconds // 60 % 60, td.seconds % 60
