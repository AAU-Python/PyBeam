"""A Python package for working with 2D beam element models."""
import logging

from colorama import Fore, Style


class _ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": f"{Fore.RED}{Style.BRIGHT}",
    }

    RESET_COLOR = Style.RESET_ALL

    def format(self, record):
        if record.levelname in self.COLORS:
            colored_level = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET_COLOR}"
        else:
            colored_level = record.levelname

        return self._fmt.format(name=record.name, levelname=colored_level, message=record.msg)


_LOGGER = logging.getLogger(__name__)
_LOGGER.propagate = False
_LOGGER.setLevel(20)

_HANDLER = logging.StreamHandler()
_HANDLER.setFormatter(_ColoredFormatter("[{name}] [{levelname}] {message}", style="{"))
_LOGGER.addHandler(_HANDLER)
