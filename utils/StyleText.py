from typing import Optional
from colorama import Fore, Back, Style
import datetime


def style_warning(text: str = f"[W]"):
    time_ = {datetime.datetime.now().strftime('[%H:%M:%S]')}
    return Fore.YELLOW + text + Style.RESET_ALL


def style_error(text: str = "[E]"):
    return Fore.RED + text + Style.RESET_ALL


def style_pass(text: str = "[P]"):
    return Fore.LIGHTGREEN_EX + text + Style.RESET_ALL


def style_info(text: str = "[I]"):
    return f"\033[94m{text}\033[0m"


if __name__ == "__main__":
    print(style_warning())
    print(style_error())
    print(style_pass())
    print(style_info())

    from colorama import Fore, Back, Style
