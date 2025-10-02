import random
import shutil
import textwrap
from dataclasses import fields, is_dataclass
from typing import Any, Optional

import numpy as np

BOLD = '\033[1m'
BLACK = '\033[30m'
MAG_BG = '\033[45m'
END = '\033[0m'


def cprint(element, **kwargs):
    element = f'{BOLD}{MAG_BG}{BLACK} === {element} === {END}'
    if kwargs.get('end', False):
        print(element, end='')
    elif kwargs.get('sec', True):
        print('\n' + element)
    else:
        print(element)


def line() -> None:
    print('-' * 30)


def set_seed(seed: int = 42, use_gpu: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)

    if use_gpu:
        try:
            import torch

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            raise ImportError('PyTorch is not installed. Please install it to use this feature.')


def _fmt_value(v, float_sig: int = 6) -> str:
    if isinstance(v, float):
        return f'{v:.{float_sig}g}'
    if isinstance(v, (tuple, list)):
        open_b, close_b = ('(', ')') if isinstance(v, tuple) else ('[', ']')
        return open_b + ', '.join(_fmt_value(x, float_sig) for x in v) + close_b
    if isinstance(v, dict):
        items = ', '.join(f'{_fmt_value(k, float_sig)}: {_fmt_value(val, float_sig)}' for k, val in v.items())
        return '{' + items + '}'
    if isinstance(v, str):
        return repr(v)
    return str(v)


def format_config_box(
    obj: dict,
    title: str = 'Config',
    show_types: bool = False,
    width: Optional[int] = None,
    unicode_box: bool = True,
    *,
    max_val_width: int = 56,
    max_key_width: int = 24,
) -> str:
    if unicode_box:
        H, V = '─', '│'
        TL, TR, BL, BR = '┌', '┐', '└', '┘'
        LSEP, RSEP = '├', '┤'
        TSEP, BSEP = '┬', '┴'
    else:
        H, V = '-', '|'
        TL = TR = BL = BR = '+'
        LSEP = RSEP = '+'
        TSEP = BSEP = '+'

    items: list[tuple[str, Any]] = []
    if is_dataclass(obj):
        for f in fields(obj):
            items.append((f.name, getattr(obj, f.name)))
    elif isinstance(obj, dict):
        for k in obj:
            items.append((str(k), obj[k]))
    else:
        raise TypeError('format_config_box: Please provide `dataclass` or `dict`.')

    key_max = max((len(k) for k, _ in items), default=8)
    key_w = min(max(10, key_max), max_key_width)

    FIXED = 7

    if width is None:
        try:
            term_w = shutil.get_terminal_size((100, 20)).columns
        except Exception:
            term_w = 100
        width = min(term_w, FIXED + key_w + max_val_width)
    width = max(48, min(200, width))

    val_w = width - FIXED - key_w
    if val_w < 10:
        need = 10 - val_w
        shrinkable = max(0, key_w - 10)
        take = min(need, shrinkable)
        key_w -= take
        val_w = width - FIXED - key_w
        val_w = max(10, val_w)

    top_border = TL + H * (key_w + 2) + H * (val_w + 3) + TR
    header_sep = LSEP + H * (key_w + 2) + TSEP + H * (val_w + 2) + RSEP
    bottom_border = BL + H * (key_w + 2) + BSEP + H * (val_w + 2) + BR

    title_str = f' {title} '
    inside_total_w = width - 3
    left_inside_w = key_w + 2
    inside_total = title_str.center(inside_total_w, ' ')
    title_line = V + inside_total[:left_inside_w] + ' ' + inside_total[left_inside_w:] + V

    lines = [top_border, title_line, header_sep]
    for k, v in items:
        v_str = _fmt_value(v)
        if show_types:
            v_str = f'{v_str}    ({type(v).__name__})'
        wrapped = textwrap.wrap(v_str, width=val_w) or ['']
        lines.append(f'{V} {k:<{key_w}} {V} {wrapped[0]:<{val_w}} {V}')
        for cont in wrapped[1:]:
            lines.append(f'{V} {"":<{key_w}} {V} {cont:<{val_w}} {V}')

    lines.append(bottom_border)
    return '\n'.join(lines)


def print_cfg(obj: dict, **kwargs) -> None:
    print(format_config_box(obj, **kwargs))
