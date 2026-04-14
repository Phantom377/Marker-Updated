import sys as _sys
import os as _os

# Force Python to use only the installed marker package (0.2.15 in venv),
# not this local source directory (1.10.2 which requires Python 3.10+).
_site_marker = None
for _p in _sys.path:
    _candidate = _os.path.join(_p, 'marker')
    if (
        'site-packages' in _p
        and _os.path.isdir(_candidate)
        and _candidate != _os.path.dirname(__file__)
    ):
        _site_marker = _candidate
        break

if _site_marker:
    __path__ = [_site_marker]
