"""Load a single file out of a vendored baseline tree without touching ``sys.path``.

Two of the baseline trees ship a top-level package called ``Network``, namely
``redi/image/Network`` and ``di4c/maskgit-pytorch/Network``. Both define a ``transformer``
submodule and the two are not interchangeable, since the ReDi one takes a required
``randmask`` argument that the Di4C one does not have. The usual
``sys.path.insert(0, tree); from Network.transformer import ...`` idiom therefore resolves to
whichever tree happened to be imported first in the process, and the loser silently gets the
wrong class. That bites as soon as one process builds the VQGAN (ReDi) and the MaskGIT teacher
(Di4C) together, which is exactly what an evaluation run does.

``load_file_module`` sidesteps the ambiguity by loading a named ``.py`` file directly under a
private module name, so nothing is registered as ``Network`` and no ordering matters. It only
works for files that have no intra-package imports of their own, which is true of
``di4c/maskgit-pytorch/Network/transformer.py``. Anything that genuinely needs its package
around it, such as the ReDi ``Network.Taming`` tree, still has to go through ``sys.path``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_file_module(name: str, path: str | Path) -> ModuleType:
    """Import ``path`` as a standalone module registered under ``name``.

    Repeated calls with the same ``name`` return the cached module, so the loaded classes stay
    identical across calls and ``isinstance`` keeps working.
    """
    if name in sys.modules:
        return sys.modules[name]
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"cannot import {name}, no such file: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build a module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        del sys.modules[name]
        raise
    return mod
