"""Preserve host IEEE floating-point settings around Quadrants initialization."""

from __future__ import annotations

import ctypes
from contextlib import contextmanager


@contextmanager
def preserve_host_floats():
    """Quadrants Program construction enables flush-to-zero on the calling thread.

    The standard C fenv API saves/restores the opaque host environment. A
    generously sized aligned buffer avoids depending on platform fenv_t layout.
    toyLIFE chemistry kernels use integers; SSWM host math retains subnormals.
    """
    library = ctypes.CDLL(None)
    if not hasattr(library, "fegetenv") or not hasattr(library, "fesetenv"):
        raise RuntimeError("host C floating-point environment API is unavailable")
    saved = (ctypes.c_longdouble * 32)()
    library.fegetenv.argtypes = [ctypes.c_void_p]
    library.fegetenv.restype = ctypes.c_int
    library.fesetenv.argtypes = [ctypes.c_void_p]
    library.fesetenv.restype = ctypes.c_int
    if library.fegetenv(ctypes.byref(saved)):
        raise RuntimeError("cannot save the host floating-point environment")
    try:
        yield
    finally:
        if library.fesetenv(ctypes.byref(saved)):
            raise RuntimeError("cannot restore the host floating-point environment")
