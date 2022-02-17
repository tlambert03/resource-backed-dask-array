"""experimental Dask array that opens/closes a resource when computing"""
from __future__ import annotations

from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Any, ContextManager, Optional

import dask.array as da
import numpy as np

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
__author__ = "Talley Lambert"


if TYPE_CHECKING:
    from typing_extensions import Protocol

    # fmt: off
    class CheckableContext(ContextManager, Protocol):
        @property
        def closed(self) -> bool: ...  # noqa: E704
    # fmt: on


def _copy_doc(method):
    extra: str = getattr(method, "__doc__", None) or ""
    original_method = getattr(da.Array, method.__name__)
    doc = original_method.__doc__ or ""
    if extra:
        doc += extra.rstrip("\n") + "\n\n"

    method.__doc__ = doc
    return method


class ResourceBackedDaskArray(da.Array):
    _context: CheckableContext

    def __new__(
        cls,
        dask,
        name,
        chunks,
        dtype=None,
        meta=None,
        shape=None,
        _context: Optional[CheckableContext] = None,
    ):
        arr = super().__new__(
            cls, dask, name, chunks, dtype=dtype, meta=meta, shape=shape
        )
        assert _context is not None
        arr._context = _context
        return arr

    @classmethod
    def from_array(cls, arr, ctx: CheckableContext) -> ResourceBackedDaskArray:
        """Create an OpeningDaskArray with a checkable context.

        `ctx` must be a context manager that opens/closes some underlying resource (like
        a file), and has a `closed` attribute that returns the current state of the
        resource.  This subclass will take care of opening and closing the resource on
        compute.
        """
        if isinstance(arr, ResourceBackedDaskArray):
            return arr
        _a = arr if isinstance(arr, da.Array) else da.from_array(arr)
        arr = cls(
            _a.dask,
            _a.name,
            _a.chunks,
            dtype=_a.dtype,
            meta=_a._meta,
            shape=_a.shape,
            _context=ctx,
        )
        return arr

    @_copy_doc
    def compute(self, **kwargs: Any) -> np.ndarray:
        """
        Notes
        -----
        This subclass of da.Array will re-open the underlying file before compute."""
        _ctx = self._context if self._context.closed else nullcontext()
        with _ctx:
            return super().compute(**kwargs)

    def __getitem__(self, index):
        # indexing should also return an Opening Array
        return ResourceBackedDaskArray.from_array(
            super().__getitem__(index), self._context
        )

    def __getattribute__(self, name: Any) -> Any:
        # allows methods like `array.mean()` to also return an OpeningDaskArray
        attr = object.__getattribute__(self, name)
        if (
            not name.startswith("_")
            and name not in ResourceBackedDaskArray.__dict__
            and callable(attr)
        ):
            return _ArrayMethodProxy(attr, self._context)
        return attr

    def __array_function__(self, func, types, args, kwargs):
        # obey NEP18
        types = tuple(da.Array if x is ResourceBackedDaskArray else x for x in types)
        arr = super().__array_function__(func, types, args, kwargs)
        if isinstance(arr, da.Array):
            return ResourceBackedDaskArray.from_array(arr, self._context)
        return arr  # pragma: no cover

    def __reduce__(self):
        # for pickle
        return (
            ResourceBackedDaskArray,
            (
                self.dask,
                self.name,
                self.chunks,
                self.dtype,
                None,
                None,
                self._context,
            ),
            # this empty dict causes __setstate__ to be called during pickle.load
            # allowing us to close the newly created file_ctx, preventing leaked handle
            {},
        )

    def __setstate__(self, d):
        if not self._context.closed:
            self._context.__exit__(None, None, None)


class _ArrayMethodProxy:
    """Wraps method on a dask array and returns a OpeningDaskArray if the result of the
    method is a dask array.  see details in OpeningDaskArray docstring."""

    def __init__(self, method: MethodType, file_ctx: CheckableContext) -> None:
        self.method = method
        self._context = file_ctx

    def __repr__(self) -> str:
        return repr(self.method)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        with self._context if self._context.closed else nullcontext():
            result = self.method(*args, **kwds)
        if isinstance(result, da.Array):
            return ResourceBackedDaskArray.from_array(result, self._context)
        return result
