from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Callable, Collection

import dask.array as da
from dask.array.core import Array as DaskArray

if TYPE_CHECKING:
    from types import MethodType

if TYPE_CHECKING:
    from typing import ContextManager, TypeVar

    import numpy as np
    from typing_extensions import Protocol

    C = TypeVar("C", bound=Callable)

    # fmt: off
    class CheckableContext(ContextManager, Protocol):
        @property
        def closed(self) -> bool: ...
    # fmt: on


def _copy_doc(method: C) -> C:
    extra: str = getattr(method, "__doc__", None) or ""
    original_method = getattr(DaskArray, method.__name__)
    doc = original_method.__doc__ or ""
    if extra:
        doc += extra.rstrip("\n") + "\n\n"

    method.__doc__ = doc
    return method


def resource_backed_dask_array(
    arr: DaskArray, ctx: CheckableContext
) -> ResourceBackedDaskArray:
    """Create an ResourceBackedDaskArray with a checkable context.

    Parameters
    ----------
    arr : DaskArray
        A dask array
    ctx : CheckableContext
        a context manager that:
            1) opens the underlying resource on `__enter__`
            2) closes the underlying resource on `__exit__`
            3) implements a `.closed` attribute that reports the open-state of
               the resource.

    Returns
    -------
    ResourceBackedDaskArray
    """
    return ResourceBackedDaskArray.from_array(arr, ctx)


class ResourceBackedDaskArray(DaskArray):
    _context: CheckableContext

    def __new__(  # type: ignore
        cls,
        dask,
        name,
        chunks,
        dtype=None,
        meta=None,
        shape=None,
        _context: CheckableContext | None = None,
    ):
        arr = super().__new__(
            cls, dask, name, chunks, dtype=dtype, meta=meta, shape=shape
        )
        if _context is None:
            raise TypeError(
                "Must provide _context when creating a ResourceBackedDaskArray"
            )
        arr._context = _context
        return arr

    @classmethod
    def from_array(cls, arr: Any, ctx: CheckableContext) -> ResourceBackedDaskArray:
        """Create a ResourceBackedDaskArray with a checkable context.

        `ctx` must be a context manager that:
            1) opens the underlying resource on `__enter__`
            2) closes the underlying resource on `__exit__`
            3) implements a `.closed` attribute that reports the open-state of the
               resource.
        """
        if isinstance(arr, ResourceBackedDaskArray):
            return arr
        _a = arr if isinstance(arr, DaskArray) else da.from_array(arr)
        new_arr = cls(
            _a.dask,
            _a.name,
            _a.chunks,
            dtype=_a.dtype,
            meta=_a._meta,
            shape=_a.shape,
            _context=ctx,
        )
        return new_arr

    @_copy_doc
    def compute(self, **kwargs: Any) -> np.ndarray:
        """Compute this dask collection.

        This turns a lazy Dask collection into its in-memory equivalent.
        For example a Dask array turns into a NumPy array and a Dask dataframe
        turns into a Pandas dataframe.  The entire dataset must fit into memory
        before calling this operation.

        Parameters
        ----------
        scheduler : string, optional
            Which scheduler to use like "threads", "synchronous" or "processes".
            If not provided, the default is to check the global settings first,
            and then fall back to the collection defaults.
        optimize_graph : bool, optional
            If True [default], the graph is optimized before computation.
            Otherwise the graph is run as is. This can be useful for debugging.
        kwargs
            Extra keywords to forward to the scheduler function.

        See Also
        --------
        dask.compute

        Notes
        -----
        This subclass of DaskArray will re-open the underlying file before compute.
        """
        _ctx = self._context if self._context.closed else nullcontext()
        with _ctx:
            return super().compute(**kwargs)

    def __getitem__(self, index: Any) -> ResourceBackedDaskArray:
        # indexing should also return an Opening Array
        super_item = super().__getitem__(index)
        return ResourceBackedDaskArray.from_array(super_item, self._context)

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

    def __array_function__(
        self, func: Callable, types: Collection, args: tuple, kwargs: dict
    ) -> Any:
        # obey NEP18
        types = tuple(DaskArray if x is ResourceBackedDaskArray else x for x in types)
        arr = super().__array_function__(func, types, args, kwargs)
        if isinstance(arr, DaskArray):
            return ResourceBackedDaskArray.from_array(arr, self._context)
        return arr  # pragma: no cover

    def __reduce__(self) -> tuple[type, tuple, dict]:
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

    def __setstate__(self, d: dict) -> None:
        if not self._context.closed:
            self._context.__exit__(None, None, None)


class _ArrayMethodProxy:
    """Wraps method on a dask array and returns a OpeningDaskArray if the result of the
    method is a dask array.  see details in OpeningDaskArray docstring.
    """  # noqa: D205

    def __init__(self, method: MethodType, file_ctx: CheckableContext) -> None:
        self.method = method
        self._context = file_ctx

    def __repr__(self) -> str:
        return repr(self.method)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        with self._context if self._context.closed else nullcontext():
            result = self.method(*args, **kwds)
        if isinstance(result, DaskArray):
            return ResourceBackedDaskArray.from_array(result, self._context)
        return result
