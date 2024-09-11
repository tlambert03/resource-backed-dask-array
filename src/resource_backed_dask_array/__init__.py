"""experimental Dask array that opens/closes a resource when computing."""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

try:
    __version__ = version("resource-backed-dask-array")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__author__ = "Talley Lambert"
__all__ = ["resource_backed_dask_array", "ResourceBackedDaskArray"]

from ._array import ResourceBackedDaskArray, resource_backed_dask_array
