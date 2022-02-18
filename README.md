# resource-backed-dask-array

[![License](https://img.shields.io/pypi/l/resource-backed-dask-array.svg?color=green)](https://github.com/tlambert03/resource-backed-dask-array/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/resource-backed-dask-array.svg?color=green)](https://pypi.org/project/resource-backed-dask-array)
[![Python Version](https://img.shields.io/pypi/pyversions/resource-backed-dask-array.svg?color=green)](https://python.org)
[![CI](https://github.com/tlambert03/resource-backed-dask-array/actions/workflows/ci.yml/badge.svg)](https://github.com/tlambert03/resource-backed-dask-array/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/tlambert03/resource-backed-dask-array/branch/main/graph/badge.svg)](https://codecov.io/gh/tlambert03/resource-backed-dask-array)

`ResourceBackedDaskArray` is an experimental Dask array subclass
that opens/closes a resource when computing (but only once per compute call).

## installation

```sh
pip install resource-backed-dask-array
```

## motivation for this package

Consider the following class that simulates a file reader capable of returning a
dask array (using
[dask.array.map_blocks](https://docs.dask.org/en/latest/generated/dask.array.map_blocks.html))
The file handle must be in an *open* state in order to read a chunk, otherwise
it segfaults (or otherwise errors)

```python
import dask.array as da
import numpy as np


class FileReader:

    def __init__(self):
        self._closed = False

    def close(self):
        """close the imaginary file"""
        self._closed = True

    @property
    def closed(self):
        return self._closed

    def __enter__(self):
        if self.closed:
            self._closed = False  # open
        return self

    def __exit__(self, *_):
        self.close()

    def to_dask(self) -> da.Array:
        """Method that returns a dask array for this file."""
        return da.map_blocks(
            self._dask_block,
            chunks=((1,) * 4, 4, 4),
            dtype=float,
        )

    def _dask_block(self):
        """simulate getting a single chunk of the file."""
        if self.closed:
            raise RuntimeError("Segfault!")
        return np.random.rand(1, 4, 4)
```

As long as the file stays open, everything works fine:

```python
>>> fr = FileReader()
>>> dsk_ary = fr.to_dask()
>>> dsk_ary.compute().shape
(4, 4, 4)
```

However, if one closes the file, the dask array returned
from `to_dask` will now fail:

```python
>>> fr.close()
>>> dsk_ary.compute()  # RuntimeError: Segfault!
```

A "quick-and-dirty" solution here might be to force the `_dask_block` method to
temporarily reopen the file if it finds the file in the closed state, but if the
file-open process takes any amount of time, this could incur significant
overhead as it opens-and-closes for *every* chunk in the array.

## usage

`ResourceBackedDaskArray.from_array`

This library attempts to provide a solution to the above problem with a
`ResourceBackedDaskArray` object.  This manages the opening/closing of
an underlying resource whenever [`.compute()`](https://docs.dask.org/en/stable/generated/dask.array.Array.compute.html#dask.array.Array.compute) is called – and does so only once for all chunks in a single compute task graph.

```python
>>> from resource_backed_dask_array import resource_backed_dask_array
>>> safe_dsk_ary = resource_backed_dask_array(dsk_ary, fr)
>>> safe_dsk_ary.compute().shape
(4, 4, 4)

>>> fr.closed  # leave it as we found it
True
```

The second argument passed to `from_array` must be a [resuable context manager](https://docs.python.org/3/library/contextlib.html#reusable-context-managers)
that additionally provides a `closed` attribute (like [io.IOBase](https://docs.python.org/3/library/io.html#io.IOBase.closed)).  In other words,
it must implement the following protocol:

1. it must have an [`__enter__` method](https://docs.python.org/3/reference/datamodel.html#object.__enter__) that opens the underlying resource
2. it must have an [`__exit__` method](https://docs.python.org/3/reference/datamodel.html#object.__exit__) that closes the resource and optionally handles exceptions
3. it must have a `closed` attribute that reports whether or not the resource is closed.

In the example above, the `FileReader` class itself implemented this protocol, and so was suitable as the second argument to `ResourceBackedDaskArray.from_array` above.

## Important Caveats

This was created for single-process (and maybe just single-threaded?)
use cases where dask's out-of-core lazy loading is still very desireable.  Usage
with `dask.distributed` is untested and may very well fail.  Using stateful objects (such as the reusable context manager used here) in multi-threaded/processed tasks is error prone.
