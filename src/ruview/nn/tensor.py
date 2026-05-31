"""Lightweight tensor helpers for neural research code.

The Rust reference abstracts over several inference backends. This Python port
keeps the default path NumPy-only and exposes optional PyTorch conversion hooks
for callers that install the ``nn`` extra.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


class TensorError(ValueError):
    """Raised when tensor shape, dtype, or operation validation fails."""


class DataType(str, Enum):
    """Supported tensor element types."""

    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT32 = "int32"
    INT64 = "int64"
    UINT8 = "uint8"
    BOOL = "bool"

    @property
    def numpy_dtype(self) -> np.dtype[Any]:
        """Return the NumPy dtype for this data type."""

        return np.dtype(self.value)

    @property
    def size_bytes(self) -> int:
        """Return the element size in bytes."""

        return int(self.numpy_dtype.itemsize)

    @classmethod
    def from_value(cls, value: "DataType | str | np.dtype[Any] | type[Any]") -> "DataType":
        """Coerce a dtype-like value to a :class:`DataType`."""

        if isinstance(value, DataType):
            return value

        dtype = np.dtype(value)
        for candidate in cls:
            if dtype == candidate.numpy_dtype:
                return candidate
        raise TensorError(f"unsupported tensor dtype {dtype}")


@dataclass(frozen=True, init=False)
class TensorShape:
    """Concrete tensor shape with broadcasting helpers."""

    dims: tuple[int, ...]

    def __init__(self, dims: Iterable[int]) -> None:
        parsed = tuple(_coerce_dim(dim) for dim in dims)
        object.__setattr__(self, "dims", parsed)

    @classmethod
    def from_array(cls, value: ArrayLike) -> "TensorShape":
        """Build a shape from a NumPy-coercible value."""

        return cls(np.asarray(value).shape)

    @property
    def ndim(self) -> int:
        """Number of dimensions."""

        return len(self.dims)

    @property
    def numel(self) -> int:
        """Total number of elements implied by the shape."""

        return math.prod(self.dims)

    def dim(self, index: int) -> int | None:
        """Return a dimension by index, or ``None`` when out of bounds."""

        if -self.ndim <= index < self.ndim:
            return self.dims[index]
        return None

    def is_broadcast_compatible(self, other: "TensorShape | Sequence[int]") -> bool:
        """Return whether this shape can broadcast with ``other``."""

        try:
            broadcast_shape(self, other)
        except TensorError:
            return False
        return True

    def broadcast_with(self, other: "TensorShape | Sequence[int]") -> "TensorShape":
        """Return the broadcast result shape for this shape and ``other``."""

        return broadcast_shape(self, other)

    def validate_array(self, value: ArrayLike, *, name: str = "tensor") -> NDArray[Any]:
        """Validate that ``value`` has exactly this shape."""

        array = np.asarray(value)
        if tuple(array.shape) != self.dims:
            raise TensorError(f"{name} expected shape {self}, got {tuple(array.shape)}")
        return array

    def __iter__(self):
        return iter(self.dims)

    def __len__(self) -> int:
        return self.ndim

    def __getitem__(self, index: int) -> int:
        return self.dims[index]

    def __str__(self) -> str:
        return "[" + ", ".join(str(dim) for dim in self.dims) + "]"


@dataclass(frozen=True, init=False)
class TensorSpec:
    """Tensor validation spec with optional wildcard dimensions.

    ``None`` in ``shape`` means "any size for this dimension"; rank and dtype
    are still checked.
    """

    shape: tuple[int | None, ...]
    dtype: DataType
    name: str

    def __init__(
        self,
        shape: Iterable[int | None],
        *,
        dtype: DataType | str | np.dtype[Any] | type[Any] = DataType.FLOAT32,
        name: str = "tensor",
    ) -> None:
        object.__setattr__(self, "shape", tuple(_coerce_spec_dim(dim) for dim in shape))
        object.__setattr__(self, "dtype", DataType.from_value(dtype))
        object.__setattr__(self, "name", str(name))

    @property
    def ndim(self) -> int:
        """Expected rank."""

        return len(self.shape)

    def matches_shape(self, shape: TensorShape | Sequence[int]) -> bool:
        """Return whether a concrete shape satisfies this spec."""

        concrete = _as_shape(shape).dims
        if len(concrete) != len(self.shape):
            return False
        return all(expected is None or expected == actual for expected, actual in zip(self.shape, concrete, strict=True))

    def validate(self, value: ArrayLike, *, allow_cast: bool = False) -> NDArray[Any]:
        """Validate ``value`` against this spec and return a NumPy array."""

        array = np.asarray(value)
        actual = TensorShape(array.shape)
        if not self.matches_shape(actual):
            raise TensorError(
                f"{self.name} expected shape {self._shape_text()}, got {tuple(array.shape)}"
            )

        expected_dtype = self.dtype.numpy_dtype
        if array.dtype != expected_dtype:
            if not allow_cast:
                raise TensorError(
                    f"{self.name} expected dtype {expected_dtype}, got {array.dtype}"
                )
            array = array.astype(expected_dtype, copy=False)
        return array

    def with_batch(self, batch_size: int | None = None) -> "TensorSpec":
        """Return a spec with a leading batch dimension."""

        return TensorSpec((batch_size, *self.shape), dtype=self.dtype, name=self.name)

    def _shape_text(self) -> str:
        return "[" + ", ".join("*" if dim is None else str(dim) for dim in self.shape) + "]"


@dataclass(frozen=True)
class TensorStats:
    """Scalar summary statistics for a tensor."""

    mean: float
    std: float
    min: float
    max: float
    sparsity: float

    @classmethod
    def from_tensor(cls, tensor: "Tensor | ArrayLike") -> "TensorStats":
        """Compute statistics from a tensor-like value."""

        array = tensor.data if isinstance(tensor, Tensor) else np.asarray(tensor)
        if array.size == 0:
            return cls(mean=0.0, std=0.0, min=0.0, max=0.0, sparsity=0.0)
        return cls(
            mean=float(np.mean(array)),
            std=float(np.std(array)),
            min=float(np.min(array)),
            max=float(np.max(array)),
            sparsity=float(np.count_nonzero(array == 0) / array.size),
        )


@dataclass(frozen=True, init=False)
class Tensor:
    """Small NumPy-backed tensor wrapper used by training helpers."""

    data: NDArray[Any]

    def __init__(
        self,
        data: ArrayLike,
        *,
        dtype: DataType | str | np.dtype[Any] | type[Any] | None = None,
        copy: bool = False,
    ) -> None:
        np_dtype = None if dtype is None else DataType.from_value(dtype).numpy_dtype
        array = np.asarray(data, dtype=np_dtype)
        if copy:
            array = array.copy()
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        object.__setattr__(self, "data", array)

    @classmethod
    def zeros(
        cls,
        shape: TensorShape | Sequence[int],
        *,
        dtype: DataType | str | np.dtype[Any] | type[Any] = DataType.FLOAT32,
    ) -> "Tensor":
        """Create a zero-filled tensor."""

        return cls(np.zeros(_as_shape(shape).dims, dtype=DataType.from_value(dtype).numpy_dtype))

    @classmethod
    def ones(
        cls,
        shape: TensorShape | Sequence[int],
        *,
        dtype: DataType | str | np.dtype[Any] | type[Any] = DataType.FLOAT32,
    ) -> "Tensor":
        """Create a one-filled tensor."""

        return cls(np.ones(_as_shape(shape).dims, dtype=DataType.from_value(dtype).numpy_dtype))

    @classmethod
    def from_torch(cls, tensor: Any, *, dtype: DataType | str | np.dtype[Any] | type[Any] | None = None) -> "Tensor":
        """Create a NumPy-backed tensor from a PyTorch tensor."""

        if not hasattr(tensor, "detach"):
            raise TypeError("from_torch expects a torch.Tensor-like object")
        return cls(tensor.detach().cpu().numpy(), dtype=dtype)

    @property
    def shape(self) -> TensorShape:
        """Concrete shape."""

        return TensorShape(self.data.shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        """NumPy dtype."""

        return self.data.dtype

    @property
    def ndim(self) -> int:
        """Number of dimensions."""

        return self.data.ndim

    @property
    def numel(self) -> int:
        """Number of elements."""

        return int(self.data.size)

    def to_numpy(self, *, copy: bool = False) -> NDArray[Any]:
        """Return the underlying NumPy array."""

        return self.data.copy() if copy else self.data

    def to_torch(self, *, device: str | None = None, copy: bool = False) -> Any:
        """Return a PyTorch tensor, importing torch only when requested."""

        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch is optional; install ruview-python[nn] to use to_torch") from exc

        array = self.data.copy() if copy else self.data
        tensor = torch.from_numpy(array)
        return tensor if device is None else tensor.to(device)

    def astype(
        self,
        dtype: DataType | str | np.dtype[Any] | type[Any],
        *,
        copy: bool = False,
    ) -> "Tensor":
        """Return this tensor converted to ``dtype``."""

        return Tensor(self.data.astype(DataType.from_value(dtype).numpy_dtype, copy=copy))

    def reshape(self, shape: TensorShape | Sequence[int]) -> "Tensor":
        """Return a reshaped tensor."""

        return Tensor(np.reshape(self.data, _as_shape(shape).dims))

    def as_slice(self) -> NDArray[Any]:
        """Return a flat contiguous view of the tensor data."""

        return self.data.reshape(-1)

    def to_vec(self) -> list[Any]:
        """Return a Python list of flattened tensor values."""

        return self.as_slice().tolist()

    def relu(self) -> "Tensor":
        """Apply ReLU elementwise."""

        _require_numeric(self.data, "relu")
        return Tensor(np.maximum(self.data, 0))

    def sigmoid(self) -> "Tensor":
        """Apply sigmoid elementwise."""

        _require_numeric(self.data, "sigmoid")
        data = self.data.astype(np.float32, copy=False)
        return Tensor(1.0 / (1.0 + np.exp(-data)), dtype=DataType.FLOAT32)

    def tanh(self) -> "Tensor":
        """Apply tanh elementwise."""

        _require_numeric(self.data, "tanh")
        return Tensor(np.tanh(self.data.astype(np.float32, copy=False)), dtype=DataType.FLOAT32)

    def softmax(self, axis: int = -1) -> "Tensor":
        """Apply numerically stable softmax along ``axis``."""

        _require_numeric(self.data, "softmax")
        axis = _normalize_axis(axis, self.ndim)
        data = self.data.astype(np.float32, copy=False)
        shifted = data - np.max(data, axis=axis, keepdims=True)
        exp = np.exp(shifted)
        denom = np.sum(exp, axis=axis, keepdims=True)
        return Tensor(exp / denom, dtype=DataType.FLOAT32)

    def argmax(self, axis: int = -1) -> "Tensor":
        """Return integer indices of maxima along ``axis``."""

        axis = _normalize_axis(axis, self.ndim)
        return Tensor(np.argmax(self.data, axis=axis), dtype=DataType.INT64)

    def mean(self) -> float:
        """Mean value, or ``0.0`` for an empty tensor."""

        return float(np.mean(self.data)) if self.data.size else 0.0

    def std(self) -> float:
        """Population standard deviation, or ``0.0`` for an empty tensor."""

        return float(np.std(self.data)) if self.data.size else 0.0

    def min(self) -> float:
        """Minimum value, or ``0.0`` for an empty tensor."""

        return float(np.min(self.data)) if self.data.size else 0.0

    def max(self) -> float:
        """Maximum value, or ``0.0`` for an empty tensor."""

        return float(np.max(self.data)) if self.data.size else 0.0

    def stats(self) -> TensorStats:
        """Compute scalar tensor statistics."""

        return TensorStats.from_tensor(self)

    @classmethod
    def stack(cls, tensors: Sequence["Tensor | ArrayLike"], *, axis: int = 0) -> "Tensor":
        """Stack tensors along a new axis."""

        if not tensors:
            raise TensorError("cannot stack zero tensors")
        arrays = [tensor.data if isinstance(tensor, Tensor) else np.asarray(tensor) for tensor in tensors]
        first_shape = arrays[0].shape
        for index, array in enumerate(arrays[1:], start=1):
            if array.shape != first_shape:
                raise TensorError(
                    f"shape mismatch at index {index}: expected {first_shape}, got {array.shape}"
                )
        return cls(np.stack(arrays, axis=axis))

    def split(self, pieces: int) -> list["Tensor"]:
        """Split this tensor into equal chunks along the leading axis."""

        if pieces <= 0:
            raise TensorError("cannot split into zero pieces")
        if self.ndim == 0:
            raise TensorError("scalar tensor cannot be split")
        batch = self.data.shape[0]
        if batch % pieces != 0:
            raise TensorError(f"leading dimension {batch} is not divisible by {pieces}")
        return [Tensor(array) for array in np.split(self.data, pieces, axis=0)]


def broadcast_shape(*shapes: TensorShape | Sequence[int]) -> TensorShape:
    """Return the NumPy broadcast shape for concrete shape values."""

    if not shapes:
        return TensorShape(())

    parsed = [_as_shape(shape).dims for shape in shapes]
    max_ndim = max(len(shape) for shape in parsed)
    result: list[int] = []

    for offset in range(1, max_ndim + 1):
        dims = [shape[-offset] if offset <= len(shape) else 1 for shape in parsed]
        non_singletons = {dim for dim in dims if dim != 1}
        if len(non_singletons) > 1:
            raise TensorError(f"shapes are not broadcast-compatible: {parsed}")
        result.append(max(non_singletons) if non_singletons else 1)

    result.reverse()
    return TensorShape(result)


def are_broadcast_compatible(*shapes: TensorShape | Sequence[int]) -> bool:
    """Return whether all provided shapes are broadcast-compatible."""

    try:
        broadcast_shape(*shapes)
    except TensorError:
        return False
    return True


def as_tensor(
    value: ArrayLike,
    *,
    dtype: DataType | str | np.dtype[Any] | type[Any] | None = None,
    copy: bool = False,
) -> Tensor:
    """Coerce a value to :class:`Tensor`."""

    return Tensor(value, dtype=dtype, copy=copy)


def _as_shape(shape: TensorShape | Sequence[int]) -> TensorShape:
    return shape if isinstance(shape, TensorShape) else TensorShape(shape)


def _coerce_dim(value: int) -> int:
    if isinstance(value, bool):
        raise TensorError("shape dimensions must be integers, not bool")
    dim = int(value)
    if dim != value:
        raise TensorError(f"shape dimension {value!r} is not an integer")
    if dim < 0:
        raise TensorError(f"shape dimension {dim} must be non-negative")
    return dim


def _coerce_spec_dim(value: int | None) -> int | None:
    return None if value is None else _coerce_dim(value)


def _normalize_axis(axis: int, ndim: int) -> int:
    if ndim == 0:
        raise TensorError("axis is invalid for a scalar tensor")
    if not -ndim <= axis < ndim:
        raise TensorError(f"axis {axis} is out of bounds for tensor of dimension {ndim}")
    return axis % ndim


def _require_numeric(array: NDArray[Any], operation: str) -> None:
    if not np.issubdtype(array.dtype, np.number):
        raise TensorError(f"{operation} requires a numeric tensor")


__all__ = [
    "DataType",
    "Tensor",
    "TensorError",
    "TensorShape",
    "TensorSpec",
    "TensorStats",
    "are_broadcast_compatible",
    "as_tensor",
    "broadcast_shape",
]
