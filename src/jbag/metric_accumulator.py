import numbers

import numpy as np
import torch


class MetricAccumulator:
    """
    Accumulate the values of one metric and compute their mean and standard deviation.

    Values can be real scalars (Python/numpy) or torch tensors, but a single accumulator must not mix the two
    families. All values must have the same shape (usually a scalar). For non-scalar values, ``mean`` and ``std``
    are computed over all elements of all recorded values. Torch tensors are detached before being stored, so
    accumulating metric values does not keep an autograd graph alive.

    The standard deviation is the sample standard deviation (``ddof=1``) for both numpy and torch. It is ``nan``
    when fewer than two values have been recorded.

    This class performs no distributed reduction; when training with multiple processes, reduce the values or
    accumulators yourself.
    """

    def __init__(self, metric_fn=None):
        """
        Args:
            metric_fn (callable or None, optional, default=None): function computing the metric. Its return value is
                accumulated and returned by ``__call__``. If `None`, use :meth:`add_value` to register values.
        """
        if metric_fn is not None and not callable(metric_fn):
            raise TypeError(f'metric_fn must be callable or None, but got {type(metric_fn)!r}.')

        self.metric_fn = metric_fn
        self._evaluations = []
        self._is_torch = None
        self._shape = None

    def _normalize(self, value):
        if isinstance(value, torch.Tensor):
            if self._is_torch is False:
                raise TypeError('Cannot mix torch.Tensor and non-tensor metric values in one MetricAccumulator.')
            self._is_torch = True
            value = value.detach()
            if self._evaluations:
                first = self._evaluations[0]
                if value.device != first.device or value.dtype != first.dtype:
                    raise ValueError('All torch metric values must have the same device and dtype.')
        elif isinstance(value, (numbers.Number, np.ndarray, np.generic)):
            if self._is_torch is True:
                raise TypeError('Cannot mix torch.Tensor and non-tensor metric values in one MetricAccumulator.')
            self._is_torch = False
        else:
            raise TypeError(
                f'Metric values must be numbers, numpy arrays/scalars or torch tensors, but got {type(value)!r}.')

        shape = tuple(value.shape) if hasattr(value, 'shape') else ()
        if self._shape is None:
            self._shape = shape
        elif shape != self._shape:
            raise ValueError(f'All metric values must have the same shape, but got {shape} and {self._shape}.')
        return value

    def __call__(self, *args, **kwargs):
        """Compute the metric from the given arguments, record it and return it."""
        if self.metric_fn is None:
            raise TypeError('metric_fn is not set; pass it to the constructor or use add_value().')
        evaluation = self.metric_fn(*args, **kwargs)
        self.add_value(evaluation)
        return evaluation

    def add_value(self, value):
        """Record an already computed metric value."""
        self._evaluations.append(self._normalize(value))

    def reset(self):
        """Remove all recorded values."""
        self._evaluations = []
        self._is_torch = None
        self._shape = None

    def mean(self):
        """Mean of all recorded values."""
        if not self._evaluations:
            raise ValueError('No evaluation exists.')
        if self._is_torch:
            return torch.stack(self._evaluations).mean()
        return np.mean(self._evaluations)

    def std(self):
        """Sample standard deviation (``ddof=1``) of all recorded values.

        Returns ``nan`` when fewer than two values have been recorded, because the sample standard deviation is
        undefined in that case.
        """
        if not self._evaluations:
            raise ValueError('No evaluation exists.')
        if len(self._evaluations) < 2:
            if self._is_torch:
                first = self._evaluations[0]
                dtype = first.dtype if first.is_floating_point() else torch.get_default_dtype()
                return torch.full((), float('nan'), dtype=dtype, device=first.device)
            return float('nan')
        if self._is_torch:
            # torch defaults to the sample std (unbiased=True), i.e. ddof=1.
            return torch.stack(self._evaluations).std()
        return np.std(self._evaluations, ddof=1)

    def count(self):
        """Number of recorded values."""
        return len(self._evaluations)

    def __len__(self):
        return len(self._evaluations)

    @property
    def evaluations(self):
        """A read-only snapshot of all recorded values."""
        return tuple(self._evaluations)
