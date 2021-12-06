# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Multiple sampling from simplex tool."""
import numpy as np


def sample_from_simplices_m_times(p: np.ndarray, m: int) -> np.ndarray:
  """Samples from each of n probability simplices for m times.

  Args:
    p: n-times-K matrix where each row describes a probability simplex
    m: number of times to sample

  Returns:
    n-times-m matrix of indices of simplex corners.
  """
  axis = 1
  r = np.expand_dims(np.random.rand(p.shape[1 - axis], m), axis=axis)
  p_ = np.expand_dims(p.cumsum(axis=axis), axis=2)
  return (np.repeat(p_, m, axis=2) > r).argmax(axis=1)
