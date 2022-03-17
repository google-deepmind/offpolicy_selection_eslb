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

"""Implements a Dataset class which is an interface for OpenML dataset."""
from typing import NamedTuple

import numpy as np
import sklearn.datasets as skl_data
import sklearn.model_selection as skl_ms
import sklearn.preprocessing as skl_prep

from offpolicy_selection_eslb import policies


class FullInfoLoggedData(NamedTuple):
  """A dataset logged by a bandit policy and the true labels for testing.

  Attributes:
    contexts: n-times-d Array -- feature vectors for each entry
    actions: n-times-1 Array -- action taken by logging policy
    rewards: n-times-1 Array -- reward received
    labels: n-times-1 Array -- True label
  """

  contexts: np.ndarray
  actions: np.ndarray
  rewards: np.ndarray
  labels: np.ndarray


def generate_binary_noise(
    n: int,
    p: float,
) -> np.ndarray:
  """Returns a Bernoulli-distributed noise vector.

  Args:
    n: Number of points to generate.
    p: Bernoulli parameter (same for each point).
  Returns: Binary vector of length n.
  """
  return np.random.binomial(n=1, p=p, size=n)


def get_reward(
    actions: np.ndarray,
    labels: np.ndarray,
    reward_noise_p: float = 0.1,
    low_rew: float = 0.0,
    high_rew: float = 1.,
):
  """Returns rewards and corrupted labels for matching actions.

  Args:
    actions: A n-vector of actions (integer in {0,nb_class -1}).
    labels: A n-vector of labels.
    reward_noise_p: A noise-level parameter in (0,1).
    low_rew: Reward for incorrect action.
    high_rew: Reward for correct action.
  Returns: A n-vector of rewards after adding noise and rescaling.
  """
  rewards = np.equal(actions, labels)
  rewards = (rewards + generate_binary_noise(rewards.size, reward_noise_p)) % 2
  rewards = high_rew * rewards + low_rew * (1 - rewards)
  return rewards


class Dataset:
  """Represents an OpenML dataset.

  Attributes:
    openml_id: OpenML id of the dataset (for loading).
    log_frac: In (0,1), fraction of the data to be used as train data (logged
      dataset).
    reward_noise: In (0,1) noise level in the rewards obtained by a policy.
    name: Name of a dataset according to OpenML.
    encoder: Instance of scikit-learn LabelEncoder() preprocessing labels.
    contexts_all: Full dataset contexts (unless subsample >1, then subsampled
      contexts).
    labels_all: Full dataset labels (unless subsample >1, then subsampled
      labels).
    contexts_train: Train data contexts.
    contexts_test: Test data context.
    labels_train: Train data labels.
    labels_test: Test data labels.
    n_train: Train data size.
    n_test: Test data size.
    size: Total size of the dataset.
  """

  def __init__(
      self,
      openml_id: int,
      standardize: bool = True,
      log_frac: float = 0.50,
      subsample: int = 1,
      random_state: int = 0,
      reward_noise: float = 0.1,
  ):
    """Constructs Dataset object.

    Args:
        openml_id: OpenML id of the dataset (for loading).
        standardize: Binary, use True to standardize dataset.
        log_frac: In (0,1), fraction of the data to be used as train data
          (logged dataset).
        subsample: Subsample rate -- use only every "subsample" point from the
          dataset.
        random_state: Seed for train-test split (sklearn).
        reward_noise: In (0,1) noise level in the rewards obtained by a policy.
    """
    self.openml_id = openml_id
    self.log_frac = log_frac
    self.reward_noise = reward_noise

    dataset = skl_data.fetch_openml(
        data_id=openml_id, cache=True, as_frame=False)
    data = dataset.data
    target = dataset.target
    self.name = dataset.details["name"]

    self.encoder = skl_prep.LabelEncoder()
    self.encoder.fit(target)
    target = self.encoder.transform(target)

    self.contexts_all = data[::subsample]
    self.labels_all = target[::subsample]

    if standardize:
      scaler = skl_prep.StandardScaler()
      scaler.fit(self.contexts_all)
      self.contexts_all = scaler.transform(self.contexts_all)

    (self.contexts_train, self.contexts_test, self.labels_train,
     self.labels_test) = skl_ms.train_test_split(
         self.contexts_all,
         self.labels_all,
         test_size=1 - self.log_frac,
         shuffle=True,
         random_state=random_state)

    self.n_train = len(self.labels_train)
    self.n_test = len(self.labels_test)
    self.size = len(self.labels_all)

  def get_test(self, policy: policies.Policy) -> FullInfoLoggedData:
    """Returns test data contexts, action, rewards, test labels.

    Args:
      policy: An object of class Policy
    Returns: A tuple FullInfoLoggedData (contexts, actions, rewards, labels).
    """
    actions, _ = policy.query(self.contexts_test)
    rewards = get_reward(actions, self.labels_test, self.reward_noise)

    return FullInfoLoggedData(self.contexts_test, actions, rewards,
                              self.labels_test)

  def get_action_set(self):
    """Returns dictionary mapping labels to corresponding actions."""

    return self.encoder.transform(self.encoder.classes_)
