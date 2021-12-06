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

"""Implements experimental routines.

Experimental protocol corresponds to the one discussed in
"Kuzborskij, I., Vernade, C., Gyorgy, A., & Szepesvári, C. (2021, March).
Confident off-policy evaluation and selection through self-normalized importance
weighting. In International Conference on Artificial Intelligence and Statistics
(pp. 640-648). PMLR.".

function run_single_experiment(...) defines an experiment performed on a
single dataset given multiple estimators over multiple data splits.

function run_experiment_suite(...) generalizes the above to multiple datasets.

function print_results prints the results returned by run_experiment_suite.
"""
from typing import Dict, List, Tuple, Any
from absl import logging

import numpy as np
import tabulate

import offpolicy_selection_eslb.data as data
import offpolicy_selection_eslb.estimators as est
import offpolicy_selection_eslb.policies as policies


def run_single_experiment(
    estimators: List[est.Estimator],
    openml_id: int,
    n_trials: int,
    behavior_policy_temperature: float,
    behavior_faulty_actions: List[int],
    target_policy_specs: List[Tuple[str, Dict[str, Any]]],
    reward_noise_p: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
  """Returns scores of an experiment on a single dataset for all estimators.

  Evaluates all estimators on a single dataset for a given number of trials,
  given description of the behavior policy and specifications of target
  policies.

  Args:
    estimators: A list of objects of a base class Estimator (imported as est).
    openml_id: OpenML dataset id (integer).
    n_trials: Number of experimental trials (data splits).
    behavior_policy_temperature: Positive float controlling the temperature of a
      Softmax behavior policy.
    behavior_faulty_actions: List of labels on which the behavior policy makes
      mistakes.
    target_policy_specs: Tuple of target policy specifications consisting of
      two-element
    tuples("<policy class name>", <dict of arguments to be passed to the
      constructor>) e.g. ( ("SoftmaxGAPolicy", dict( step_size=0.1, steps=1000,
      temperature=0.1, obj_type=policies.TrainedPolicyObjType.IW)), ).
    reward_noise_p: Probability of a Bernoulli noise added to the reward.

  Returns:
    Tuple (all_test_rewards, all_reference_rewards, dataset_name).

    Here all_test_rewards is np.array
    of dimension (#estimators, #datasets, n_trials) where each entry is a
    reward of a given estimator on a dataset at a particular trial (data split).

    all_reference_rewards is is np.array
    of dimension (#datasets, n_trials) where each entry is a
    reward of a best estimator in a hindsight on a dataset at a particular
    trial (data split).

    dataset_name is a human-readable OpenML dataset name.
  """

  np.random.seed(1)

  dataset = data.Dataset(
      openml_id,
      standardize=True,
      log_frac=0.50,
      subsample=1,
      reward_noise=reward_noise_p,
      random_state=0)

  all_test_rewards = np.zeros((len(estimators), dataset.n_test, n_trials))
  all_reference_rewards = np.zeros((dataset.n_test, n_trials))

  contexts, labels = dataset.contexts_train, dataset.labels_train
  test_contexts, test_labels = dataset.contexts_test, dataset.labels_test

  action_set = dataset.get_action_set()

  behavior_policy = policies.SoftmaxDataPolicy(
      train_contexts=contexts,
      train_labels=labels,
      test_contexts=test_contexts,
      test_labels=test_labels,
      action_set=action_set,
      temperature=behavior_policy_temperature,
      faulty_actions=behavior_faulty_actions)

  for i_trial in range(n_trials):
    logging.info(
        "\u001b[32mrun_single_experiment:: trial = %d/%d ... \u001b[0m",
        i_trial + 1, n_trials)
    np.random.seed(i_trial)

    actions, _ = behavior_policy.query(contexts)
    behavior_probs = behavior_policy.get_probs_by_actions(contexts, actions)
    rewards = data.get_reward(actions, labels, reward_noise_p)

    target_policies = []
    for (t_pol_name, t_pol_params) in target_policy_specs:

      if t_pol_name == "SoftmaxDataPolicy":
        t_pol = policies.SoftmaxDataPolicy(
            train_contexts=contexts,
            train_labels=labels,
            test_contexts=test_contexts,
            test_labels=test_labels,
            action_set=action_set,
            **t_pol_params)

      elif t_pol_name == "SoftmaxGAPolicy":
        t_pol = policies.SoftmaxGAPolicy(action_set=action_set, **t_pol_params)

        logging.debug("run_single_experiment:: training %s", str(t_pol))
        t_pol.train(contexts, actions, rewards, behavior_probs)

      target_policies.append(t_pol)

    test_rewards, _, reference_test_rewards = est.evaluate_estimators(
        contexts, actions, rewards, behavior_policy, target_policies,
        estimators, dataset)

    all_test_rewards[:, :, i_trial] = test_rewards
    all_reference_rewards[:, i_trial] = reference_test_rewards

  return all_test_rewards, all_reference_rewards, dataset.name


def run_experiment_suite(
    list_data_ids: List[int],
    n_trials: int,
    behavior_policy_temperature: float,
    behavior_faulty_actions: List[int],
    target_policy_specs: List[Tuple[str, Dict[str, Any]]],
    reward_noise_p: float,
    estimators: List[est.Estimator],
):
  """Returns results of an experimental suite.

  Evaluates all estimators on all datasets for a given number of trials,
  given description of the behavior policy and specifications of target
  policies.

  Args:
    list_data_ids: List of OpenML dataset IDs.
    n_trials: Number of experimental trials (data splits).
    behavior_policy_temperature: Positive float controlling the temperature of a
      behavior Softmax policy.
    behavior_faulty_actions: List of labels on which the behavior policy makes
      mistakes.
    target_policy_specs: Tuple of target policy specifications consisting of
      two-element tuples ("<policy class name>", <dict of arguments to
      be passed to the constructor>) e.g.
      ("SoftmaxGAPolicy",
       dict(step_size=0.1, steps=1000, temperature=0.1,
            obj_type=policies.TrainedPolicyObjType.IW))
    reward_noise_p: Probability of a Bernoulli noise added to the reward.
    estimators: List of bjects of a base class Estimator.
  Returns: A tuple (mean_test_rewards, std_test_rewards, mean_reference_rewards,
    std_reference_rewards, dataset_names).  Here mean_test_rewards and
    std_test_rewards are np.array's of dimension (#estimators, #datasets).
    mean_test_rewards stand for the average over data splits.
    mean_reference_rewards is np.array of dimension #datasets, and stands for
    the average reward of the best policy in a hindsight, over data splits.
    std_* stands for the standard deviation.  dataset_names stands for
    human-readable dataset names.
  """

  mean_test_rewards = np.zeros((len(estimators), len(list_data_ids)))
  std_test_rewards = np.zeros((len(estimators), len(list_data_ids)))

  mean_reference_rewards = np.zeros(len(list_data_ids))
  std_reference_rewards = np.zeros(len(list_data_ids))

  dataset_names = []

  for data_id_i, data_id in enumerate(list_data_ids):
    logging.info("\u001b[32mrun_experiment_suite:: dataset = %d\u001b[0m",
                 data_id)

    (test_rewards_for_dataset, reference_rewards_for_dataset,
     dataset_name) = run_single_experiment(
         estimators=estimators,
         openml_id=data_id,
         n_trials=n_trials,
         behavior_policy_temperature=behavior_policy_temperature,
         behavior_faulty_actions=behavior_faulty_actions,
         target_policy_specs=target_policy_specs,
         reward_noise_p=reward_noise_p)

    mean_test_rewards[:, data_id_i] = np.array([
        test_rewards_for_dataset[i, :, :].mean()
        for i in range(test_rewards_for_dataset.shape[0])
    ])
    std_test_rewards[:, data_id_i] = np.array([
        test_rewards_for_dataset[i, :, :].std(axis=0).mean()
        for i in range(test_rewards_for_dataset.shape[0])
    ])

    mean_reference_rewards[data_id_i] = np.nanmean(
        reference_rewards_for_dataset)
    std_reference_rewards[data_id_i] = np.nanstd(reference_rewards_for_dataset)
    dataset_names.append(dataset_name)

  return (mean_test_rewards, std_test_rewards, mean_reference_rewards,
          std_reference_rewards, dataset_names)


def print_results(
    estimators: List[est.Estimator],
    dataset_names: List[str],
    mean_test_rewards: np.ndarray,
    std_test_rewards: np.ndarray,
    mean_reference_rewards: np.ndarray,
    std_reference_rewards: np.ndarray,
    table_format: str = "psql",
):
  """Prints results of run_experiment_suite(...) in a pretty table.

  Printing routines are implemented by the tabulate package.

  Args:
    estimators: A list of n_est objects of a base class Estimator.
    dataset_names: A list of strings, names of the corresponding ids.
    mean_test_rewards: Returned by run_experiment_suite.
    std_test_rewards: Returned by run_experiment_suite.
    mean_reference_rewards: Returned by run_experiment_suite.
    std_reference_rewards: Returned by run_experiment_suite.
    table_format: Parameter tablefmt of tabulate(...).
  """

  headers = [r"Estimator  \  Dataset"] + dataset_names
  rows = []
  for (i, est_) in enumerate(estimators):
    rows.append([est_.get_abbrev()] + list(
        map(lambda x: "%.3f ± %.3f" % x,
            zip(mean_test_rewards[i, :], std_test_rewards[i, :]))))

  rows.append(["Best policy on the test set"] + list(
      map(lambda x: "%.3f ± %.3f" % x,
          zip(mean_reference_rewards, std_reference_rewards))))

  print(tabulate.tabulate(rows, headers=headers, tablefmt=table_format))
