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

"""Implements a benchmark executable.

The executable runs experimental suite by calling
run_experiment_suite(...) defined in experiment module.
The type of the experiment and its parameters are set by the flags
(see help for each flag).
The suite is described in
"Kuzborskij, I., Vernade, C., Gyorgy, A., & Szepesvári, C. (2021, March).
Confident off-policy evaluation and selection through self-normalized importance
weighting. In International Conference on Artificial Intelligence and Statistics
(pp. 640-648). PMLR."..

Examples:

# Run evaluation on all datasets considered in the paper for
# 10 trials (data splits) with error probability = 0.01
python3 benchmark.py --dataset_type=uci_all --n_trials=10 --delta=0.01
"""

from absl import app
from absl import flags
from absl import logging
import termcolor

from offpolicy_selection_eslb import estimators
from offpolicy_selection_eslb import experiment
from offpolicy_selection_eslb import policies


_DATASET_TYPE = flags.DEFINE_enum(
    "dataset_type",
    "demo",
    ["uci_small", "uci_medium", "uci_all", "demo"],
    "UCI dataset subset.",
)

_N_TRIALS = flags.DEFINE_integer(
    "n_trials",
    10,
    "Number of experimental trials (for sample statistics).",
    lower_bound=0)

_FAULTY_ACTIONS = flags.DEFINE_list("faulty_actions", ["1", "2"],
                                    "Faulty action indices.")

_DELTA = flags.DEFINE_float(
    "delta",
    0.05,
    "Error probability delta (i.e. confidence intervals hold w.p. at least 1-delta).",
    lower_bound=0,
    upper_bound=1)

_BEHAVIOR_POL_TEMPERATURE = flags.DEFINE_float(
    "behavior_policy_temperature",
    0.2,
    "Temperature of a softmax behavior policy (small = peaked actions).",
    lower_bound=0)

_TARGET_POL_TEMPERATURE = flags.DEFINE_float(
    "target_policy_temperature",
    0.1,
    "Temperature of a softmax target policy (small = peaked actions).",
    lower_bound=0)

_REWARD_NOISE_P = flags.DEFINE_float(
    "reward_noise_p",
    0.1,
    "Reward noise probability.",
    lower_bound=0,
    upper_bound=1)

_GA_STEP_SIZE = flags.DEFINE_float(
    "GA_step_size",
    0.01,
    "Gradient Ascent step size for training softmax target policies.",
    lower_bound=0)

_GA_ITER = flags.DEFINE_integer(
    "GA_n_iter",
    10000,
    "Gradient Ascent steps for training softmax target policies.",
    lower_bound=0)

_ESLB_ITER = flags.DEFINE_integer(
    "eslb_n_iter",
    10,
    "Number of Monte-Carlo iterations for ESLB.",
    lower_bound=0)

_ESLB_BATCH = flags.DEFINE_integer(
    "eslb_batch", 1000, "Monte-Carlo batch size for ESLB.", lower_bound=0)

_TABLE_FORMAT = flags.DEFINE_string(
    "table_format", "psql",
    "Result table format (e.g. psql, latex, html). See https://pypi.org/project/tabulate/"
)

green = lambda x: termcolor.colored(x, color="green")


def main(argv):
  del argv  # Unused.

  logging.set_verbosity(logging.INFO)

  # Datasets by OpenML IDs (see https://www.openml.org/search?type=data)
  if _DATASET_TYPE.value == "uci_small":
    dataset_ids = [39, 41, 54, 181, 30, 28, 182, 32]
  elif _DATASET_TYPE.value == "uci_medium":
    dataset_ids = [181, 30, 28, 182]
  elif _DATASET_TYPE.value == "uci_all":
    dataset_ids = [181, 30, 28, 182, 300, 32, 6, 184]
  elif _DATASET_TYPE.value == "demo":
    dataset_ids = [28, 30]

  logging.info(
      green("running on '%s' dataset suite (openml ids: %s); see --help"),
      _DATASET_TYPE.value, ", ".join(map(str, dataset_ids)))
  logging.info(green("number of trials = %d"), _N_TRIALS.value)
  logging.info(green("faulty action indices = %s"),
               ", ".join(map(str, _FAULTY_ACTIONS.value)))
  logging.info(green("confidence bound failure probability (δ) = %f"),
               _DELTA.value)
  logging.info(green("behavior policy temperature = %f"),
               _BEHAVIOR_POL_TEMPERATURE.value)
  logging.info(green("target policy temperature = %f"),
               _TARGET_POL_TEMPERATURE.value)
  logging.info(green("reward noise prob. = %f"),
               _REWARD_NOISE_P.value)
  logging.info(green("steps of gradient ascent for fitting policies = %d"),
               _GA_ITER.value)
  logging.info(green("gradient ascent step size = %f"),
               _GA_STEP_SIZE.value)
  logging.info(green("ESLB estimator Monte-Carlo estimation steps = %d"),
               _ESLB_ITER.value)
  logging.info(green("ESLB estimator Monte-Carlo batch size = %d"),
               _ESLB_BATCH.value)

  estimators_ = estimators.get_estimators(
      delta=_DELTA.value,
      eslb_iter=_ESLB_ITER.value,
      eslb_batch_size=_ESLB_BATCH.value)

  target_policy_specs = [
      ("SoftmaxGAPolicy",
       dict(
           step_size=_GA_STEP_SIZE.value,
           steps=_GA_ITER.value,
           temperature=_TARGET_POL_TEMPERATURE.value,
           obj_type=policies.TrainedPolicyObjType.IW)),
      ("SoftmaxGAPolicy",
       dict(
           step_size=_GA_STEP_SIZE.value,
           steps=_GA_ITER.value,
           temperature=_TARGET_POL_TEMPERATURE.value,
           obj_type=policies.TrainedPolicyObjType.SNIW)),
      ("SoftmaxDataPolicy",
       dict(temperature=_TARGET_POL_TEMPERATURE.value, faulty_actions=[]))
  ]

  behavior_faulty_actions = list(map(int, _FAULTY_ACTIONS.value))

  (mean_test_rewards, std_test_rewards, mean_reference_rewards,
   std_reference_rewards, dataset_names) = experiment.run_experiment_suite(
       list_data_ids=dataset_ids,
       n_trials=_N_TRIALS.value,
       behavior_policy_temperature=_BEHAVIOR_POL_TEMPERATURE.value,
       behavior_faulty_actions=behavior_faulty_actions,
       target_policy_specs=target_policy_specs,
       reward_noise_p=_REWARD_NOISE_P.value,
       estimators=estimators_)
  experiment.print_results(
      estimators_,
      dataset_names,
      mean_test_rewards,
      std_test_rewards,
      mean_reference_rewards,
      std_reference_rewards,
      table_format=_TABLE_FORMAT.value)


if __name__ == "__main__":
  app.run(main)
