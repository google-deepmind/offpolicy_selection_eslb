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

"""Implements several value estimator for contextual bandit off-policy problem.

All estimators are described in
"Kuzborskij, I., Vernade, C., Gyorgy, A., & Szepesvári, C. (2021, March).
Confident off-policy evaluation and selection through self-normalized importance
weighting. In International Conference on Artificial Intelligence and Statistics
(pp. 640-648). PMLR.".
In the following we occasionally refer to the statements in the paper
(e.g. Theorem 1, Proposition 1).

class ESLB implements an Efron-Stein high probability bound for off-policy
evaluation (Theorem 1 and Algorithm 1).

class IWEstimator implements the standard importance weighted estimator (IW).

class SNIWEstimator implements a self-normalized version of IW.

class IWLambdaEmpBernsteinEstimator implements a high probability empirical
Bernstein bound for λ-corrected IW (the estimator is stabilized by adding λ
to the denominator) with appropriate tuning of λ (see Proposition 1).

"""

import abc
import math
from typing import List
from absl import logging

import numpy as np

from offpolicy_selection_eslb import data
from offpolicy_selection_eslb import policies
from offpolicy_selection_eslb import utils


class Estimator(abc.ABC):
  """Abstract class for a value estimator."""

  @abc.abstractmethod
  def __call__(
      self,
      t_probs: np.ndarray,
      b_probs: np.ndarray,
      actions: np.ndarray,
      rewards: np.ndarray,
  ):
    """Computes an estimate.

    Here n is a sample size, while K is a number actions.

    Args:
      t_probs: n-times-K matrix, where i-th row corresponds to π_t(. | X_i)
        (target probabilities under the target policy).
      b_probs: n-times-K matrix, where i-th row corresponds to π_b(. | X_i)
        (target probabilities under the behavior policy).
      actions: n-sized vector of actions.
      rewards: n-sized reward vector.

    Returns:
      Must return a Dict with "estimate" entry present.
    """

  @abc.abstractmethod
  def get_name(self) -> str:
    """Returns a the name of the estimator.
    """

  @abc.abstractmethod
  def get_abbrev(self) -> str:
    """Returns a shorter version of the name returned by get_name.
    """


class ESLB(Estimator):
  """Implements a Semi-Empirical Efron-Stein bound for the SNIW (Self-normalized Importance Weighted estimator).

  Attributes:
    delta: Error probability in (0,1).
    n_iterations: Number of Monte-Carlo simulation iterations for approximating
      a multiplicative bias and a variance proxy.
    n_batch_size: Monte-Carlo simulation batch size.
  """

  def __init__(
      self,
      delta: float,
      n_iterations: int,
      n_batch_size: int,
  ):
    """Constructs an estimator.

    The estimate holds with probability 1-delta.

    Args:
      delta: delta: Error probability in (0,1) for a confidence interval.
      n_iterations: Monte-Carlo simulation iterations.
      n_batch_size: Monte-Carlo simulation batch size.
    """
    self.delta = delta
    self.n_iterations = n_iterations
    self.n_batch_size = n_batch_size

  def get_name(self):
    """Returns the long name of the estimator."""
    return "Semi-Empirical Efron-Stein bound for the Self-normalized Estimator"

  def get_abbrev(self):
    """Returns the short name of the estimator."""
    return "ESLB"

  def __call__(
      self,
      t_probs: np.ndarray,
      b_probs: np.ndarray,
      actions: np.ndarray,
      rewards: np.ndarray,
  ):
    """Computes Efron-Stein lower bound of Theorem 1 as described in Algorithm 1.

    Here n is a sample size, while K is a number actions.

    Args:
      t_probs: n-times-K matrix, where i-th row corresponds to π_t(. | X_i)
        (target probabilities under the target policy).
      b_probs: n-times-K matrix, where i-th row corresponds to π_b(. | X_i)
        (target probabilities under the behavior policy).
      actions: n-sized vector of actions.
      rewards: n-sized reward vector.

    Returns:
      A dictionary with 8 entries:
        lower_bound: Corresponds to the actual lower bound.
        estimate: Same as lower_bound (required by select_policy(...)).
        est_value: Empirical value.
        mult_bias: Multiplicative bias.
        concentration_of_contexts: Hoeffding term, concentration of contexts.
        var_proxy: Variance proxy.
        expected_variance_proxy: Estimated expected counterpart.
    """
    conf = math.log(2.0 / self.delta)
    n = len(actions)
    ix_1_n = np.arange(n)

    # Importance weights
    weights = t_probs[ix_1_n, actions] / b_probs[ix_1_n, actions]

    weights_cumsum = weights.cumsum()
    weights_cumsum = np.repeat(
        np.expand_dims(weights_cumsum, axis=1), self.n_batch_size, axis=1)
    weights_repeated = np.repeat(
        np.expand_dims(weights, axis=1), self.n_batch_size, axis=1)

    weight_table = t_probs / b_probs

    var_proxy_unsumed = np.zeros((n,))
    expected_var_proxy_unsumed = np.zeros((n,))
    expected_recip_weights = 0.0

    logging.debug(
        "ESLB:: Running Monte-Carlo estimation of the variance proxy and bias")
    logging.debug("ESLB:: iterations = %d, batch size = %d", self.n_iterations,
                  self.n_batch_size)

    for i in range(self.n_iterations):
      actions_sampled = utils.sample_from_simplices_m_times(
          b_probs, self.n_batch_size)
      weights_sampled = weight_table[ix_1_n, actions_sampled.T].T
      weights_sampled_cumsum = weights_sampled[::-1, :].cumsum(axis=0)[::-1, :]
      # Hybrid sums: sums of empirical and sampled weights
      weights_hybrid_sums = np.copy(weights_cumsum)
      weights_hybrid_sums[:-1, :] += weights_sampled_cumsum[1:, :]

      actions_sampled_for_u = utils.sample_from_simplices_m_times(
          b_probs, self.n_batch_size)
      weights_sampled_for_u = weight_table[ix_1_n, actions_sampled_for_u.T].T
      actions_sampled_for_bias = utils.sample_from_simplices_m_times(
          b_probs, self.n_batch_size)
      weights_sampled_for_bias = weight_table[ix_1_n,
                                              actions_sampled_for_bias.T].T

      weights_hybrid_sums_replace_k = weights_hybrid_sums - weights_repeated + weights_sampled
      weights_tilde = weights_repeated / weights_hybrid_sums
      u_tilde = weights_sampled_for_u / weights_hybrid_sums_replace_k
      var_proxy_t = (weights_tilde + u_tilde)**2

      expected_var_proxy_new_item = ((weights_sampled /
                                      weights_sampled.sum(axis=0))**2).mean(
                                          axis=1)
      var_proxy_new_item = var_proxy_t.mean(axis=1)

      expected_var_proxy_unsumed += (expected_var_proxy_new_item -
                                     expected_var_proxy_unsumed) / (
                                         i + 1)
      var_proxy_unsumed += (var_proxy_new_item - var_proxy_unsumed) / (i + 1)

      bias_t = (1.0 / weights_sampled_for_bias.sum(axis=0)).mean()
      expected_recip_weights += (bias_t - expected_recip_weights) / (i + 1)

    var_proxy = var_proxy_unsumed.sum()
    expected_var_proxy = expected_var_proxy_unsumed.sum()

    eff_sample_size = 1.0 / expected_recip_weights

    mult_bias = min(1.0, eff_sample_size / n)
    concentration = math.sqrt(
        2.0 * (var_proxy + expected_var_proxy) *
        (conf + 0.5 * math.log(1 + var_proxy / expected_var_proxy)))
    concentration_of_contexts = math.sqrt(conf / (2 * n))
    est_value = weights.dot(rewards) / weights.sum()
    lower_bound = mult_bias * (est_value -
                               concentration) - concentration_of_contexts

    return dict(
        estimate=max(0, lower_bound),
        lower_bound=max(0, lower_bound),
        est_value=est_value,
        concentration=concentration,
        mult_bias=mult_bias,
        concentration_of_contexts=concentration_of_contexts,
        var_proxy=var_proxy,
        expected_var_proxy=expected_var_proxy)


class IWEstimator(Estimator):
  """Implements an importance-weighted estimator of the value."""

  def __call__(
      self,
      t_probs: np.ndarray,
      b_probs: np.ndarray,
      actions: np.ndarray,
      rewards: np.ndarray,
  ):
    """Computes an importance-weighted (IW) estimate.

    Here n is a sample size, while K is a number actions.

    Args:
      t_probs: n-times-K matrix, where i-th row corresponds to π_t(. | X_i)
        (target probabilities under the target policy).
      b_probs: n-times-K matrix, where i-th row corresponds to π_b(. | X_i)
        (target probabilities under the behavior policy).
      actions: n-sized vector of actions.
      rewards: n-sized reward vector.

    Returns:
      A dictionary with 1 entry:
        estimate: Importance-weighted estimate (required by select_policy(...)).
    """
    n = len(actions)
    ix_1_n = np.arange(n)

    # Importance weights
    weights = t_probs[ix_1_n, actions] / b_probs[ix_1_n, actions]

    estimate = rewards.dot(weights) / n

    return dict(estimate=estimate)

  def get_name(self):
    """Returns a long name of an estimator."""
    return "Importance-weighted estimator"

  def get_abbrev(self):
    """Returns a short name of an estimator."""
    return "IW"


class SNIWEstimator(Estimator):
  """Implements a self-normalized importance-weighted estimator of the value."""

  def get_name(self):
    """Returns a long name of an estimator."""
    return "Self-normalized importance-weighted estimator"

  def get_abbrev(self):
    """Returns a short name of an estimator."""
    return "SNIW"

  def __call__(
      self,
      t_probs: np.ndarray,
      b_probs: np.ndarray,
      actions: np.ndarray,
      rewards: np.ndarray,
  ):
    """Computes a self-normalized importance-weighted (SNIW) estimate.

    Here n is a sample size, while K is a number actions.

    Args:
      t_probs: n-times-K matrix, where i-th row corresponds to π_t(. | X_i)
        (target probabilities under the target policy).
      b_probs: n-times-K matrix, where i-th row corresponds to π_b(. | X_i)
        (target probabilities under the behavior policy).
      actions: n-sized vector of actions.
      rewards: n-sized reward vector.

    Returns:
      A dictionary with 1 entry:
        estimate: SNIW estimate (required by select_policy(...)).
    """
    n = len(actions)
    ix_1_n = np.arange(n)

    # Importance weights
    weights = t_probs[ix_1_n, actions] / b_probs[ix_1_n, actions]

    estimate = rewards.dot(weights) / weights.sum()

    return dict(estimate=estimate)


class IWLambdaEmpBernsteinEstimator(Estimator):
  """Implements an empirical Bernstein confidence bound for λ-corrected IW.

  λ-corrected importance-weighted (IW) estimator is defined w.r.t. weights of a
  form π_target(A|X) / (π_behavior(A|X) + λ), where λ=1/sqrt(n),
  and the choice of λ ensures asymptotic convergence of the confidence bound
  (see accompaying paper for details).

  Attributes:
    delta: Error probability in (0,1).
  """

  def __init__(self, delta: float):
    """Constructs an estimator.

    The estimate holds with probability 1-delta.
    Args:
      delta: Error probability in (0,1).
    """
    self.delta = delta

  def get_name(self):
    """Returns a long name of an estimator."""
    return ("Empirical Bernstein bound for λ-corrected importance-weighted "
            "estimator (λ=1/sqrt(n))")

  def get_abbrev(self):
    """Returns a short name of an estimator."""
    return "Emp. Bernstein for λ-IW"

  def __call__(
      self,
      t_probs: np.ndarray,
      b_probs: np.ndarray,
      actions: np.ndarray,
      rewards: np.ndarray,
  ):
    """Computes Empirical Bernstein λ-IW estimate.

    Computes an estimate according to the Empirical Bernstein bound for
    λ-corrected importance-weighted (see Proposition 1).

    Here n is a sample size, while K is a number actions.

    Args:
      t_probs: n-times-K matrix, where i-th row corresponds to π_t(. | X_i)
        (target probabilities under the target policy).
      b_probs: n-times-K matrix, where i-th row corresponds to π_b(. | X_i)
        (target probabilities under the behavior policy).
      actions: n-sized vector of actions.
      rewards: n-sized reward vector.

    Returns:
      A dictionary with 8 entries:
        lower_bound: Corresponds to the actual lower bound.
        estimate: Same as lower_bound (required by select_policy(...)).
        est_value: Empirical value.
        concentration: Concentration term.
        bias: Bias term.
        concentration_of_contexts: Hoeffding term, concentration of contexts.
        est_var: sample variance of the estimator.
    """
    n = len(actions)
    ix_1_n = np.arange(n)

    conf = math.log(3.0 / self.delta)
    lambda_corr = 1.0 / math.sqrt(n)

    # Importance weights with lambda correction
    weights = t_probs[ix_1_n, actions] / (
        b_probs[ix_1_n, actions] + lambda_corr)

    v_estimates = weights * rewards

    est_value = np.mean(v_estimates)
    est_var = np.var(v_estimates)

    bias = 0.0

    # Computing the bias term
    for i_feature in range(n):
      for k_action in range(b_probs.shape[1]):
        t_prob_context_k = t_probs[i_feature, k_action]
        b_prob_context_k = b_probs[i_feature, k_action]
        bias += t_prob_context_k * abs(b_prob_context_k /
                                       (b_prob_context_k + lambda_corr) - 1.0)
    bias /= n

    concentration = math.sqrt(
        (2 * conf / n) * est_var) + (7 * conf) / (3 * lambda_corr * (n - 1))
    concentration_of_contexts = math.sqrt(2 * conf / n)

    lower_bound = est_value - concentration - bias - concentration_of_contexts

    return dict(
        estimate=max(0, lower_bound),
        lower_bound=max(0, lower_bound),
        est_value=est_value,
        concentration=concentration,
        bias=bias,
        concentration_of_contexts=concentration_of_contexts,
        est_var=est_var)


class SNIWChebyshevEstimator(Estimator):
  """Implements Chebyshev bound for SNIW estimator.

  Attributes:
    delta: Error probability in (0,1).
  """

  def __init__(self, delta: float):
    """Constructs an estimator.

    The estimate holds with probability 1-delta.
    Args:
      delta: Error probability in (0,1).
    """
    self.delta = delta

  def __call__(
      self,
      t_probs: np.ndarray,
      b_probs: np.ndarray,
      actions: np.ndarray,
      rewards: np.ndarray,
  ):
    """Computes Chebyshev bound for SNIW estimate.

    Computes an estimate according to the Chebyshev bound for
    the self-normalized importance-weighted estimator (see Proposition 1).

    Here n is a sample size, while K is a number actions.

    Args:
      t_probs: n-times-K matrix, where i-th row corresponds to π_t(. | X_i)
        (target probabilities under the target policy).
      b_probs: n-times-K matrix, where i-th row corresponds to π_b(. | X_i)
        (target probabilities under the behavior policy).
      actions: n-sized vector of actions.
      rewards: n-sized reward vector.

    Returns:
      A dictionary with 8 entries:
        lower_bound: Corresponds to the actual lower bound;
        estimate: Same as lower_bound (required by select_policy(...))
        est_value: Empirical value;
        concentration: Concentration term;
        mult_bias: Multiplicative bias term;
        concentration_of_contexts: Hoeffding term, concentration of contexts;
        est_var: Sample variance of the estimator;
    """
    n = len(actions)
    ix_1_n = np.arange(n)

    conf = 3.0 / self.delta
    ln_conf = math.log(3.0 / self.delta)

    # Importance weights
    weights = t_probs[ix_1_n, actions] / b_probs[ix_1_n, actions]

    t_probs_all_actions = t_probs[ix_1_n, :]
    b_probs_all_actions = b_probs[ix_1_n, :]

    weights = weights.squeeze()
    est_value = rewards.dot(weights) / weights.sum()

    expected_sum_weights_sq = (t_probs_all_actions**2 /
                               b_probs_all_actions).sum()
    eff_sample_size = max(
        n - math.sqrt(2.0 * ln_conf * expected_sum_weights_sq), 0)

    if eff_sample_size > 0:
      est_var = expected_sum_weights_sq / eff_sample_size**2
      concentration = math.sqrt(conf * est_var)
      concentration_of_contexts = math.sqrt((2 * ln_conf) / n)
      mult_bias = eff_sample_size / n

      lower_bound = mult_bias * (est_value -
                                 concentration) - concentration_of_contexts
    else:
      est_var = np.nan
      est_var = np.nan
      concentration = np.nan
      concentration_of_contexts = np.nan
      mult_bias = np.nan
      lower_bound = 0.0

    return dict(
        estimate=max(0, lower_bound),
        lower_bound=max(0, lower_bound),
        est_value=est_value,
        concentration=concentration,
        mult_bias=mult_bias,
        concentration_of_contexts=concentration_of_contexts,
        est_var=est_var)

  def get_name(self):
    """Returns a long name of an estimator."""
    return "Chebyshev bound for self-normalized importance-weighted estimator"

  def get_abbrev(self):
    """Returns a short name of an estimator."""
    return "Cheb-SNIW"


def select_policy(
    contexts: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    b_policy: policies.Policy,
    t_policies: List[policies.Policy],
    estimator: Estimator,
):
  """Selects a policy given an estimator.

  Args:
    contexts: A n x d matrix of n context vectors.
    actions: A n-vector of actions.
    rewards: A n-vector of rewards.
    b_policy: Behavior policy implementing get_probs(...) method (see
      SoftmaxDataPolicy in policies.py).
    t_policies: A list of objects of implementing get_probs(...) method
      (see SoftmaxGAPolicy).
    estimator: An object of a base class Estimator.

  Returns:
    A tuple (estimate, policy) with the highest estimate.
  """

  estimates_and_policies = []
  b_probs = b_policy.get_probs(contexts)

  for pol in t_policies:
    t_probs = pol.get_probs(contexts)

    result_dict = estimator(
        t_probs=t_probs, b_probs=b_probs, actions=actions, rewards=rewards)
    estimates_and_policies.append((result_dict["estimate"], pol))

  ordered_estimates_and_policies = sorted(
      estimates_and_policies, key=lambda x: x[0])
  return ordered_estimates_and_policies[-1]


def evaluate_estimators(
    contexts: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    b_policy: policies.Policy,
    t_policies: List[policies.Policy],
    estimators: List[Estimator],
    dataset: data.Dataset,
):
  """Evaluates multiple estimators based on their ability to select a policy.

  Args:
    contexts: A n x d matrix of n context vectors.
    actions: A n-vector of actions.
    rewards: A n-vector of rewards.
    b_policy: Behavior policy implementing get_probs(...) method (see
      SoftmaxDataPolicy).
    t_policies: A list of n_pol objects of implementing get_probs(...) method
      (see SoftmaxGAPolicy).
    estimators: A list of n_est objects of a base class Estimator.
    dataset: Object of the class Dataset.

  Returns:
    A tuple with three elements: (test_rewards, winners, reference_test_rewards)
    where test_rewards is a (n_est x n_test) matrix of test rewards such
    that n_test is a test sample size; winners a list of size n_est of
    high-scoring policies according to each estimator; reference_test_rewards
    is a n_test-vector of highest-scoring policy on a test set in hindsight.
  """

  winners = []  # winner policies of each estimator
  test_rewards = np.zeros((len(estimators), dataset.n_test))

  for (est_i, est) in enumerate(estimators):
    est_winner, pol_winner = select_policy(contexts, actions, rewards, b_policy,
                                           t_policies, est)
    winners.append(pol_winner)

    if est_winner > 0:
      _, _, pol_winner_test_rewards, _ = dataset.get_test(pol_winner)
      test_rewards[est_i, :] = pol_winner_test_rewards
    else:
      test_rewards[est_i, :] = np.nan
      logging.debug("evaluate_estimators:: est '%s' didn't score anything.",
                    est.get_abbrev())

    # Getting test reward of the best policy (as a reference)
    reference_test_rewards = []
    for pol in t_policies:
      _, _, reference_test_rewards_for_pol, _ = dataset.get_test(pol)
      reference_test_rewards.append(reference_test_rewards_for_pol)
    reference_test_rewards = sorted(reference_test_rewards, key=np.mean)[-1]

  return test_rewards, winners, reference_test_rewards


def get_estimators(
    delta,
    eslb_iter: int,
    eslb_batch_size: int,
):
  """Constructs estimators to be used in the benchmark.

  Args:
    delta: Error probability in (0,1).
    eslb_iter: Monte-Carlo simulation iterations for ESLB estimator.
    eslb_batch_size: Monte-Carlo simulation batch size for ESLB estimator.

  Returns:
    A list of dictionaries containing at least one entry "estimate" (key).

  """
  estimators = [
      IWEstimator(),
      SNIWEstimator(),
      SNIWChebyshevEstimator(delta=delta),
      IWLambdaEmpBernsteinEstimator(delta=delta),
      ESLB(delta=delta, n_iterations=eslb_iter, n_batch_size=eslb_batch_size),
  ]
  return estimators
