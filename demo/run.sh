#!/bin/sh
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


set -e

python3 -m venv /tmp/eslb_venv
source /tmp/eslb_venv/bin/activate
pip3 install --upgrade pip setuptools wheel
pip3 install -r ../requirements.txt

# Runs an off-policy benchmark on UCI datasets with OpenML IDs 181, 30, 28, 182, 300, 32, 6, 184,
# for 10 trials and confidence bound failure probability δ=0.01
# ERROR python3 -m offpolicy_selection_eslb.demo.benchmark --dataset_type=uci_all --n_trials=10 --delta=0.01
python3 benchmark.py --dataset_type=uci_all --n_trials=10 --delta=0.01
