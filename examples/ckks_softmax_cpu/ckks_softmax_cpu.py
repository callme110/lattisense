# Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = None
for _candidate in (_THIS_FILE.parents[2], _THIS_FILE.parents[3], _THIS_FILE.parents[4]):
    if (_candidate / 'frontend' / 'custom_task.py').exists():
        _PROJECT_ROOT = _candidate
        break
if _PROJECT_ROOT is None:
    raise RuntimeError('Unable to locate project root containing frontend/custom_task.py')
sys.path.append(str(_PROJECT_ROOT))

from frontend.custom_task import *


NUM_CLASSES = 8
INPUT_LEVEL = 13
ROTATE_STEPS = [1, 2, 4]


def repeated_block_sum(x: CkksCiphertextNode, rotate_steps: list[int], prefix: str) -> CkksCiphertextNode:
    # The input is packed as a repeated class block, so 1/2/4 rotations
    # produce the block-wise sum in every active slot.
    total = x
    for step in rotate_steps:
        total = add(total, rotate_cols(total, step, output_id=f'{prefix}_rot_{step}')[0], output_id=f'{prefix}_sum_{step}')
    return total


def eval_exp_poly_v1(
    x: CkksCiphertextNode,
    c5: CkksPlaintextMulNode,
    c4: CkksPlaintextNode,
    c3: CkksPlaintextNode,
    c2: CkksPlaintextNode,
    c1: CkksPlaintextNode,
    c0: CkksPlaintextNode,
) -> CkksCiphertextNode:
    x_l10 = drop_level(x, 1, 'centered_quarter_l10')
    x_l9 = drop_level(x, 2, 'centered_quarter_l9')
    x_l8 = drop_level(x, 3, 'centered_quarter_l8')
    x_l7 = drop_level(x, 4, 'centered_quarter_l7')

    acc = rescale(mult(x, c5, output_id='exp_leading_mul'), 'exp_acc_1')
    acc = add(acc, c4, 'exp_acc_2')
    acc = rescale(mult_relin(acc, x_l10, output_id='exp_mul_3'), 'exp_acc_3')
    acc = add(acc, c3, 'exp_acc_4')
    acc = rescale(mult_relin(acc, x_l9, output_id='exp_mul_2'), 'exp_acc_5')
    acc = add(acc, c2, 'exp_acc_6')
    acc = rescale(mult_relin(acc, x_l8, output_id='exp_mul_1'), 'exp_acc_7')
    acc = add(acc, c1, 'exp_acc_8')
    acc = rescale(mult_relin(acc, x_l7, output_id='exp_mul_0'), 'exp_acc_9')
    acc = add(acc, c0, 'exp_poly')
    exp_half = rescale(mult_relin(acc, acc, output_id='exp_square_1'), 'exp_half_logits')
    return rescale(mult_relin(exp_half, exp_half, output_id='exp_square_2'), 'exp_logits')


def eval_recip_poly_v1(
    x: CkksCiphertextNode,
    c3: CkksPlaintextMulNode,
    c2: CkksPlaintextNode,
    c1: CkksPlaintextNode,
    c0: CkksPlaintextNode,
) -> CkksCiphertextNode:
    x_l3 = drop_level(x, 1, 'denom_l3')
    x_l2 = drop_level(x, 2, 'denom_l2')

    acc = rescale(mult(x, c3, output_id='recip_leading_mul'), 'recip_acc_1')
    acc = add(acc, c2, 'recip_acc_2')
    acc = rescale(mult_relin(acc, x_l3, output_id='recip_mul_1'), 'recip_acc_3')
    acc = add(acc, c1, 'recip_acc_4')
    acc = rescale(mult_relin(acc, x_l2, output_id='recip_mul_0'), 'recip_acc_5')
    return add(acc, c0, 'inv_denom')


def ckks_softmax_cpu():
    param = Param.create_ckks_default_param(n=32768)
    set_fhe_param(param)

    logits = CkksCiphertextNode('logits', level=INPUT_LEVEL)

    pt_quarter = CkksPlaintextRingtNode('pt_quarter')
    pt_inv_classes = CkksPlaintextRingtNode('pt_inv_classes')

    exp_c5 = CkksPlaintextMulNode('exp_c5', level=11)
    exp_c4 = CkksPlaintextNode('exp_c4', level=10)
    exp_c3 = CkksPlaintextNode('exp_c3', level=9)
    exp_c2 = CkksPlaintextNode('exp_c2', level=8)
    exp_c1 = CkksPlaintextNode('exp_c1', level=7)
    exp_c0 = CkksPlaintextNode('exp_c0', level=6)

    recip_c3 = CkksPlaintextMulNode('recip_c3', level=4)
    recip_c2 = CkksPlaintextNode('recip_c2', level=3)
    recip_c1 = CkksPlaintextNode('recip_c1', level=2)
    recip_c0 = CkksPlaintextNode('recip_c0', level=1)

    logits_quarter = rescale(mult(logits, pt_quarter, output_id='logits_quarter_mul'), 'logits_quarter')
    quarter_sum = repeated_block_sum(logits_quarter, ROTATE_STEPS, 'quarter_sum')
    # softmax(x) = softmax(x - c * 1). Mean-centering preserves exact softmax,
    # and quarter scaling keeps p5 input in the fitted interval.
    mean_quarter = rescale(mult(quarter_sum, pt_inv_classes, output_id='mean_quarter_mul'), 'mean_quarter')
    logits_quarter_l11 = drop_level(logits_quarter, 1, 'logits_quarter_l11')
    centered_quarter = sub(logits_quarter_l11, mean_quarter, 'centered_quarter')

    exp_logits = eval_exp_poly_v1(centered_quarter, exp_c5, exp_c4, exp_c3, exp_c2, exp_c1, exp_c0)
    denom = repeated_block_sum(exp_logits, ROTATE_STEPS, 'denom')
    inv_denom = eval_recip_poly_v1(denom, recip_c3, recip_c2, recip_c1, recip_c0)

    exp_logits_l1 = drop_level(exp_logits, 3, 'exp_logits_l1')
    softmax = rescale(mult_relin(exp_logits_l1, inv_denom, output_id='softmax_mul'), 'softmax')

    process_custom_task(
        input_args=[Argument('logits', logits)],
        output_args=[Argument('softmax', softmax)],
        offline_input_args=[
            Argument('pt_quarter', pt_quarter),
            Argument('pt_inv_classes', pt_inv_classes),
            Argument('exp_c5', exp_c5),
            Argument('exp_c4', exp_c4),
            Argument('exp_c3', exp_c3),
            Argument('exp_c2', exp_c2),
            Argument('exp_c1', exp_c1),
            Argument('exp_c0', exp_c0),
            Argument('recip_c3', recip_c3),
            Argument('recip_c2', recip_c2),
            Argument('recip_c1', recip_c1),
            Argument('recip_c0', recip_c0),
        ],
        output_instruction_path='project',
        fpga_acc=False,
    )


if __name__ == '__main__':
    ckks_softmax_cpu()
