/*
 * Copyright (c) 2025-2026 CipherFlow (Shenzhen) Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cxx_sdk_v2/cxx_fhe_task.h>
#include <fhe_ops_lib/fhe_lib_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace fhe_ops_lib;
using namespace lattisense;
using namespace std;

namespace {

constexpr int kNumClasses = 8;
constexpr int kInputLevel = 13;
constexpr int kPolyInputLevel = 11;
constexpr int kSlotCount = 8192;

vector<double> tile_pattern(const vector<double>& pattern, int slot_count) {
    vector<double> tiled(slot_count, 0.0);
    for (int i = 0; i < slot_count; ++i) {
        tiled[i] = pattern[i % pattern.size()];
    }
    return tiled;
}

vector<double> tile_scalar(double value, int slot_count) { return vector<double>(slot_count, value); }

vector<double> softmax_reference(const vector<double>& logits) {
    vector<double> probs(logits.size());
    double max_logit = *max_element(logits.begin(), logits.end());
    double denom = 0.0;
    for (double logit : logits) {
        denom += exp(logit - max_logit);
    }
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = exp(logits[i] - max_logit) / denom;
    }
    return probs;
}

}  // namespace

void ckks_softmax_cpu() {
    const vector<double> logits{1.2, 0.5, -0.3, -0.8, 0.1, -0.4, 0.9, -1.0};
    const vector<double> ref_softmax = softmax_reference(logits);

    CkksParameter param = CkksParameter::create_parameter(32768);
    CkksContext ctx = CkksContext::create_random_context(param);
    const double default_scale = param.get_default_scale();
    auto q = [&](int level) { return param.get_q(level); };

    ctx.gen_rotation_keys();

    const vector<double> exp_coeffs{
        1.0031377334916605,
        1.0026864218461349,
        0.4860498435309526,
        0.1624711376226941,
        0.05072464694309538,
        0.010053701974162384,
    };
    const vector<double> recip_coeffs{
        0.24885999074111392,
        -0.021622407621476325,
        0.0007824595670968044,
        -0.000010035965681343485,
    };

    const double scale_exp_1 = default_scale;
    const double scale_exp_2 = scale_exp_1 * default_scale / q(10);
    const double scale_exp_3 = scale_exp_2 * default_scale / q(9);
    const double scale_exp_4 = scale_exp_3 * default_scale / q(8);
    const double scale_exp_5 = scale_exp_4 * default_scale / q(7);
    const double scale_exp_half = scale_exp_5 * scale_exp_5 / q(6);
    const double scale_exp = scale_exp_half * scale_exp_half / q(5);

    const double scale_recip_1 = default_scale;
    const double scale_recip_2 = scale_recip_1 * scale_exp / q(3);
    const double scale_recip_3 = scale_recip_2 * scale_exp / q(2);
    const double scale_recip_c3_mul = q(4) * scale_recip_1 / scale_exp;
    const double scale_softmax = scale_exp * scale_recip_3 / q(1);

    // Each class block is repeated across the full SIMD packing so that the
    // 1/2/4 rotation tree yields the denominator in every class slot.
    const vector<double> logits_tiled = tile_pattern(logits, kSlotCount);
    const auto logits_pt = ctx.encode(logits_tiled, kInputLevel, default_scale);
    auto logits_ct = ctx.encrypt_asymmetric(logits_pt);

    // The ring-t plaintext scales are chosen to keep the post-rescale
    // ciphertext scale aligned with the add/sub branches that follow.
    auto pt_quarter = ctx.encode_ringt(tile_scalar(0.25, kSlotCount), q(kInputLevel));
    auto pt_inv_classes = ctx.encode_ringt(tile_scalar(1.0 / kNumClasses, kSlotCount), q(12));

    auto exp_c5 = ctx.encode_mul(tile_scalar(exp_coeffs[5], kSlotCount), kPolyInputLevel, q(kPolyInputLevel));
    auto exp_c4 = ctx.encode(tile_scalar(exp_coeffs[4], kSlotCount), 10, scale_exp_1);
    auto exp_c3 = ctx.encode(tile_scalar(exp_coeffs[3], kSlotCount), 9, scale_exp_2);
    auto exp_c2 = ctx.encode(tile_scalar(exp_coeffs[2], kSlotCount), 8, scale_exp_3);
    auto exp_c1 = ctx.encode(tile_scalar(exp_coeffs[1], kSlotCount), 7, scale_exp_4);
    auto exp_c0 = ctx.encode(tile_scalar(exp_coeffs[0], kSlotCount), 6, scale_exp_5);

    auto recip_c3 = ctx.encode_mul(tile_scalar(recip_coeffs[3], kSlotCount), 4, scale_recip_c3_mul);
    auto recip_c2 = ctx.encode(tile_scalar(recip_coeffs[2], kSlotCount), 3, scale_recip_1);
    auto recip_c1 = ctx.encode(tile_scalar(recip_coeffs[1], kSlotCount), 2, scale_recip_2);
    auto recip_c0 = ctx.encode(tile_scalar(recip_coeffs[0], kSlotCount), 1, scale_recip_3);

    auto softmax_ct = ctx.new_ciphertext(0, scale_softmax);

    FheTaskCpu cpu_project("project");
    vector<CxxVectorArgument> cxx_args = {
        {"logits", &logits_ct},
        {"pt_quarter", &pt_quarter},
        {"pt_inv_classes", &pt_inv_classes},
        {"exp_c5", &exp_c5},
        {"exp_c4", &exp_c4},
        {"exp_c3", &exp_c3},
        {"exp_c2", &exp_c2},
        {"exp_c1", &exp_c1},
        {"exp_c0", &exp_c0},
        {"recip_c3", &recip_c3},
        {"recip_c2", &recip_c2},
        {"recip_c1", &recip_c1},
        {"recip_c0", &recip_c0},
        {"softmax", &softmax_ct},
    };
    cpu_project.run(&ctx, cxx_args);

    CkksPlaintext softmax_pt = ctx.decrypt(softmax_ct);
    vector<double> softmax_all = ctx.decode(softmax_pt);
    vector<double> softmax_head(softmax_all.begin(), softmax_all.begin() + kNumClasses);

    printf("CKKS softmax approximation, computed by CPU\n");
    print_double_message(logits.data(), "logits", kNumClasses);
    print_double_message(ref_softmax.data(), "softmax_ref", kNumClasses);
    print_double_message(softmax_head.data(), "softmax_ckks", kNumClasses);
}

int main() {
    ckks_softmax_cpu();
    return 0;
}
