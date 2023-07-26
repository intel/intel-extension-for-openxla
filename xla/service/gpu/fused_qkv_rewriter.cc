/* Copyright (c) 2023 Intel Corporation

Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/gpu/fused_qkv_rewriter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/protobuf/dnn.pb.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_reachability.h"
#include "xla/literal_comparison.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/instruction_fusion.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/sycl/hw_info.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

bool IsProfitableOperand(HloInstruction* instr) {
  // kConstant instruction will not have memory reads, so it won't be a profit
  // source. Skip them.
  if (instr->opcode() == HloOpcode::kConstant &&
      ShapeUtil::IsEffectiveScalar(instr->shape())) {
    return false;
  }
  return true;
}

FusionDecision LegalToFuse(HloInstruction* instr1, HloInstruction* instr2,
                           const GpuDeviceInfo& device_info,
                           FusionInfoCache* fusion_info_cache) {
  // Do this check last, as it may be expensive.
  return FusionFitsInBudget(*instr1, *instr2, device_info,
                            /*is_consumer_producer_fusion=*/false,
                            fusion_info_cache);
}

FusionDecision IsEqualShapeAndAttrs(HloInstruction* instr1,
                                    HloInstruction* instr2) {
  if (instr1->opcode() != instr2->opcode()) {
    return "has different opcode";
  }

  if (instr1->shape() != instr2->shape()) {
    return "has different shape";
  }

  if (instr1->shape().element_type() != instr2->shape().element_type()) {
    return "has different datatype";
  }
  return {};
}
}  // namespace

void FusedQKVRewriter::RecomputeReachability(HloComputation* computation) {
  reachability_ = HloReachabilityMap::Build(computation);
}

bool FusedQKVRewriter::FuseQKVSiblings(HloComputation* computation,
                                       HloInstruction* parent,
                                       FusionInfoCache* fusion_info_cache) {
  // TODO: to limit producer, which should not
  // be heavy hlo inst(gemm, conv, reduce, etc)
  if (!IsProfitableOperand(parent)) {
    VLOG(1) << "Operand " << parent->ToShortString() << " is not profitable";
    return false;
  }

  bool changed = false;
  std::vector<HloInstruction*> siblings = parent->users();
  std::vector<HloInstruction*> fusible_siblings;
  for (auto i = siblings.begin(); i != siblings.end(); ++i) {
    if (!IsCublasGemm((**i))) {
      continue;
    }
    VLOG(1) << "Parent: " << parent->name()
            << "  Considering its siblings: " << (*i)->name()
            << " Opcode: " << HloOpcodeString((**i).opcode());
    fusible_siblings.clear();
    fusible_siblings.push_back(*i);
    for (auto j = i + 1; j != siblings.end();) {
      auto is_disconnected = [&](const HloInstruction* a,
                                 const HloInstruction* b) -> FusionDecision {
        if (reachability_->IsConnected(a, b)) {
          return FusionDecision{} << a->name() << " and " << b->name()
                                  << " are connected";
        }
        return {};
      };

      VLOG(1) << "Considering siblingA: " << (*i)->name()
              << "  and siblingB: " << (*j)->name();
      if (NoFusionPossible sibling_fusible =
              (!is_disconnected(*i, *j) ||
               !LegalToFuse(*i, *j, device_info_, fusion_info_cache) ||
               !IsEqualShapeAndAttrs((*i), (*j)))) {
        // We pick `j` arbitrarily as a consumer.
        ++j;
        VLOG(1) << "Skip this sibling: " << (*j)->name()
                << "due to it does not has EqualShapeAndAttrs\n";
        continue;
      }

      if (!IsCublasGemm((**j))) {
        ++j;
        continue;
      }

      if (!ConsumeFuel(name(), [&] {
            return absl::StrFormat("Not fusing siblings %s and %s.",
                                   (*i)->name(), (*j)->name());
          })) {
        ++j;
        continue;
      }

      if ((*i)->operand(1)->shape() != (*j)->operand(1)->shape()) {
        ++j;
        VLOG(1) << "two gemm weights have different shape";
        continue;
      }
      VLOG(1) << "Fuse siblings " << (*i)->name() << " and " << (*j)->name();
      fusion_info_cache->Invalidate(*i);
      fusion_info_cache->Invalidate(*j);

      fusible_siblings.push_back(*j);
      if (fusible_siblings.size() == 3) break;
      ++j;
    }
    if (fusible_siblings.size() == 3) break;
  }
  if (!fusible_siblings.empty()) {
    VLOG(1) << " fusbile candidates ware found !!!\n";
    // get weight of each gemm
    std::vector<HloInstruction*> wei_operands;
    for (auto gemm : fusible_siblings) {
      auto gemm_weight = gemm->mutable_operand(1);
      PrimitiveType element_type = gemm_weight->shape().element_type();
      Shape wei_reshaped = ShapeUtil::MakeShape(
          element_type, {1, gemm_weight->shape().dimensions(0),
                         gemm_weight->shape().dimensions(1)});
      wei_operands.push_back(computation->AddInstruction(
          HloInstruction::CreateReshape(wei_reshaped, gemm_weight)));
    }
    PrimitiveType element_type = wei_operands[0]->shape().element_type();

    Shape wei_operand_shape = wei_operands[0]->shape();
    Shape input_operand_shape =
        fusible_siblings[0]->mutable_operand(0)->shape();

    Shape wei_concated_shape = ShapeUtil::MakeShape(
        element_type, {3, /*k = */ wei_operand_shape.dimensions(1),
                       /*n = */ wei_operand_shape.dimensions(2)});

    Shape output_shape = ShapeUtil::MakeShape(
        element_type, {/*m = */ input_operand_shape.dimensions(0),
                       /*n = */ wei_operand_shape.dimensions(2)});

    HloInstruction* wei_concated_inst = computation->AddInstruction(
        HloInstruction::CreateConcatenate(wei_concated_shape, wei_operands, 0));
    // shape of 3 outputs
    Shape custom_call_shape =
        ShapeUtil::MakeTupleShape({output_shape, output_shape, output_shape});
    HloInstruction* fqkv_call =
        computation->AddInstruction(HloInstruction::CreateCustomCall(
            custom_call_shape,
            {fusible_siblings[0]->mutable_operand(0), wei_concated_inst},
            kCudnnfQKVCallTarget));

    auto config = fusible_siblings[0]->backend_config<GemmBackendConfig>();
    fqkv_call->set_backend_config(config.value());

    HloInstruction* gte_0 =
        computation->AddInstruction(HloInstruction::CreateGetTupleElement(
            fqkv_call->shape().tuple_shapes(0), fqkv_call, 0));
    HloInstruction* gte_1 =
        computation->AddInstruction(HloInstruction::CreateGetTupleElement(
            fqkv_call->shape().tuple_shapes(1), fqkv_call, 1));
    HloInstruction* gte_2 =
        computation->AddInstruction(HloInstruction::CreateGetTupleElement(
            fqkv_call->shape().tuple_shapes(2), fqkv_call, 2));

    fusible_siblings[0]->ReplaceAllUsesWith(gte_0);
    fusible_siblings[1]->ReplaceAllUsesWith(gte_1);
    fusible_siblings[2]->ReplaceAllUsesWith(gte_2);
    changed = true;
  }
}

StatusOr<bool> FusedQKVRewriter::DoQKVMultiOutputFusion(
    HloComputation* computation) {
  bool changed = false;
  RecomputeReachability(computation);
  std::vector<HloInstruction*> defs_before_uses =
      computation->MakeInstructionPostOrder();

  FusionInfoCache fusion_info_cache;
  while (!defs_before_uses.empty()) {
    // Traverse the HLO in uses-before-defs order by removing instruction from
    // the back of the vector.
    HloInstruction* producer = defs_before_uses.back();

    // Copy on purpose: to use after removing the producer.
    absl::string_view producer_name = producer->name();
    defs_before_uses.pop_back();

    // producer should have at least 3 consumers
    if (producer->IsDead() || producer->user_count() < 3) {
      continue;
    }
    if (FuseQKVSiblings(computation, /*parent=*/producer, &fusion_info_cache)) {
      changed = true;
    }
  }
  return changed;
}

StatusOr<bool> FusedQKVRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  if (!IsXetlaHardwareSupport()) return changed;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool fusion_changed,
                        DoQKVMultiOutputFusion(computation));
    if (fusion_changed) {
      changed = true;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
