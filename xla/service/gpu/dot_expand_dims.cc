/* Copyright (c) 2023 Intel Corporation

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

#include "xla/service/gpu/dot_expand_dims.h"

#include <cstdlib>
#include <functional>
#include <numeric>
#include <utility>
#include <vector>

#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

StatusOr<bool> ExpandDotDims(HloInstruction* original_dot) {
  if (original_dot->opcode() != HloOpcode::kDot) {
    return false;
  }
  auto computation = original_dot->parent();
  const auto& original_dnums = original_dot->dot_dimension_numbers();

  const int64_t num_batch_dims = original_dnums.lhs_batch_dimensions_size();
  const int64_t lhs_contracting_dims =
      original_dnums.lhs_contracting_dimensions_size();
  const int64_t rhs_contracting_dims =
      original_dnums.rhs_contracting_dimensions_size();

  HloInstruction* lhs_operand = original_dot->mutable_operand(0);
  const auto& lhs_shape = lhs_operand->shape();
  const int64_t lhs_rank = lhs_shape.rank();
  HloInstruction* rhs_operand = original_dot->mutable_operand(1);
  const auto& rhs_shape = rhs_operand->shape();
  const int64_t rhs_rank = rhs_shape.rank();

  int64_t lhs_contracting_size = 1;
  int64_t lhs_non_contracting_size = 1;
  std::vector<int64_t> batch_dim_sizes;
  batch_dim_sizes.reserve(num_batch_dims);
  for (int64_t i = 0; i < lhs_rank; ++i) {
    if (absl::c_linear_search(original_dnums.lhs_contracting_dimensions(), i)) {
      lhs_contracting_size *= lhs_shape.dimensions(i);
    } else if (absl::c_linear_search(original_dnums.lhs_batch_dimensions(),
                                     i)) {
      batch_dim_sizes.push_back(lhs_shape.dimensions(i));
    } else {
      lhs_non_contracting_size *= lhs_shape.dimensions(i);
    }
  }
  int64_t rhs_contracting_size = 1;
  int64_t rhs_non_contracting_size = 1;
  for (int64_t i = 0; i < rhs_rank; ++i) {
    if (absl::c_linear_search(original_dnums.rhs_contracting_dimensions(), i)) {
      rhs_contracting_size *= rhs_shape.dimensions(i);
    } else if (!absl::c_linear_search(original_dnums.rhs_batch_dimensions(),
                                      i)) {
      rhs_non_contracting_size *= rhs_shape.dimensions(i);
    }
  }

  if (lhs_non_contracting_size != 1 && rhs_non_contracting_size != 1)
    return false;

  // lhs: [B0, B1, ..., N, C]
  std::vector<int64_t> lhs_reshape_dims = batch_dim_sizes;
  lhs_reshape_dims.push_back(lhs_non_contracting_size);
  lhs_reshape_dims.push_back(lhs_contracting_size);

  HloInstruction* lhs_reshaped = lhs_operand;
  if (lhs_non_contracting_size == 1)
    lhs_reshaped = computation->AddInstruction(
        HloInstruction::CreateReshape(
            ShapeUtil::MakeShape(lhs_shape.element_type(), lhs_reshape_dims),
            lhs_operand),
        &lhs_operand->metadata());

  // rhs: [B0, B1, ..., C, N]
  std::vector<int64_t> rhs_reshape_dims = batch_dim_sizes;
  rhs_reshape_dims.push_back(rhs_contracting_size);
  rhs_reshape_dims.push_back(rhs_non_contracting_size);

  HloInstruction* rhs_reshaped = rhs_operand;
  if (rhs_non_contracting_size == 1)
    rhs_reshaped = computation->AddInstruction(
        HloInstruction::CreateReshape(
            ShapeUtil::MakeShape(rhs_shape.element_type(), rhs_reshape_dims),
            rhs_operand),
        &rhs_operand->metadata());

  // Create new dot.
  std::vector<int64_t> dot_dims = batch_dim_sizes;
  dot_dims.push_back(lhs_non_contracting_size);
  dot_dims.push_back(rhs_non_contracting_size);

  DotDimensionNumbers dot_dnums;
  for (int64_t i = 0; i < num_batch_dims; ++i) {
    dot_dnums.add_lhs_batch_dimensions(i);
    dot_dnums.add_rhs_batch_dimensions(i);
  }
  dot_dnums.add_lhs_contracting_dimensions(num_batch_dims + 1);
  dot_dnums.add_rhs_contracting_dimensions(num_batch_dims);

  HloInstruction* dot = computation->AddInstruction(HloInstruction::CreateDot(
      ShapeUtil::MakeShape(original_dot->shape().element_type(), dot_dims),
      lhs_reshaped, rhs_reshaped, dot_dnums, original_dot->precision_config()));
  original_dot->SetupDerivedInstruction(dot);

  std::unique_ptr<HloInstruction> replacement =
      HloInstruction::CreateReshape(original_dot->shape(), dot);

  TF_RETURN_IF_ERROR(computation->ReplaceWithNewInstruction(
      original_dot, std::move(replacement)));
  return true;
}

}  // namespace

DotExpandDims::DotExpandDims() = default;

StatusOr<bool> DotExpandDims::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool any_changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      bool changed = false;
      TF_ASSIGN_OR_RETURN(changed, ExpandDotDims(instr));
      any_changed |= changed;
    }
  }
  return any_changed;
}

}  // namespace gpu
}  // namespace xla