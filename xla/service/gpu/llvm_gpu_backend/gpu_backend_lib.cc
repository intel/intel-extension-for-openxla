/* Copyright (c) 2023 Intel Corporation

Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"

#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "LLVMSPIRVLib.h"
#include "LLVMSPIRVOpts.h"
#include "absl/base/call_once.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/PassRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/DeadArgumentElimination.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/Scalar.h"
#include "tsl/platform/env.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/path.h"
#include "tsl/util/env_var.h"
#include "xla/service/dump.h"
#include "xla/service/gpu/llvm_gpu_backend/utils.h"
#include "xla/service/llvm_ir/llvm_command_line_options.h"
#include "xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/status_macros.h"
#include "xla/types.h"
#include "xla/util.h"

namespace xla {
namespace gpu {
namespace {

static llvm::codegen::RegisterCodeGenFlags CGF;

// Default inline threshold value to use in llvm.
const int kDefaultInlineThreshold = 1100;

// Initializes LLVM passes. Uses the PassRegistry mechanism.
void InitializePasses(llvm::PassRegistry* pass_registry) {
  llvm::initializeCore(*pass_registry);
  llvm::initializeCodeGen(*pass_registry);
  llvm::initializeScalarOpts(*pass_registry);
  llvm::initializeVectorization(*pass_registry);
  llvm::initializeIPO(*pass_registry);
  llvm::initializeAnalysis(*pass_registry);
  llvm::initializeTransformUtils(*pass_registry);
  llvm::initializeInstCombine(*pass_registry);
  llvm::initializeTarget(*pass_registry);
  llvm::initializeCodeGenPreparePass(*pass_registry);
}

void DumpModule(const std::string output_filename, const llvm::Module* module) {
  std::error_code ec;
  auto out = std::make_unique<llvm::raw_fd_ostream>(
      llvm::StringRef(output_filename), ec, llvm::sys::fs::OF_None);
  if (ec) {
    LOG(FATAL) << "Unable to open " << output_filename
               << " to dump LLVM IR: " << ec.message();
    return;
  }
  module->print(*out, /*AAW=*/nullptr);
  out->close();
}

const llvm::Module* GetModule(llvm::Any IR) {
  if (llvm::any_isa<const llvm::Module*>(IR))
    return llvm::any_cast<const llvm::Module*>(IR);

  if (llvm::any_isa<const llvm::Function*>(IR)) {
    const llvm::Function* F = llvm::any_cast<const llvm::Function*>(IR);
    return F->getParent();
  }

  if (llvm::any_isa<const llvm::LazyCallGraph::SCC*>(IR)) {
    const llvm::LazyCallGraph::SCC* C =
        llvm::any_cast<const llvm::LazyCallGraph::SCC*>(IR);
    return C->begin()->getFunction().getParent();
  }

  if (llvm::any_isa<const llvm::Loop*>(IR)) {
    const llvm::Loop* L = llvm::any_cast<const llvm::Loop*>(IR);
    const llvm::Function* F = L->getHeader()->getParent();
    return F->getParent();
  }

  return nullptr;
}

auto DumpCallbackForModule(std::string module_identifier) {
  int i = 0;
  return [module_identifier, i](llvm::StringRef pass, llvm::Any ir) mutable {
    const llvm::Module* module = GetModule(ir);
    if (!module) {
      return;
    }

    const std::string basename = ReplaceFilenameExtension(
        absl::string_view(tsl::io::Basename(module_identifier)),
        absl::StrFormat("pass-%02d.before.%s.ll", i++,
                        absl::string_view(pass.str())));
    std::string outputs_dir;
    tsl::io::GetTestUndeclaredOutputsDir(&outputs_dir);
    DumpModule(tsl::io::JoinPath(outputs_dir, basename), module);
  };
}

// Refer to function `EmitAssemblyHelper::RunOptimizationPipeline` defined in
// clang/lib/CodeGen/BackendUtil.cpp.
void RunOptimizationPipeline(llvm::Module* module,
                             const HloModuleConfig& hlo_module_config,
                             llvm::TargetMachine* target_machine) {
  std::optional<llvm::PGOOptions> PGOOpt;
  llvm::PipelineTuningOptions PTO;
  PTO.LoopUnrolling = 1;
  PTO.LoopInterleaving = 1;
  PTO.LoopVectorization = 1;
  PTO.SLPVectorization = 1;
  PTO.MergeFunctions = 0;
  PTO.CallGraphProfile = 1;

  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;

  llvm::PassInstrumentationCallbacks PIC;
  llvm::PrintPassOptions PrintPassOpts;
  PrintPassOpts.Indent = 0;
  PrintPassOpts.SkipAnalyses = 0;
  llvm::StandardInstrumentations SI(module->getContext(), false, false,
                                    PrintPassOpts);
  SI.registerCallbacks(PIC, &MAM);
  llvm::PassBuilder PB(target_machine, PTO, PGOOpt, &PIC);

#define HANDLE_EXTENSION(Ext) \
  get##Ext##PluginInfo().RegisterPassBuilderCallbacks(PB);
#include "llvm/Support/Extension.def"

  // Register the target library analysis directly and give it a customized
  // preset TLI.
  auto target_triple = llvm::Triple(module->getTargetTriple());
  auto TLII = std::make_unique<llvm::TargetLibraryInfoImpl>(target_triple);
  FAM.registerPass([&] { return llvm::TargetLibraryAnalysis(*TLII); });

  // Register all the basic analyses with the managers.
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  if (hlo_module_config.debug_options().xla_gpu_dump_llvmir()) {
    PIC.registerBeforeNonSkippedPassCallback(
        DumpCallbackForModule(module->getModuleIdentifier()));
  }

  llvm::ModulePassManager MPM;

  llvm::OptimizationLevel Level = llvm::OptimizationLevel::O2;
  MPM = PB.buildPerModuleDefaultPipeline(Level);
  MPM.addPass(llvm::VerifierPass());
  MPM.run(*module, MAM);
}

// LLVM has an extensive flags mechanism of its own, which is only accessible
// through the command line. Internal libraries within LLVM register parsers for
// flags, with no other way to configure them except pass these flags.
// To do this programmatically, we invoke ParseCommandLineOptions manually with
// a "fake argv".
// Note: setting flags with this method is stateful, since flags are just
// static globals within LLVM libraries.
void FeedLLVMWithFlags(const std::vector<std::string>& cl_opts) {
  std::vector<const char*> fake_argv = {""};
  for (const std::string& cl_opt : cl_opts) {
    fake_argv.push_back(cl_opt.c_str());
  }
  llvm::cl::ParseCommandLineOptions(fake_argv.size(), &fake_argv[0]);
}

using TargetModuleLinker = std::function<Status(
    llvm::Module*, const HloModuleConfig&, const std::string&)>;

Status LinkAndOptimizeModule(llvm::Module* module,
                             const HloModuleConfig& hlo_module_config,
                             llvm::Triple default_target_triple,
                             llvm::TargetMachine* target_machine,
                             int inline_threshold) {
  bool opt = true;
  tsl::ReadBoolFromEnvVar("DPCPP_LLVM_OPT", true, &opt);
  if (opt) {
    RunOptimizationPipeline(module, hlo_module_config, target_machine);
  }

  std::string err;
  llvm::raw_string_ostream err_stream(err);

  // verifyModule() returns true if the module is broken.
  TF_RET_CHECK(!llvm::verifyModule(*module, &err_stream))
      << "Invalid LLVM IR after dpcpp optimizations:\n"
      << err_stream.str() << "\n";

  return OkStatus();
}

// One-time module initializer.
// Must be called only once -- DO NOT CALL DIRECTLY.
void SPIRBackendInit(const HloModuleConfig& hlo_module_config) {
  // Feed all customized flags here, so we can override them with llvm_cl_opts
  // without redeploy the compiler for development purpose.

  // This flag tunes a threshold in branch folding. The default threshold, which
  // is one, is not suitable for CUDA programs where branches are more expensive
  // than for CPU programs. Setting the threshold to 2 improves the latency of
  // TwoDPatchDotProductKernel_IND_3_ND_48 by over 5%, and does not affect the
  // latency of other benchmarks so far.
  //
  // I also tried setting this threshold to other values:
  // * 3-6 gives similar results as 2;
  // * >6 start hurting the performance of at least dot product kernels.
  //
  // TODO(jingyue): The current threshold only considers the number of IR
  // instructions which do not accurately reflect the true cost. We need a
  // better cost model.
  FeedLLVMWithFlags({"-bonus-inst-threshold=2"});
  // Increase limit when scanning memory dependencies.  This helps to reduce
  // more redundant load instructions.
  //
  // The specific value is currently large enough for s3d in shoc benchmark,
  // which contains a lot of load instructions and many arithmetic instructions
  // between those loads.
  FeedLLVMWithFlags({"-memdep-block-scan-limit=500"});

  bool vec = false;
  tsl::ReadBoolFromEnvVar("VECTORIZE", false, &vec);
  if (vec) {
    FeedLLVMWithFlags({
        "-slp-vectorize-hor=false",
        "-slp-min-reg-size=64",
        "-slp-max-reg-size=64",
    });
  } else {
    // TODO: sycl-opt disables all LLVM vectorization passes. Evaluate if it is
    // needed.
    FeedLLVMWithFlags({"-sycl-opt=1"});
  }

  llvm_ir::InitializeLLVMCommandLineOptions(
      hlo_module_config.debug_options().xla_backend_extra_options());

  // Initialize the LLVM optimization passes.
  llvm::PassRegistry* registry = llvm::PassRegistry::getPassRegistry();
  InitializePasses(registry);
}

}  // namespace

namespace {
StatusOr<std::string> EmitModuleToSpir(llvm::Module* module,
                                       const HloModuleConfig& module_config) {
  SPIRV::TranslatorOpts::ExtensionsStatusMap ExtensionsStatus;
  SPIRV::TranslatorOpts opts(SPIRV::VersionNumber::MaximumVersion,
                             ExtensionsStatus);
  opts.enableAllExtensions();  // enable all SPIR-V extension first

  std::ostringstream oss;
  std::string err;
  bool success = llvm::writeSpirv(module, opts, oss, err);
  if (!success) {
    return xla::InternalError("Fails to convert LLVM as SPIR-V: %s", err);
  }
  return oss.str();
}
}  // namespace

namespace spir {
StatusOr<std::string> CompileToSpir(llvm::Module* module,
                                    const HloModuleConfig& hlo_module_config,
                                    const std::string& libdevice_dir_path) {
  static absl::once_flag backend_init_flag;
  absl::call_once(backend_init_flag, SPIRBackendInit, hlo_module_config);

  std::string spir;
  {
    // itex::profiler::TraceMe activity(
    //     [&] { return absl::StrCat("Compiling IR:", module->getName().str());
    //     }, itex::profiler::TraceMeLevel::kInfo);
    XLA_SCOPED_LOGGING_TIMER("Compile module " + module->getName().str());

    // If the module has no functions or globals, there's nothing to compile.
    // Just return an empty string.
    if (module->empty() && module->global_empty()) {
      VLOG(2) << "Module '" << module->getName().str()
              << "' is empty. Skipping compilation.";
      return std::string();
    }

    // No SPIR target machine?
    llvm::Triple default_target_triple("spir64-unknown-unknown");

    bool reuse = true;
    tsl::ReadBoolFromEnvVar("TF_LLVM_OPT", true, &reuse);
    if (reuse) {
      // Link with libdevice, and optimize the LLVM module.
      TF_RETURN_IF_ERROR(LinkAndOptimizeModule(module, hlo_module_config,
                                               default_target_triple, nullptr,
                                               kDefaultInlineThreshold));
    }

    DumpToFileInDir(hlo_module_config.debug_options(), "module_opt.ll",
                    llvm_ir::DumpToString(module));

    // Lower optimized LLVM module to SPIR.
    TF_ASSIGN_OR_RETURN(spir, EmitModuleToSpir(module, hlo_module_config));
  }
  return spir;
}
}  // namespace spir

}  // namespace gpu
}  // namespace xla
