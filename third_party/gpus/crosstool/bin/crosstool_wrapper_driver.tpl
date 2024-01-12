#!/usr/bin/env python

"""Crosstool wrapper for compiling DPC++ program
SYNOPSIS:
  crosstool_wrapper_driver [options passed in by cc_library()
                            or cc_binary() rule]

DESCRIPTION:
  call DPC++ compiler for device-side code, and host
  compiler for other code.
"""

from __future__ import print_function
from argparse import ArgumentParser
import os
import subprocess
import re
import sys
import shlex
import tempfile

TMPDIR = "%{TMP_DIRECTORY}"

os.environ["TMPDIR"] = TMPDIR
os.environ["TEMP"] = TMPDIR
os.environ["TMP"] = TMPDIR

if not os.path.exists(TMPDIR):
  os.makedirs(TMPDIR, exist_ok=True)

def check_is_intel_llvm(path):
  cmd = path + " -dM -E -x c /dev/null | grep '__INTEL_LLVM_COMPILER'"
  check_result = subprocess.getoutput(cmd)
  if len(check_result) > 0 and check_result.find('__INTEL_LLVM_COMPILER') > -1:
    return True
  return False

SYCL_PATH = os.path.join("%{sycl_compiler_root}", "bin/icx")

if not os.path.exists(SYCL_PATH):
  SYCL_PATH = os.path.join('%{sycl_compiler_root}', 'bin/clang')
  if not os.path.exists(SYCL_PATH) or check_is_intel_llvm(SYCL_PATH):
    raise RuntimeError("compiler not found or invalid")

CPU_COMPILER = ('%{cpu_compiler}')
basekit_path = "%{basekit_path}"
basekit_version = "%{basekit_version}"

result = subprocess.run(["which", "ar"], capture_output=True, text=True)
if result.returncode == 0:
    AR_PATH = result.stdout.strip()  # Remove any trailing newline
else:
    raise RuntimeError("ar not found or invalid")

def system(cmd):
  """Invokes cmd with os.system()"""
  
  ret = os.system(cmd)
  if os.WIFEXITED(ret):
    return os.WEXITSTATUS(ret)
  else:
    return -os.WTERMSIG(ret)

def GetHostCompilerOptions(argv, xetla):
  parser = ArgumentParser()
  args, leftover = parser.parse_known_args(argv)
  sycl_host_compile_flags = leftover
  if xetla:
    sycl_host_compile_flags.append('-std=c++20')
    sycl_host_compile_flags.append("-DXETPP_NEW_XMAIN")
  else:
    sycl_host_compile_flags.append('-std=c++17')
  host_flags = ['-fsycl-host-compiler-options=\'%s\'' % (' '.join(sycl_host_compile_flags))]
  return host_flags

def call_compiler(argv, is_sycl = False, link = False, sycl_compile = True, xetla = False):
  parser = ArgumentParser()
  parser.add_argument('-c', nargs=1, action='append')
  parser.add_argument('-o', nargs=1, action='append')
  args, leftover = parser.parse_known_args(argv)

  flags = leftover

  sycl_device_only_flags = ['-fsycl']
  sycl_device_only_flags.append('-fno-sycl-unnamed-lambda')
  sycl_device_only_flags.append('-fsycl-targets=spir64_gen,spir64')
  sycl_device_only_flags.append('-sycl-std=2020')
  sycl_device_only_flags.append('-fhonor-infinities')
  sycl_device_only_flags.append('-fhonor-nans')
  sycl_device_only_flags.append('-fno-sycl-use-footer')
  sycl_device_only_flags.append('-Xclang -fdenormal-fp-math=preserve-sign')
  sycl_device_only_flags.append('-Xclang -cl-mad-enable')
  sycl_device_only_flags.append('-cl-fp32-correctly-rounded-divide-sqrt')
  sycl_device_only_flags.append('-fsycl-device-code-split=per_source')

  common_flags = []
  common_flags.extend([%{sycl_builtin_include_directories}])
  # ref: https://github.com/intel/llvm/blob/sycl/clang/docs/UsersManual.rst#controlling-floating-point-behavior
  common_flags.append("-fno-finite-math-only")
  common_flags.append("-fno-fast-math")
  common_flags.append("-fexceptions")

  compile_flags = []
  compile_flags.append('-DDNNL_GRAPH_WITH_SYCL=1')
  if xetla:
    compile_flags.append("-std=c++20")
    compile_flags.append("-DXETPP_NEW_XMAIN")
  else:
    compile_flags.append("-std=c++17")

  # link flags
  link_flags = ['-fPIC']
  link_flags.append('-lsycl')
  link_flags.append("-Wl,-no-as-needed")
  link_flags.append("-Wl,--enable-new-dtags")
  link_flags.append("-Wl,-rpath=%{SYCL_ROOT_DIR}/lib/")
  link_flags.append("-Wl,-rpath=%{SYCL_ROOT_DIR}/compiler/lib/intel64_lin/")
  link_flags.append("-lze_loader")
  link_flags.append("-lOpenCL")

  sycl_link_flags = ['-fPIC']
  sycl_link_flags.append("-fsycl")
  sycl_link_flags.append('-fsycl-max-parallel-link-jobs=8')
  sycl_link_flags.append('-fsycl-link')
  if xetla:
    sycl_link_flags.append('-Xs "-doubleGRF -Xfinalizer -printregusage  -Xfinalizer -DPASTokenReduction  -Xfinalizer -enableBCR"')
  else:
    sycl_link_flags.append('-Xs \'-options "-cl-poison-unsupported-fp64-kernels -cl-intel-enable-auto-large-GRF-mode"\'')

  # TODO: enable host-compiler for icx
  # sycl_device_only_flags.append('-fsycl-host-compiler=%{cpu_compiler}')
  # host_flags = GetHostCompilerOptions(flags+common_flags, xetla)

  in_files, out_files = [], []
  if sycl_compile:
    # device compilation
    if args.c:
      in_files.append('-c')
      in_files.extend(args.c[0])
      assert len(args.c[0]) == 1
    if args.o:
      out_files.append('-o')
      out_files.extend(args.o[0])
      assert len(args.o[0]) == 1

    in_file = args.c[0][0]
    out_file = args.o[0][0]
    out_file_name = out_file.split('/')[-1].split('.')[0]

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
      object_file = os.path.join(temp_dir, out_file_name + '.compile.o')
      dev_file = os.path.join(temp_dir, out_file_name + '.dev.o')

      # compile object file
      # icx -fsycl -c kernel.cpp -o kernel.compile.o
      sycl_compile_flags = [" -c {} -o {} ".format(in_file, object_file)]
      sycl_compile_flags += (flags + common_flags + compile_flags + sycl_device_only_flags)
      compile_cmd = ('env ' + 'TMPDIR=' + TMPDIR  + ' ' + 'TEMP=' + TMPDIR + ' ' + 'TMP=' + TMPDIR + ' ' + SYCL_PATH + ' ' + ' '.join(sycl_compile_flags))
      exit_status = system(compile_cmd)
      if exit_status != 0:
        return exit_status

      # generate device object file that can be used by host compiler
      # icx -fsycl -fPIC -fsycl-link kernel.compile.o -o kernel.dev.o
      sycl_link_flags_dev = [" {} -o {} ".format(object_file, dev_file)]
      sycl_link_flags_dev += (common_flags + sycl_link_flags)
      link_cmd = ('env ' + 'TMPDIR=' + TMPDIR  + ' ' + 'TEMP=' + TMPDIR + ' ' + 'TMP=' + TMPDIR + ' ' + SYCL_PATH + ' ' + ' '.join(sycl_link_flags_dev))
      exit_status = system(link_cmd)
      if exit_status != 0:
        return exit_status

      # archive object files
      # ar rcsD output.o kernel.compile.o kernel.dev.o
      ar_flags = " rcsD {} {} {}".format(out_file, object_file, dev_file)
      ar_cmd = ('env ' + 'TMPDIR=' + TMPDIR  + ' ' + 'TEMP=' + TMPDIR + ' ' + 'TMP=' + TMPDIR + ' ' + AR_PATH + ar_flags)
      return system(ar_cmd)
  elif link:
    # sycl link
    out_files.append('-o')
    out_files.extend(args.o[0])
    flags += (common_flags + in_files + out_files + link_flags)
    cmd = ('env ' + 'TMPDIR=' + TMPDIR  + ' ' + 'TEMP=' + TMPDIR + ' ' + 'TMP=' + TMPDIR + ' ' + CPU_COMPILER + ' ' + ' '.join(flags))
    return system(cmd)
  else:
    # host compilation
    if args.c:
      in_files.append('-c')
      in_files.extend(args.c[0])
    if args.o:
      out_files.append('-o')
      out_files.extend(args.o[0])
    flags += (common_flags + in_files + out_files)
    cmd = ('env ' + 'TMPDIR=' + TMPDIR  + ' ' + 'TEMP=' + TMPDIR + ' ' + 'TMP=' + TMPDIR + ' ' + SYCL_PATH + ' ' + ' '.join(flags))
    return system(cmd)

def main():
  parser = ArgumentParser()
  parser = ArgumentParser(fromfile_prefix_chars='@')
  parser.add_argument('-target', nargs=1)
  parser.add_argument('--xetla', action='store_true')
  parser.add_argument('-sycl_compile', action='store_true')
  parser.add_argument('-link_stage', action='store_true')
  args, leftover = parser.parse_known_args(sys.argv[1:])

  leftover = [shlex.quote(s) for s in leftover]
  return call_compiler(leftover, is_sycl=(args.target and args.target[0] == 'sycl'), link=args.link_stage, sycl_compile=args.sycl_compile, xetla=args.xetla)

if __name__ == '__main__':
  sys.exit(main())
