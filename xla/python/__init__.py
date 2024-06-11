# Copyright (c) 2023 Intel Corporation
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
# ==============================================================================
'''Init file for register XPU backend'''

import logging
from pathlib import Path
import platform
import sys

import jax._src.xla_bridge as xb
from jax._src.lib import xla_client
from jax_plugins.intel_extension_for_openxla.version import VersionClass

logger = logging.getLogger(__name__)

def initialize():
  path = Path(__file__).resolve().parent / "pjrt_plugin_xpu.so"
  xla_extension_version = VersionClass()
  logger.warning("INFO: Intel Extension for OpenXLA version: %s, commit: %s",
                 xla_extension_version.get_version(),
                 xla_extension_version.get_hash())
  if not path.exists():
    logger.warning(
        f"WARNING: Native library {path} does not exist. "
        f"This most likely indicates an issue with how {__package__} "
        f"was built or installed.")

  options = dict()

  # xb.CUDA_VISIBLE_DEVICES is set by jax.distribute.initialize(local_device_ids).
  # xb.CUDA_VISIBLE_DEVICES would has default value 'all' if users not call 
  # jax.distribute.initialize or call it without setting local_device_ids.
  visible_devices = xb.CUDA_VISIBLE_DEVICES.value
  if visible_devices != 'all':
    options['visible_devices'] = [int(x) for x in visible_devices.split(',')]
  
  c_api = xb.register_plugin("xpu",
                     priority=500,
                     library_path=str(path),
                     options=options)
  
  try:
    import functools
    from .python import xpu_plugin_extension
    xla_client.register_custom_call_handler(
        "SYCL",
        functools.partial(
            xpu_plugin_extension.register_custom_call_target, c_api
        ),
    )
  except:
    raise RuntimeError("Fail to load xpu_plugin_extension.so.")
