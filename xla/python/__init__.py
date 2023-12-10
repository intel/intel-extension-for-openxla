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

logger = logging.getLogger(__name__)

def initialize():
  path = Path(__file__).resolve().parent / "pjrt_plugin_xpu.so"
  if not path.exists():
    logger.warning(
        f"WARNING: Native library {path} does not exist. "
        f"This most likely indicates an issue with how {__package__} "
        f"was built or installed.")
  c_api = xb.register_plugin("xpu",
                     priority=500,
                     library_path=str(path))
  
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
