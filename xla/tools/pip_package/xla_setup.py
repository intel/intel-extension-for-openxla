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
# limitations under the License..
# ==============================================================================
'''
lib_setup.py file to build wheel for Intel® Extension for OpenXLA*
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fnmatch
import os
import sys

from datetime import date
from setuptools import setup
from setuptools.command.install import install as InstallCommandBase
from setuptools.dist import Distribution

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                'jax_plugins/intel_extension_for_openxla'))
from version import VersionClass


# This version string is semver compatible, but incompatible with pip.
# For pip, we will remove all '-' characters from this string, and use the
# result for pip.
# Also update xla/xla.bzl
_VERSION = VersionClass().get_version()

REQUIRED_PACKAGES = []

project_name = 'intel_extension_for_openxla'
DEV_VERSION_SUFFIX = ""
if "--weekly_build" in sys.argv:
        today_number = date.today().strftime("%Y%m%d")
        DEV_VERSION_SUFFIX = ".dev" + today_number
        sys.argv.remove("--weekly_build")
        project_name = "xla_lib_weekly"
if '--project_name' in sys.argv:
  project_name_idx = sys.argv.index('--project_name')
  project_name = sys.argv[project_name_idx + 1] + "_lib"
  sys.argv.remove('--project_name')
  sys.argv.pop(project_name_idx)
REQUIRED_PACKAGES.append('wheel')
REQUIRED_PACKAGES.append('numpy>=1.24.0')
REQUIRED_PACKAGES.append('scipy<1.12.0')
CONSOLE_SCRIPTS = []

_ext_path = 'jax_plugins.intel_extension_for_openxla'

class BinaryDistribution(Distribution):

  def has_ext_modules(self):
    return True

def find_files(pattern, root):
  """Return all the files matching pattern below root dir."""
  for dirpath, _, files in os.walk(root):
    for filename in fnmatch.filter(files, pattern):
      yield os.path.join(dirpath, filename)


so_lib_paths = [
    i for i in os.listdir('.')
    if os.path.isdir(i) and fnmatch.fnmatch(i, '_solib_*')
]

matches = []
for path in so_lib_paths:
  matches.extend(
      ['../' + x for x in find_files('*', path) if '.py' not in x]
  )

env_check_tool = ['tools/*']

long_description = ''
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()


class InstallCommand(InstallCommandBase):
  """Override the dir where the headers go."""

  def finalize_options(self):
    ret = InstallCommandBase.finalize_options(self)  # pylint: disable=assignment-from-no-return
    self.install_headers = os.path.join(self.install_platlib, \
                                        'jax_plugins', 'include')
    self.install_lib = self.install_platlib
    return ret

setup(
    name=project_name,
    version=_VERSION.replace('-', '') + DEV_VERSION_SUFFIX,
    description='Intel® Extension for OpenXLA* library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    # pylint: disable=line-too-long
    url='https://github.com/intel/intel-extension-for-openxla',
    download_url='https://github.com/intel/intel-extension-for-openxla/tags',
    project_urls={
        "Bug Tracker": "https://github.com/intel/intel-extension-for-openxla/issues",
    },
    # pylint: enable=line-too-long
    author='Intel Corporation',
    author_email='itex.maintainers@intel.com',
    # Contained modules and scripts.
    packages=[_ext_path],
    entry_points={
        'console_scripts': CONSOLE_SCRIPTS,
    },
    install_requires=REQUIRED_PACKAGES,
    tests_require=REQUIRED_PACKAGES,
    # Add in any packaged data.
    package_data={
        _ext_path: [
            'python/*.so',
            'pjrt_plugin_xpu.so',
            'third-party-programs/*',
        ] + matches,
    },
    exclude_package_data={
        'jax_plugins': ['tools']
    },
    python_requires='>=3.9',
    zip_safe=False,
    distclass=BinaryDistribution,
    # PyPI package information.
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    license='Apache 2.0',
    keywords='Intel® Extension for OpenXLA*',
        cmdclass={
            'install': InstallCommand,
        },
)
