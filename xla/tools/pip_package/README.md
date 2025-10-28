# Intel® Extension for OpenXLA*

[![Python](https://img.shields.io/pypi/pyversions/intel_extension_for_openxla)](https://badge.fury.io/py/intel-extension-for-openxla)
[![PyPI version](https://badge.fury.io/py/intel-extension-for-openxla.svg)](https://badge.fury.io/py/intel-extension-for-openxla)
[![version](https://img.shields.io/github/v/release/intel/intel-extension-for-openxla?color=brightgreen)](https://github.com/intel/intel-extension-for-openxla/releases)

The [OpenXLA](https://github.com/openxla/xla) Project brings together a community of developers and leading AI/ML teams to accelerate ML and address infrastructure fragmentation across ML frameworks and hardware.

Intel® Extension for OpenXLA* includes PJRT plugin implementation, which seamlessly runs JAX models on Intel GPU. The PJRT API simplified the integration, which allowed the Intel GPU plugin to be developed separately and quickly integrated into JAX.

## Installation

The following table tracks intel-extension-for-openxla versions and compatible versions of jax, jaxlib.
| **intel-extension-for-openxla** | **jaxlib** | **jax** |
|:-:|:-:|:-:|
| 0.6.0 | 0.4.38 | 0.4.38 |
| 0.5.0 | 0.4.30 | >= 0.4.30, <= 0.4.31|
| 0.4.0 | 0.4.26 | >= 0.4.26, <= 0.4.27|
| 0.3.0 | 0.4.24 | >= 0.4.24, <= 0.4.27|
| 0.2.1 | 0.4.20 | >= 0.4.20, <= 0.4.26|
| 0.2.0 | 0.4.20 | >= 0.4.20, <= 0.4.26|
| 0.1.0 | 0.4.13 | >= 0.4.13, <= 0.4.14|


```
pip install --upgrade intel-extension-for-openxla
```

## Security
See Intel's [Security Center](https://www.intel.com/content/www/us/en/security-center/default.html) for information on how to report a potential security issue or vulnerability.

See also: [Security Policy](https://github.com/intel/intel-extension-for-openxla/blob/main/security.md)
