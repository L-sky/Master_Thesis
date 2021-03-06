# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import CUDAExtension, CUDA_HOME

from setup_helper import BuildRDCExtension

# python setup.py develop    - if you wont to be able to execute from PyCharm (or similar IDE) - places .so file into se3cnn folder from which real_spherical_harmonics imports

# Or:
# python setup.py build_ext
# python setup.py install    - PyCharm won't work, because it can't resolve import, but executable from terminal

if not torch.cuda.is_available():
    ext_modules = None
    print("GPU is not available. Skip building CUDA extensions.")
elif torch.cuda.is_available() and CUDA_HOME is not None:
    ext_modules = [
        CUDAExtension('e3_layer.real_spherical_harmonics',
                      sources=['src/real_spherical_harmonics/rsh_bind.cpp',
                               'src/real_spherical_harmonics/rsh_cuda.cu'],
                      extra_compile_args={'cxx': ['-std=c++14'],
                                          'nvcc': ['-std=c++14']}),
        CUDAExtension('e3_layer.pconv_with_kernel',
                      sources=[
                               'src/pconv_with_kernel/link_pconv_cuda.cu',
                               'src/pconv_with_kernel/pconv_cuda.cu',
                               'src/pconv_with_kernel/pconv_bind.cpp'],
                      extra_compile_args={'nvcc': ['-std=c++14', '-rdc=true', '-lcudadevrt'],
                                          'cxx': ['-std=c++14']},
                      libraries=['cudadevrt'])
    ]
else:
    # GPU is available, but CUDA_HOME is None
    raise AssertionError("CUDA_HOME is undefined. Make sure nvcc compiler is available (cuda toolkit installed?)")

setup(
    name='e3_layer',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildRDCExtension},
    packages=find_packages(),
)
