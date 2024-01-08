import io
import os
import re
import subprocess
from typing import List, Set
import warnings

from packaging.version import parse, Version
import setuptools
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

ROOT_DIR = os.path.dirname(__file__)

# Supported NVIDIA GPU architectures.
SUPPORTED_ARCHS = {"8.0"}

# Compiler flags.
CXX_FLAGS = ["-g", "-O2", "-std=c++17"]
# TODO(woosuk): Should we use -O3?
NVCC_FLAGS = ["-O2", "-std=c++17"]

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0


NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__"]

if CUDA_HOME is None:
    raise RuntimeError(
        "Cannot find CUDA_HOME. CUDA must be available to build the package.")


def get_nvcc_cuda_version(cuda_dir: str) -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    nvcc_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                          universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


def get_torch_arch_list() -> Set[str]:
    # TORCH_CUDA_ARCH_LIST can have one or more architectures,
    # e.g. "8.0" or "7.5,8.0,8.6+PTX". Here, the "8.6+PTX" option asks the
    # compiler to additionally include PTX code that can be runtime-compiled
    # and executed on the 8.6 or newer architectures. While the PTX code will
    # not give the best performance on the newer architectures, it provides
    # forward compatibility.
    env_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
    if env_arch_list is None:
        return set()

    # List are separated by ; or space.
    torch_arch_list = set(env_arch_list.replace(" ", ";").split(";"))
    if not torch_arch_list:
        return set()

    # Filter out the invalid architectures and print a warning.
    valid_archs = SUPPORTED_ARCHS.union({s + "+PTX" for s in SUPPORTED_ARCHS})
    arch_list = torch_arch_list.intersection(valid_archs)
    # If none of the specified architectures are valid, raise an error.
    if not arch_list:
        raise RuntimeError(
            "None of the CUDA architectures in `TORCH_CUDA_ARCH_LIST` env "
            f"variable ({env_arch_list}) is supported. "
            f"Supported CUDA architectures are: {valid_archs}.")
    invalid_arch_list = torch_arch_list - valid_archs
    if invalid_arch_list:
        warnings.warn(
            f"Unsupported CUDA architectures ({invalid_arch_list}) are "
            "excluded from the `TORCH_CUDA_ARCH_LIST` env variable "
            f"({env_arch_list}). Supported CUDA architectures are: "
            f"{valid_archs}.")
    return arch_list


# First, check the TORCH_CUDA_ARCH_LIST environment variable.
compute_capabilities = get_torch_arch_list()
print(f"compute_capabilities: {compute_capabilities}")

if not compute_capabilities:
    # If TORCH_CUDA_ARCH_LIST is not defined or empty, target all available
    # GPUs on the current machine.
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        major, minor = torch.cuda.get_device_capability(i)
        if major < 7:
            raise RuntimeError(
                "GPUs with compute capability below 7.0 are not supported.")
        compute_capabilities.add(f"{major}.{minor}")

nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)
if not compute_capabilities:
    # If no GPU is specified nor available, add all supported architectures
    # based on the NVCC CUDA version.
    compute_capabilities = SUPPORTED_ARCHS.copy()
    if nvcc_cuda_version < Version("11.1"):
        compute_capabilities.remove("8.6")
    if nvcc_cuda_version < Version("11.8"):
        compute_capabilities.remove("8.9")
        compute_capabilities.remove("9.0")

# Validate the NVCC CUDA version.
if nvcc_cuda_version < Version("11.0"):
    raise RuntimeError("CUDA 11.0 or higher is required to build the package.")
if nvcc_cuda_version < Version("11.1"):
    if any(cc.startswith("8.6") for cc in compute_capabilities):
        raise RuntimeError(
            "CUDA 11.1 or higher is required for compute capability 8.6.")
if nvcc_cuda_version < Version("11.8"):
    if any(cc.startswith("8.9") for cc in compute_capabilities):
        # CUDA 11.8 is required to generate the code targeting compute capability 8.9.
        # However, GPUs with compute capability 8.9 can also run the code generated by
        # the previous versions of CUDA 11 and targeting compute capability 8.0.
        # Therefore, if CUDA 11.8 is not available, we target compute capability 8.0
        # instead of 8.9.
        warnings.warn(
            "CUDA 11.8 or higher is required for compute capability 8.9. "
            "Targeting compute capability 8.0 instead.")
        compute_capabilities = set(cc for cc in compute_capabilities
                                   if not cc.startswith("8.9"))
        compute_capabilities.add("8.0+PTX")
    if any(cc.startswith("9.0") for cc in compute_capabilities):
        raise RuntimeError(
            "CUDA 11.8 or higher is required for compute capability 9.0.")

# Add target compute capabilities to NVCC flags.
for capability in compute_capabilities:
    num = capability[0] + capability[2]
    NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=sm_{num}"]
    if capability.endswith("+PTX"):
        NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=compute_{num}"]

# Use NVCC threads to parallelize the build.
if nvcc_cuda_version >= Version("11.2"):
    num_threads = min(os.cpu_count(), 8)
    NVCC_FLAGS += ["--threads", str(num_threads)]

ext_modules = []

sources=["cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp",
        "cpp/tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.cu",
        "cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_bf16_bf16.cu",
        "cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_bf16_uint4.cu",
        "cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_bf16_uint8.cu",
        "cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp16_fp16.cu",
        "cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp16_uint4.cu",
        "cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp16_uint8.cu",
        "cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_kernels_fp32_fp32.cu",
        "cpp/tensorrt_llm/common/stringUtils.cpp",
        "cpp/tensorrt_llm/common/logger.cpp",
        "cpp/tensorrt_llm/common/tllmException.cpp",
        "tests/moe/moe.cpp"
        ]

base_dir = os.path.abspath(os.path.dirname(__file__))
include_path = []
include_path.append(os.path.join(base_dir, '3rdparty/cutlass/include/'))
include_path.append(os.path.join(base_dir, 'cpp/'))
include_path.append(os.path.join(base_dir, 'cpp/tensorrt_llm/cutlass_extensions/include/'))
include_path.append("/usr/local/tensorrt/include/")

th_moe_extension = CUDAExtension(
    name="moe_ops",
    sources=sources,
    extra_compile_args={"nvcc": NVCC_FLAGS},
    include_dirs=include_path
)
ext_modules.append(th_moe_extension)



def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def read_readme() -> str:
    """Read the README file."""
    return "从 trt-llm 中单独把 moe 分离出来，并封装为 pytorch 可以直接使用的算子"


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    with open(get_path("requirements.txt")) as f:
        requirements = f.read().strip().split("\n")
    return requirements


setuptools.setup(
    name="moe_ops",
    version="0.0.1",
    author="weishengying",
    license="Apache 2.0",
    description=("moe for pytorch"),
    long_description=read_readme(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)


'''.ssh/
/usr/local/cuda/bin/nvcc  -I/mnt/infra/weishengying/TensorRT-LLM-Skywork/cpp/ -I/mnt/infra/weishengying/TensorRT-LLM-Skywork/cpp/tensorrt_llm/cutlass_extensions/include/ -I/usr/local/lib/python3.10/dist-packages/torch/include -I/usr/local/lib/python3.10/dist-packages/torch/include/torch/csrc/api/include -I/usr/local/lib/python3.10/dist-packages/torch/include/TH -I/usr/local/lib/python3.10/dist-packages/torch/include/THC -I/usr/local/cuda/include -I/usr/include/python3.10 -c -c /mnt/infra/weishengying/TensorRT-LLM-Skywork/cpp/tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.cu -o /mnt/infra/weishengying/TensorRT-LLM-Skywork/build/temp.linux-x86_64-3.10/cpp/tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.o --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 -gencode arch=compute_80,code=sm_80 --threads 8 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1016"' -DTORCH_EXTENSION_NAME=moe_ops -D_GLIBCXX_USE_CXX11_ABI=1

'''