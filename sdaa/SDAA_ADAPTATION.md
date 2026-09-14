# roshambo2 龙芯 loongarch64 编译适配记录

> roshambo2 是 3D 分子形状重叠（shape overlay）工具，C++ 核心（pybind11 绑定）+ 可选
> CUDA 后端。本次适配的核心是**在龙芯 loongarch64（无 nvcc）环境禁用 CUDA、
> 只编译 CPU 后端并跑通**。
>
> 本仓库 fork 自官方 roshambo2，**官方功能代码零改动**，适配改动仅 2 处（见下）。

## 一、环境信息

| 项目 | 值 |
|------|-----|
| 架构 | loongarch64（Loongnix OS，龙芯 3A6000） |
| 机器 | `sunhn@10.71.13.47`，容器 `tuyi_test` |
| 编译器 | gcc/g++ 12.3.0 |
| CMake | 3.26.3 |
| Python | 3.12.13（venv `/home/py312`） |
| 构建后端 | pybind11 3.1.0 + setuptools + CMake |
| 分支 | `adapt/sdaa`（基于官方 `DecoupleCPPInterface` 合并版，commit 7ff8237） |
| roshambo2 版本 | 1.0.0 |

## 二、适配背景与改动总览

roshambo2 官方默认 `BUILD_WITH_CUDA=ON`，且代码里有 **3 处硬编码 CUDA 依赖**，
龙芯无 nvcc 会全部卡住。适配就是把这 3 处改为「CUDA 可选，CPU 兜底」：

| 位置 | 官方问题 | 适配方式 |
|------|---------|---------|
| `CMakeLists.txt:2` | `BUILD_WITH_CUDA` 默认 ON | 无需改码，编译时 `CMAKE_ARGS=-DBUILD_WITH_CUDA=OFF` 传入 |
| `setup.py` | 无条件声明 `_roshambo2_cuda` + `_roshambo2_cpp` 两个 ext | **改**：动态 ext_modules，无 CUDA 只声明显式 `_roshambo2_cpp` |
| `roshambo2/backends/__init__.py` | 无条件 `from .cuda_backend import CudaShapeOverlay` | **改**：try/except，CUDA 缺失时降级 `None` |

## 三、代码改动

### 3.1 `setup.py`（动态 ext_modules）

`setup()` 调用前插入，并把 `ext_modules=` 指向 `_ext_modules`：

```python
# 龙芯 loongarch64 适配：无 CUDA 时只构建 CPU 后端（避免 _roshambo2_cuda 产物缺失报错）
# 默认仅 CPU；需 CUDA 时设环境变量 BUILD_WITH_CUDA=ON，或在 CMAKE_ARGS 里加 -DBUILD_WITH_CUDA=ON
_build_cuda = os.environ.get("BUILD_WITH_CUDA", "").upper() == "ON"
if "CMAKE_ARGS" in os.environ and "BUILD_WITH_CUDA=ON" in os.environ["CMAKE_ARGS"]:
    _build_cuda = True
_ext_modules = [CMakeExtension("_roshambo2_cpp")]
if _build_cuda:
    _ext_modules.insert(0, CMakeExtension("_roshambo2_cuda"))

setup(
    ext_modules=_ext_modules,   # 原来是 [CMakeExtension("_roshambo2_cuda"), CMakeExtension("_roshambo2_cpp")]
    cmdclass={"build_ext": CMakeBuild},
    ...
)
```

**原因**：`setup.py` 声明了两个 `CMakeExtension`，`build_ext` 会对每个 ext 跑一次
CMake configure+build，构建结束后检查每个 ext 的 `.so` 产物。`BUILD_WITH_CUDA=OFF` 时
`_roshambo2_cuda.so` 不会生成，导致 `_roshambo2_cuda` 这个 ext 报「产物缺失」构建失败。

### 3.2 `roshambo2/backends/__init__.py`（CUDA 可选）

```python
from .cpp_backend import CppShapeOverlay
try:
    from .cuda_backend import CudaShapeOverlay
except ImportError:
    # 龙芯 loongarch64 适配：无 CUDA 后端（_roshambo2_cuda 未编译）时降级为 None
    CudaShapeOverlay = None
```

**原因**：`cuda_backend.py` 的 `from _roshambo2_cuda import optimize_overlap_color`
在 CPU-only 环境会 `ImportError`，原版 `backends/__init__.py` 无条件 import 会让整个
`roshambo2` 包 import 失败（`roshambo2.py` 第 45 行 `from roshambo2.backends import CppShapeOverlay, CudaShapeOverlay`）。

## 四、依赖准备

```bash
# 核心依赖（rdkit 用自包含 loongarch64 whl，见 RDKit 适配文档）
pip install rdkit-2026.9.1-py3-none-linux_loongarch64.whl
# 构建依赖 + 运行依赖
pip install pybind11 h5py flask requests pyyaml tqdm psutil
```

| 依赖 | 版本 | 说明 |
|------|------|------|
| rdkit | 2026.09.1pre | 自包含 whl（C++ 库 + RPATH 已打进）|
| pybind11 | 3.1.0 | 构建必需（编译期找 pybind11 CMake 配置）|
| h5py | 3.16.0 | 运行必需（读 .h5 数据集），龙芯无预编译 wheel 需源码编译 |
| OpenMP | libgomp（gcc 自带）| `CMakeLists.txt` 里 `find_package(OpenMP REQUIRED)` 需要 |

## 五、编译

```bash
cd /data01/tuyilist/roshambo2
CMAKE_ARGS="-DBUILD_WITH_CUDA=OFF" pip install -e . --no-build-isolation
```

产物：`_roshambo2_cpp.cpython-312-loongarch64-linux-gnu.so`（仅 CPU 后端）。

> 说明：`CMAKE_ARGS` 由 `setup.py` 的 `CMakeBuild.build_extension` 读取（第 75-76 行），
> 拼进 CMake configure 参数；`--no-build-isolation` 避免 build isolation 重新拉 pybind11。

## 六、验证结果

### 6.1 import 验证

```python
import roshambo2
from roshambo2.backends import CppShapeOverlay, CudaShapeOverlay
# CppShapeOverlay: <class '...CppShapeOverlay'>  ✅
# CudaShapeOverlay: None  ✅（CPU 模式降级）
```

### 6.2 端到端计算验证（example 数据）

```python
from roshambo2 import Roshambo2
roshambo2_calculator = Roshambo2("query.sdf", "dataset.sdf", color=False)
scores = roshambo2_calculator.compute(backend="cpp", optim_mode="shape",
                                      reduce_over_conformers=False, start_mode=1)
# 结果：查询 CHEMBL221029，153 个分子，12 列；overlap_volume 707~1170，耗时 1.3s ✅
```

## 七、踩坑记录

| # | 坑 | 现象 | 解决 |
|---|----|------|------|
| 1 | `BUILD_WITH_CUDA` 默认 ON | CMake 找不到 nvcc | `CMAKE_ARGS=-DBUILD_WITH_CUDA=OFF` |
| 2 | setup.py 双 ext | OFF 时 `_roshambo2_cuda` 产物缺失报错 | 动态 ext_modules，OFF 只声明 cpp |
| 3 | backends 无条件 import CUDA | 整个包 import 失败 | `__init__.py` try/except 降级 None |
| 4 | h5py 无 loongarch64 wheel | pip 源码编译（仍成功）| 用龙芯 lpypi 源自动源码编译 |

## 八、运行方式（CPU 后端）

```python
# 与官方用法一致，仅 backend 必须用 "cpp"（龙芯无 CUDA）
from roshambo2 import Roshambo2
calc = Roshambo2("query.sdf", "dataset.sdf")
scores = calc.compute(backend="cpp")   # 官方默认就是 cpp；显式传 cpp 保险
```

官方 `example/basic_run.py` 里 `backend='cuda'`，在龙芯上需改成 `backend='cpp'`。

## 九、产物清单

| 产物 | 位置 | 说明 |
|------|------|------|
| 适配分支 | `adapt/sdaa` | 2 处改动（setup.py + backends/__init__.py）|
| 编译产物 | `_roshambo2_cpp.cpython-312-loongarch64-linux-gnu.so` | CPU 后端模块 |
| 依赖 | rdkit（自包含 whl）+ pybind11 + h5py | 见第四节 |