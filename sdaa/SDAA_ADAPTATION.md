# roshambo2  SDAA 迁移适配记录

> roshambo2 是 3D 分子形状重叠（shape overlap）工具，**双后端架构**：CUDA 后端
> （自定义 CUDA kernel，GPU 细粒度并行）+ CPU 后端（OpenMP 多线程纯 C++）。
> 本次适配的核心是 **CUDA 自定义算子无法在龙芯 loongarch64 / SDAA 上编译运行，
> 采用降级方案：禁用 CUDA 后端，走 CPU 后端（OpenMP），功能完全等价**。
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

## 二、自定义算子分析

### 2.1 CUDA 后端 — 自定义 CUDA kernel

源码 `roshambo2/backends/cuda_src/cuda_functions.cu`（26KB），核心是一个 `__global__`
kernel + 一组 `__device__` 自定义算子，做「四元数刚体变换 + 体积重叠 + 梯度优化」：

| 算子 | 类型 | 作用 |
|------|------|------|
| `optimize` | `__global__` kernel | **主优化算子**：`idx = blockIdx.x*blockDim.x + threadIdx.x`，每个 GPU thread 独立优化一个分子对的 overlap（`gridDim = ceil(Nv/blockDim)`，Nv=分子对数）|
| `volume_single` | `__device__` | 计算两分子的高斯体积重叠（shape overlay 核心）|
| `get_gradient` | `__device__` | 求 overlap 对四元数/平移参数的梯度 |
| `adagrad_step` | `__device__` | Adagrad 自适应步长优化更新 |
| `quaternion_to_rotation_matrix` | `__device__` | 四元数 → 旋转矩阵 |
| `axis_angle_to_quat` | `__device__` | 轴角 → 四元数（起始模式初始化）|
| `start_mode_transform` | `__device__` | 起始模式刚体变换 |
| `transform_inplace` / `translate_inplace` | `__device__` | 原地旋转 / 平移 |
| `matvec3x3x3` / `cross_product` | `__device__` | 基础 3x3 矩阵/向量数学 |

### 2.2 CPU 后端 — OpenMP 纯 C++ 等价实现

源码 `roshambo2/backends/cpp_src/cpp_functions.cpp` + `cpp_helper_functions.cpp`：
同一套 shape overlap 算法，`#pragma omp parallel for` + `omp_get_max_threads()` 用满
CPU 线程，每个 dataset 构象一个线程并行。入口都是 `optimize_overlap_color`（pybind 绑定）。

### 2.3 双后端并行模型对比

| | CUDA 后端 | CPU 后端 |
|--|----------|---------|
| 并行粒度 | GPU thread（每 thread 一个分子对）| OpenMP 线程（每线程一个构象）|
| 算子实现 | 自定义 CUDA kernel | 纯 C++（同算法）|
| 加速硬件 | NVIDIA GPU | 多核 CPU |
| pybind 模块 | `_roshambo2_cuda` | `_roshambo2_cpp` |

### 2.4 SDAA 适配决策

roshambo2 的 CUDA 自定义算子属于**手写 CUDA kernel**（不依赖 torch、纯 CUDA C++），
在龙芯 loongarch64 / SDAA（昇腾 NPU）上**没有 CUDA 工具链（nvcc + CUDA runtime），
无法编译 `.cu` 文件**。因此采用**降级方案**：

> **禁用 CUDA 后端 → 走 CPU 后端（OpenMP + 纯 C++），功能完全等价**
> （与 Protenix「triangle 算子 kernel → 纯 torch」、RoseTTAFold2-PPI「CUDA kernel → CPU
> 等价实现」是同一类迁移思路：自定义算子无法在目标平台编译时，降级到通用等价实现。）

无需给 NPU 重写 CUDA kernel（shape overlay 是 GPU 细粒度并行算法，重写收益低），
CPU OpenMP 后端在龙芯多核上已足够（见第六节，153 分子对 1.3s）。

## 三、代码改动（正确禁用 CUDA 后端）

官方代码有 3 处硬编码 CUDA，龙芯会卡住。适配让「CUDA 可选、CPU 兜底」：

| 位置 | 官方问题 | 适配方式 |
|------|---------|---------|
| `CMakeLists.txt:2` | `BUILD_WITH_CUDA` 默认 ON（找不到 nvcc）| 不改码，编译时 `CMAKE_ARGS=-DBUILD_WITH_CUDA=OFF` |
| `setup.py` | 无条件声明 `_roshambo2_cuda` + `_roshambo2_cpp` 两个 ext | **改**：动态 ext_modules，无 CUDA 只声明 `_roshambo2_cpp` |
| `roshambo2/backends/__init__.py` | 无条件 `from .cuda_backend import CudaShapeOverlay` | **改**：try/except，无 CUDA 降级 `None` |

### 3.1 `setup.py`（动态 ext_modules）

```python
# 龙芯 loongarch64 适配：无 CUDA 时只构建 CPU 后端（避免 _roshambo2_cuda 产物缺失报错）
_build_cuda = os.environ.get("BUILD_WITH_CUDA", "").upper() == "ON"
if "CMAKE_ARGS" in os.environ and "BUILD_WITH_CUDA=ON" in os.environ["CMAKE_ARGS"]:
    _build_cuda = True
_ext_modules = [CMakeExtension("_roshambo2_cpp")]
if _build_cuda:
    _ext_modules.insert(0, CMakeExtension("_roshambo2_cuda"))

setup(ext_modules=_ext_modules, cmdclass={"build_ext": CMakeBuild}, ...)
```

### 3.2 `roshambo2/backends/__init__.py`（CUDA 可选）

```python
from .cpp_backend import CppShapeOverlay
try:
    from .cuda_backend import CudaShapeOverlay
except ImportError:
    CudaShapeOverlay = None   # 无 CUDA（_roshambo2_cuda 未编译）时降级
```

## 四、依赖准备

```bash
pip install rdkit-2026.9.1-py3-none-linux_loongarch64.whl   # 自包含 loongarch64 whl
pip install pybind11 h5py flask requests pyyaml tqdm psutil
```

| 依赖 | 版本 | 说明 |
|------|------|------|
| rdkit | 2026.09.1pre | 自包含 whl（C++ 库 + RPATH 打进）|
| pybind11 | 3.1.0 | 编译期找 pybind11 CMake 配置 |
| h5py | 3.16.0 | 读 .h5 数据集，龙芯源码编译 |
| OpenMP | libgomp（gcc 自带）| CPU 后端并行必需（`find_package(OpenMP REQUIRED)`）|

## 五、编译

```bash
cd /data01/tuyilist/roshambo2
CMAKE_ARGS="-DBUILD_WITH_CUDA=OFF" pip install -e . --no-build-isolation
```

产物：`_roshambo2_cpp.cpython-312-loongarch64-linux-gnu.so`（仅 CPU 后端）。

## 六、验证结果

### 6.1 import 验证

```python
import roshambo2
from roshambo2.backends import CppShapeOverlay, CudaShapeOverlay
# CppShapeOverlay = <class '...CppShapeOverlay'>  ✅
# CudaShapeOverlay = None  ✅（CUDA 后端已禁用，正确降级）
```

### 6.2 端到端计算验证（CPU 后端，example 数据）

```python
from roshambo2 import Roshambo2
calc = Roshambo2("query.sdf", "dataset.sdf", color=False)
scores = calc.compute(backend="cpp", optim_mode="shape", reduce_over_conformers=False, start_mode=1)
# 结果：查询 CHEMBL221029，153 分子对，12 列；overlap_volume 707~1170，耗时 1.3s ✅
```

## 七、踩坑记录

| # | 坑 | 现象 | 解决 |
|---|----|------|------|
| 1 | `BUILD_WITH_CUDA` 默认 ON | 找不到 nvcc，CUDA kernel 无法编译 | `CMAKE_ARGS=-DBUILD_WITH_CUDA=OFF` |
| 2 | setup.py 双 ext | OFF 时 `_roshambo2_cuda` 产物缺失报错 | 动态 ext_modules，OFF 只声明 cpp |
| 3 | backends 无条件 import CUDA | 整个包 import 失败 | `__init__.py` try/except 降级 None |
| 4 | h5py 无 loongarch64 wheel | pip 需源码编译 | 龙芯 lpypi 源自动源码编译 |

## 八、运行方式（CPU 后端）

```python
from roshambo2 import Roshambo2
calc = Roshambo2("query.sdf", "dataset.sdf")
scores = calc.compute(backend="cpp")   # 龙芯无 CUDA，backend 必须用 "cpp"
```

官方 `example/basic_run.py` 里 `backend='cuda'`，龙芯上改成 `backend='cpp'`。

## 九、产物清单

| 产物 | 位置 | 说明 |
|------|------|------|
| 适配分支 | `adapt/sdaa` | 2 处改动（setup.py + backends/__init__.py）|
| 编译产物 | `_roshambo2_cpp.cpython-312-loongarch64-linux-gnu.so` | CPU 后端模块 |
| 依赖 | rdkit（自包含 whl）+ pybind11 + h5py | 见第四节 |