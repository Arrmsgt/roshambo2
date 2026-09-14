from .cpp_backend import CppShapeOverlay
try:
    from .cuda_backend import CudaShapeOverlay
except ImportError:
    # 龙芯 loongarch64 适配：无 CUDA 后端（_roshambo2_cuda 未编译）时降级为 None
    CudaShapeOverlay = None