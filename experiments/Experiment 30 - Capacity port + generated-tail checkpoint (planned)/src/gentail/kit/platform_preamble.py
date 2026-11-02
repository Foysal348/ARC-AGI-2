# the NBDF kit's PLATFORM preamble (build_e27_nbdf_cpu_kit.py:268-317), verbatim
import json as _j
PLAT = {}
try:
    import torch
    import vllm.platforms as VP
    from vllm.platforms.interface import DeviceCapability
    from vllm.platforms import cuda as VPC

    class FakeL4(VPC.NonNvmlCudaPlatform):
        @classmethod
        def get_device_capability(cls, device_id=0):
            return DeviceCapability(8, 9)

        @classmethod
        def has_device_capability(cls, capability, device_id=0):
            want = tuple(capability) if isinstance(capability, tuple) else (capability // 10, capability % 10)
            return (8, 9) >= want

        @classmethod
        def get_device_name(cls, device_id=0):
            return "NVIDIA L4"

        @classmethod
        def get_device_total_memory(cls, device_id=0):
            return 23034 * 2 ** 20

        @classmethod
        def is_fully_connected(cls, physical_device_ids):
            return False

    VP._current_platform = FakeL4()
    from vllm.platforms import current_platform as CP
    PLAT.update(ok=bool(CP.is_cuda() and not CP.is_cpu()), cls=type(CP).__name__, capability=list(CP.get_device_capability()))
except BaseException as e:
    PLAT.update(ok=False, error=f"{type(e).__name__}: {e}"[:400])


def init_gloo():
    import socket
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.config import VllmConfig as _VC, set_current_vllm_config as _scvc
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    with _scvc(_VC()):
        init_distributed_environment(world_size=1, rank=0, distributed_init_method=f"tcp://127.0.0.1:{port}", local_rank=0, backend="gloo")
        initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, backend="gloo")
