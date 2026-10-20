"""Trusted inner runner inside a private chroot + mount/network/PID namespace.

This is embedded in E10 notebook by build_e10.py, never fetched at Kaggle runtime.
User-provided text runs only AFTER OS limits and seccomp are active.
"""
import os
import sys
import json
import errno
import ctypes
import resource
import traceback

# Import allowed computational libraries before the syscall filter is installed.
import numpy as np
import scipy

_RESULT_MARKER = '__E10_PROGRAM_RESULT__'


def _install_seccomp():
    """Block process creation, process control, network sockets and namespaces.

    Chroot & isolated read-only bind mounts enforce filesystem boundaries.
    The filter prevents model-generated code from undoing those boundaries.
    """
    lib = ctypes.CDLL('libseccomp.so.2', use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_rule_add.restype = ctypes.c_int
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_load.restype = ctypes.c_int
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.restype = None
    SCMP_ACT_ALLOW = 0x7FFF0000
    SCMP_ACT_ERRNO = 0x00050000 | errno.EPERM
    ctx = lib.seccomp_init(SCMP_ACT_ALLOW)
    if not ctx:
        raise RuntimeError('seccomp_init failed')
    try:
        blocked = (
            'execve', 'execveat', 'fork', 'vfork', 'clone', 'clone3',
            'socket', 'socketpair', 'connect', 'bind', 'listen', 'accept', 'accept4',
            'mount', 'umount2', 'pivot_root', 'chroot', 'unshare', 'setns',
            'ptrace', 'process_vm_readv', 'process_vm_writev', 'open_by_handle_at',
            'kill', 'tkill', 'tgkill', 'reboot', 'bpf', 'keyctl',
        )
        for name in blocked:
            nr = lib.seccomp_syscall_resolve_name(name.encode())
            if nr < 0: continue
            rc = lib.seccomp_rule_add(ctx, SCMP_ACT_ERRNO, nr, 0)
            if rc != 0: raise RuntimeError(f'seccomp_rule_add {name}: {rc}')
        rc = lib.seccomp_load(ctx)
        if rc != 0: raise RuntimeError(f'seccomp_load: {rc}')
    finally:
        lib.seccomp_release(ctx)


def _set_limits():
    cpu = max(2, min(120, int(os.environ.get('POETIQ_CHILD_CPU_SEC', '10'))))
    mem_mb = max(256, min(8192, int(os.environ.get('POETIQ_CHILD_MEM_MB', '4096'))))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
    resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    # NPROC does not protect uid=0, but seccomp blocks clone/fork/execve.
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))


def main():
    # The only data accepted from the parent is code and the puzzle's input grid.
    with open('/run/program.py', encoding='utf-8') as f:
        source = f.read()
    input_grid = json.load(sys.stdin)['input']
    _set_limits()

    # Privileges cannot be regained after leaving the trusted launcher.
    libc = ctypes.CDLL(None, use_errno=True)
    PR_SET_NO_NEW_PRIVS = 38
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise RuntimeError('no_new_privs failed')

    _install_seccomp()
    # All imports done by untrusted code must come from the isolated filesystem.
    namespace = {'__name__': 'poetiq_candidate', '__builtins__': __builtins__, 'np': np}
    try:
        exec(compile(source, '/run/program.py', 'exec'), namespace)
        result = namespace['transform'](np.array(input_grid, dtype=int))
        if hasattr(result, 'tolist'):
            result = result.tolist()
        payload = {'ok': True, 'result': result}
    except BaseException as exc:
        payload = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'[:1200]}
    try:
        # Debug prints may precede this trusted final marker.
        print(_RESULT_MARKER + json.dumps(payload, separators=(',', ':')), flush=True)
    except BaseException as exc:
        print(_RESULT_MARKER + json.dumps({'ok': False, 'error': f'JSON output error: {exc}'[:500]}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        sys.stderr.write('E10_SECURE_RUNNER_INIT_ERROR: ' + repr(exc) + '\n')
        sys.exit(110)
