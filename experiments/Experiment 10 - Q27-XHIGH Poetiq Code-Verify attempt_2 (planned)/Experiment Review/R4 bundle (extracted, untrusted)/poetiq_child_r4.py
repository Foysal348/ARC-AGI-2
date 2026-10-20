"""Trusted Poetiq runner for Kaggle's namespace-restricted Docker containers.

Not a mount namespace. It provides UID separation and syscall/resource limits;
world-readable host paths are NOT guaranteed unreadable without a mount or LSM.
"""
import sys, os, json, ctypes, errno, resource, traceback
# Import numerical/scientific libraries before syscall filtering.
import numpy as np
try:
    import scipy
except ImportError:
    scipy = None
try:
    import cv2
except ImportError:
    cv2 = None

_RESULT_MARKER = '__E10_PROGRAM_RESULT__'


def drop_privileges_and_limits():
    # Trusted code runs before compiling or executing model-generated Python.
    if os.geteuid() != 0:
        raise RuntimeError('Root is required for irreversible UID/GID drop')
    os.setgroups([])
    os.setgid(65534)
    os.setuid(65534)
    cpu = max(2,min(120,int(os.environ.get('E10_CHILD_CPU','10'))))
    mem = 4096*1024*1024
    resource.setrlimit(resource.RLIMIT_CPU,(cpu,cpu+1))
    resource.setrlimit(resource.RLIMIT_AS,(mem,mem))
    resource.setrlimit(resource.RLIMIT_FSIZE,(1024*1024,1024*1024))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))
    resource.setrlimit(resource.RLIMIT_NPROC,(64,64))


def harden_process():
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:
        raise RuntimeError('PR_SET_NO_NEW_PRIVS failed')
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
    ctx = lib.seccomp_init(0x7fff0000)
    if not ctx:
        raise RuntimeError('seccomp_init failed')
    try:
        # Blocking exec/spawn/network/ptrace/mount remains compatible with
        # numpy/scipy/cv2. Do not block clone(thread) indiscriminately.
        blocked = (
            'execve','execveat','fork','vfork','clone3',
            'socket','socketpair','connect','bind','listen','accept','accept4',
            'mount','umount2','pivot_root','chroot','unshare','setns',
            'ptrace','process_vm_readv','process_vm_writev','open_by_handle_at',
            'reboot','bpf','keyctl','setuid','setgid','setreuid','setregid',
            'setresuid','setresgid',
        )
        for name in blocked:
            nr = lib.seccomp_syscall_resolve_name(name.encode())
            if nr >= 0:
                if lib.seccomp_rule_add(ctx, 0x00050000 | errno.EPERM, nr, 0):
                    raise RuntimeError('seccomp rule failed: ' + name)
        # clone() flags: require CLONE_THREAD when creating a thread.
        # This blocks ordinary child processes without disabling numerical threads.
        clone_nr = lib.seccomp_syscall_resolve_name(b'clone')
        if clone_nr >= 0:
            class ArgCmp(ctypes.Structure):
                _fields_=[('arg',ctypes.c_uint),('op',ctypes.c_uint),
                          ('datum_a',ctypes.c_uint64),('datum_b',ctypes.c_uint64)]
            # seccomp_rule_add_array(ctx, action, syscall, n, &cmp)
            lib.seccomp_rule_add_array.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,
                                                   ctypes.c_uint,ctypes.POINTER(ArgCmp)]
            lib.seccomp_rule_add_array.restype=ctypes.c_int
            # SCMP_CMP_MASKED_EQ=7; deny clone if CLONE_THREAD bit is zero.
            cmp=ArgCmp(0,7,0x10000,0)
            rc=lib.seccomp_rule_add_array(ctx,0x00050000|errno.EPERM,clone_nr,1,ctypes.byref(cmp))
            if rc:
                raise RuntimeError('seccomp clone restriction failed: '+str(rc))
        rc=lib.seccomp_load(ctx)
        if rc:
            raise RuntimeError('seccomp_load failed '+str(rc))
    finally:
        lib.seccomp_release(ctx)


def main():
    raw = json.load(sys.stdin)
    program = raw['code']
    grid = raw['input']
    if not isinstance(program, str) or len(program)>250000:
        raise ValueError('invalid code length')
    drop_privileges_and_limits()
    harden_process()
    namespace = {'__name__': 'poetiq_candidate', '__builtins__': __builtins__, 'np': np}
    try:
        exec(compile(program, '<poetiq_candidate>', 'exec'), namespace)
        result=namespace['transform'](np.array(grid, dtype=int))
        if hasattr(result,'tolist'):
            result=result.tolist()
        payload={'ok':True,'result':result}
    except BaseException as ex:
        payload={'ok':False,'error':(type(ex).__name__+': '+str(ex))[:1200]}
    try:
        print(_RESULT_MARKER + json.dumps(payload,separators=(',',':')),flush=True)
    except BaseException as ex:
        print(_RESULT_MARKER + json.dumps({'ok':False,'error':'JSON output error: '+str(ex)[:500]}),flush=True)

if __name__=='__main__':
    try:
        main()
    except BaseException as ex:
        sys.stderr.write('E10_RUNNER_SETUP_ERROR: '+repr(ex)+'\n')
        sys.exit(110)
