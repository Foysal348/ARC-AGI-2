"""Trusted launcher: run from util-linux unshare -Urnmpf with private mounts.

The launcher only binds the host's Python runtime read-only. Kaggle inputs,
notebook, model artifacts and outside writable directories are NOT mounted.
"""
import os
import sys
import shutil
import subprocess


def check(cmd):
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=12)


def main():
    rootfs = sys.argv[1]
    # Prevent any mount propagation into the parent mount namespace.
    check(['mount', '--make-rprivate', '/'])
    # Host imports must be read-only. None of the notebook/data directories are shared.
    roots = ['/usr', '/lib', '/lib64', '/opt/conda', '/opt/venv',
             sys.prefix, sys.base_prefix, os.path.dirname(sys.executable)]
    for host_path in dict.fromkeys(roots):
        if not os.path.exists(host_path):
            continue
        inner_path = os.path.join(rootfs, host_path.lstrip('/'))
        if os.path.islink(host_path):
            os.makedirs(os.path.dirname(inner_path), exist_ok=True)
            if not os.path.lexists(inner_path):
                os.symlink(os.readlink(host_path), inner_path)
            continue
        os.makedirs(inner_path, exist_ok=True)
        check(['mount', '--bind', host_path, inner_path])
        check(['mount', '-o', 'remount,bind,ro', inner_path])
    # Dynamic linker configuration, not general host /etc.
    etcdir = os.path.join(rootfs, 'etc')
    os.makedirs(etcdir, exist_ok=True)
    if os.path.exists('/etc/ld.so.cache'):
        shutil.copyfile('/etc/ld.so.cache', os.path.join(etcdir, 'ld.so.cache'))
    # In this namespace /run, /tmp and /dev are private to each candidate.
    os.makedirs(os.path.join(rootfs, 'dev'), exist_ok=True)
    with open(os.path.join(rootfs, 'dev', 'null'), 'wb'):
        pass
    os.chroot(rootfs)
    os.chdir('/run')
    executable = os.path.realpath(sys.executable)
    if not os.path.exists(executable):
        # In a few environments sys.executable points through a symlink.
        executable = sys.executable
    os.execv(executable, [executable, '-B', '-I', '/run/poetiq_secure_child.py'])


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        import traceback;sys.stderr.write('E10_SANDBOX_BOOTSTRAP_ERROR: '+traceback.format_exc()+'\n')
        sys.exit(111)
