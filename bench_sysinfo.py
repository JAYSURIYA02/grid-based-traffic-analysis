"""
bench_sysinfo.py -- hardware/software provenance for benchmark metrics files.

Every value is collected defensively: anything that cannot be determined on
the current platform is recorded as None rather than raising, so a missing
value never aborts a benchmark run.
"""
import os
import platform
import subprocess
import sys


def _cpu_model():
    system = platform.system()
    try:
        if system == "Windows":
            import winreg  # noqa: WPS433 (Windows-only import)
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(name).strip()
        if system == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if system == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
    except Exception:
        pass
    return platform.processor() or None


def _version(module_name):
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def collect_system_info():
    """Return a JSON-serialisable dict describing the machine and software."""
    physical = logical = None
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)
    except Exception:
        pass
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "cpu_model": _cpu_model(),
        "physical_cores": physical,
        "logical_processors": logical,
        "os_cpu_count": os.cpu_count(),
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "opencv_version": _version("cv2"),
        "numpy_version": _version("numpy"),
        "psutil_version": _version("psutil"),
        "btp_env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("BTP_")},
    }


if __name__ == "__main__":
    import json
    print(json.dumps(collect_system_info(), indent=2))
