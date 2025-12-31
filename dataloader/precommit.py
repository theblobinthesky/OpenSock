import subprocess
import os
import json
import platform
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from datetime import datetime
import xml.etree.ElementTree as ET

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"
ERROR_EMOJI = "✗"
WARNING_EMOJI = "⚠"
SUCCESS_EMOJI = "✓"


def print_error(message: str) -> None:
    print(f"{RED}{BOLD}{ERROR_EMOJI} {message}{RESET}")


def print_warning(message: str) -> None:
    print(f"{YELLOW}{WARNING_EMOJI} {message}{RESET}")


def print_success(message: str) -> None:
    print(f"{GREEN}{BOLD}{SUCCESS_EMOJI} {message}{RESET}")


def print_info(message: str) -> None:
    print(f"{BLUE}{message}{RESET}")


def run_command(command_list, cwd=None, env=None):
    result = run_command_capture(command_list, cwd=cwd, env=env)
    if result is None:
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def run_command_capture(command_list, cwd=None, env=None):
    try:
        proc = subprocess.Popen(
            command_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=cwd,
            env=env,
            bufsize=1,
        )

        out_lines = []
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            out_lines.append(line)

        rc = proc.wait()
        out = "".join(out_lines)
        result = subprocess.CompletedProcess(command_list, rc, stdout=out, stderr="")

        if result.returncode != 0:
            print_error("Command failed!")
            print(f"{YELLOW}{'─' * 50}{RESET}")
            print_error("(output streamed above)")
            print(f"{YELLOW}{'─' * 50}{RESET}")

        return result
    except FileNotFoundError:
        print_warning(f"Error: {command_list[0]} not found. Is it installed?")
        return None


def get_all_h_cpp_files() -> List[str]:
    if "CHECK_ALL_FILES" in os.environ:
        files = Path("src").rglob("*")
        files = [str(file) for file in files]
    else:
        res = run_command(["git", "diff", "--name-only"])
        if res is None:
            return []
        files = res.splitlines()

    files = [file for file in files if file.endswith(".cpp") or file.endswith(".h")]

    # If running from subdirectory of git repo, strip subdirectory prefix
    # Git returns paths relative to repo root, but we need them relative to current directory
    if any(file.startswith("dataloader/") for file in files):
        files = [
            file[len("dataloader/") :] if file.startswith("dataloader/") else file
            for file in files
        ]

    print_info(f"Files to check: {files}")
    return files


def run_precommit_compile(_: List[str]) -> bool:
    print_info("→ Running Compile (Debug)...")
    return (
        run_command(
            [
                "uv",
                "pip",
                "install",
                "-e",
                ".",
                "-Cbuild-dir=build/precommit",
                '-Ccmake.args="--preset Debug"',
            ]
        )
        is not None
    )


def run_cpp_check(files: List[str]) -> bool:
    print_info("→ Running CppCheck...")
    return (
        run_command(
            [
                "cppcheck",
                "--suppress=missingIncludeSystem",
                "--suppress=unusedFunction",
                "--suppress=missingInclude",
                "--suppress=unusedStructMember",
                "--suppress=invalidPointerCast",
                "--suppress=useStlAlgorithm",
                "--std=c++20",
                "--language=c++",
                "--enable=all",
                "--error-exitcode=1",
                "--inconclusive",
            ]
            + files
        )
        is not None
    )


def run_single_clang_tidy(file: str) -> bool:
    return run_command(["clang-tidy", "--warnings-as-errors=*", file]) is not None


def run_clang_tidy(files: List[str]) -> bool:
    print_info("→ Running Clang-Tidy...")

    with ThreadPoolExecutor(max_workers=24) as executor:
        results = list(executor.map(run_single_clang_tidy, files))
        return all(results)


def run_selected_tests_with_san(_: List[str]) -> bool:
    print_info("→ Building with Sanitizers...")
    if not run_command([f"make", "install-debug-asan-usan"]):
        return False

    print_info("→ Running tests with AddressSanitizer...")
    if not run_command([f"scripts/run_tests_with_san.sh"]):
        return False
    return True


def parse_junit_report(path: Path):
    tests = []
    try:
        root = ET.fromstring(path.read_text())
    except Exception:
        return {"passed": 0, "failed": 0, "skipped": 0, "errors": 0, "tests": tests}

    for tc in root.iter("testcase"):
        classname = tc.attrib.get("classname", "")
        name = tc.attrib.get("name", "")
        full = f"{classname}::{name}" if classname else name

        status = "passed"
        msg = ""

        failure = tc.find("failure")
        error = tc.find("error")
        skipped = tc.find("skipped")

        if failure is not None:
            status = "failed"
            msg = failure.attrib.get("message", "")
        elif error is not None:
            status = "error"
            msg = error.attrib.get("message", "")
        elif skipped is not None:
            status = "skipped"
            msg = skipped.attrib.get("message", "")

        tests.append({"name": full, "status": status, "message": msg})

    passed = sum(1 for t in tests if t["status"] == "passed")
    failed = sum(1 for t in tests if t["status"] == "failed")
    skipped = sum(1 for t in tests if t["status"] == "skipped")
    errors = sum(1 for t in tests if t["status"] == "error")
    return {
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "errors": errors,
        "tests": tests,
    }


def write_tests_report(
    path: Path,
    normal,
    san,
    normal_cmd: str,
    san_cmd: str,
    normal_out: str,
    san_out: str,
):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("# Test Report")
    lines.append("")
    lines.append(f"Generated: {stamp}")
    lines.append("")

    def add_section(title: str, cmd: str, rep, _out: str):
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"Command: `{cmd}`")
        lines.append("")
        lines.append(f"- passed: {rep['passed']}")
        lines.append(f"- failed: {rep['failed']}")
        lines.append(f"- errors: {rep['errors']}")
        lines.append(f"- skipped: {rep['skipped']}")
        lines.append("")

        def emoji(status: str) -> str:
            if status == "passed":
                return SUCCESS_EMOJI
            if status in ("failed", "error"):
                return ERROR_EMOJI
            return WARNING_EMOJI

        order = {"failed": 0, "error": 0, "skipped": 1, "passed": 2}
        tests = sorted(
            rep.get("tests", []),
            key=lambda t: (order.get(t.get("status", ""), 9), t.get("name", "")),
        )

        lines.append("### Results")
        lines.append("")
        for t in tests:
            status = t.get("status", "")
            name = t.get("name", "")
            msg = t.get("message", "")
            suffix = (
                f": {msg}" if msg and status in ("failed", "error", "skipped") else ""
            )
            lines.append(f"- {emoji(status)} {name}{suffix}")
        lines.append("")

    add_section("Normal", normal_cmd, normal, normal_out)
    add_section("Sanitizers (ASan/USan)", san_cmd, san, san_out)

    path.write_text("\n".join(lines) + "\n")


def run_tests_report(_: List[str]) -> bool:
    print_info("→ Running Tests...")

    normal_xml = Path(".tests_normal.xml")
    san_xml = Path(".tests_san.xml")

    normal_cmd_list = [
        "uv",
        "run",
        "python",
        "-m",
        "pytest",
        "./tests/test_dataloader.py",
        "./tests/test_bindings.py",
        "--benchmark-disable",
        f"--junitxml={normal_xml}",
    ]

    normal_res = run_command_capture(normal_cmd_list)
    normal_out = "" if normal_res is None else (normal_res.stdout + normal_res.stderr)

    print_info("→ Building with Sanitizers...")
    build_env = os.environ.copy()
    build_env.pop("LD_PRELOAD", None)
    build_env.pop("ASAN_OPTIONS", None)
    build_env.pop("UBSAN_OPTIONS", None)
    if not run_command_capture(["make", "install-debug-asan-usan"], env=build_env):
        return False

    try:
        lib_stdcxx = subprocess.check_output(
            ["g++", "-print-file-name=libstdc++.so"], text=True
        ).strip()
        lib_asan = subprocess.check_output(
            ["gcc", "-print-file-name=libasan.so"], text=True
        ).strip()
        lib_asan = str(Path(lib_asan).resolve())
    except Exception as e:
        print_error(f"Failed to resolve sanitizer libs: {e}")
        return False

    env = os.environ.copy()
    env["LD_PRELOAD"] = f"{lib_asan}:{lib_stdcxx}"
    env["ASAN_OPTIONS"] = "detect_leaks=0:log_path=logs/asan_log"

    san_cmd_list = [
        "uv",
        "run",
        "python",
        "-m",
        "pytest",
        "tests/test_meta.py",
        "tests/test_dataset.py",
        "tests/test_augmentations.py",
        "tests/test_compression.py",
        "-xs",
        "--benchmark-disable",
        f"--junitxml={san_xml}",
    ]

    san_res = run_command_capture(san_cmd_list, env=env)
    san_out = "" if san_res is None else (san_res.stdout + san_res.stderr)

    normal_rep = (
        parse_junit_report(normal_xml)
        if normal_xml.exists()
        else {"passed": 0, "failed": 0, "skipped": 0, "errors": 1, "tests": []}
    )
    san_rep = (
        parse_junit_report(san_xml)
        if san_xml.exists()
        else {"passed": 0, "failed": 0, "skipped": 0, "errors": 1, "tests": []}
    )

    if normal_xml.exists():
        normal_xml.unlink()
    if san_xml.exists():
        san_xml.unlink()

    write_tests_report(
        Path("TESTS.md"),
        normal_rep,
        san_rep,
        "pytest ./tests/test_dataloader.py ./tests/test_bindings.py --benchmark-disable",
        "pytest tests/test_meta.py tests/test_dataset.py tests/test_augmentations.py tests/test_compression.py -xs --benchmark-disable (with ASan/USan)",
        normal_out,
        san_out,
    )
    print_success("Wrote test report to TESTS.md")

    ok = (
        normal_rep["failed"]
        + normal_rep["errors"]
        + san_rep["failed"]
        + san_rep["errors"]
    ) == 0
    ok = ok and (normal_res is not None) and (san_res is not None)
    return ok


def run_other_tests(_: List[str]) -> bool:
    return True


def get_system_info():
    cpu_model = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except Exception:
        cpu_model = None

    mem_total = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_total = line.split(":", 1)[1].strip()
                break
    except Exception:
        mem_total = None

    gpu_info = None
    try:
        gpu_info = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        if not gpu_info:
            gpu_info = None
    except Exception:
        gpu_info = None

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu_model or platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "mem_total": mem_total or "unknown",
        "gpu": gpu_info or "(nvidia-smi not available)",
    }


def run_benchmarks(_: List[str]) -> bool:
    print_info("→ Building (Release)...")
    build_env = os.environ.copy()
    build_env.pop("LD_PRELOAD", None)
    build_env.pop("ASAN_OPTIONS", None)
    build_env.pop("UBSAN_OPTIONS", None)
    if not run_command_capture(["make", "install-release"], env=build_env):
        return False

    print_info("→ Running Benchmarks...")
    bench_json = Path(".benchmark.json")
    out = run_command(
        [
            "uv",
            "run",
            "python",
            "-m",
            "pytest",
            "./tests",
            "--benchmark-only",
            f"--benchmark-json={bench_json}",
        ]
    )
    if out is None:
        return False

    try:
        data = json.loads(bench_json.read_text())
    except Exception as e:
        print_error(f"Failed to read benchmark JSON: {e}")
        return False
    finally:
        if bench_json.exists():
            bench_json.unlink()

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cmd = "pytest ./tests --benchmark-only"

    sysinfo = get_system_info()

    lines = []
    lines.append("# Benchmark Results")
    lines.append("")
    lines.append(f"Generated: {stamp}")
    lines.append("")
    lines.append(f"Command: `{cmd}`")
    lines.append("")
    lines.append("## System")
    lines.append("")
    lines.append(f"- platform: {sysinfo['platform']}")
    lines.append(f"- python: {sysinfo['python']}")
    lines.append(f"- cpu: {sysinfo['cpu']}")
    lines.append(f"- cpu_count: {sysinfo['cpu_count']}")
    lines.append(f"- mem_total: {sysinfo['mem_total']}")
    lines.append(f"- gpu: {sysinfo['gpu']}")
    lines.append("")

    benchmarks = data.get("benchmarks", [])
    if not benchmarks:
        lines.append("No benchmarks were collected.")
    else:
        for b in benchmarks:
            name = b.get("name", "(unknown)")
            group = b.get("group", "")
            stats = b.get("stats", {})
            extra = b.get("extra_info", {})

            title = f"## {group} / {name}" if group else f"## {name}"
            lines.append(title)
            lines.append("")
            if stats:
                mean = stats.get("mean")
                rounds = stats.get("rounds")
                iterations = stats.get("iterations")
                lines.append(f"- mean: {mean}")
                lines.append(f"- rounds: {rounds}")
                lines.append(f"- iterations: {iterations}")
            if extra:
                lines.append("- extra:")
                for k in sorted(extra.keys()):
                    lines.append(f"  - {k}: {extra[k]}")
            lines.append("")

        lines.append("## Raw Output")
        lines.append("")
        lines.append("```")
        lines.append(out.strip())
        lines.append("```")

    Path("BENCHMARKS.md").write_text("\n".join(lines) + "\n")
    print_success("Wrote benchmark output to BENCHMARKS.md")
    return True


def main():
    print(f"{BOLD}{'═' * 50}{RESET}")
    print(f"{BOLD}{BLUE}  Pre-commit Checks{RESET}")
    print(f"{BOLD}{'═' * 50}{RESET}")

    checks = []
    if "SKIP_COMPILE" not in os.environ:
        checks.append(("Compile (Debug)", run_precommit_compile))
    if "SKIP_CPPCHECK" not in os.environ:
        checks.append(("CppCheck", run_cpp_check))
    # Clang-tidy is buggy at the moment. Keep it disabled for now.
    # if "SKIP_CLANGTIDY" not in os.environ:
    #     checks.append(("Clang-Tidy", run_clang_tidy))
    if "SKIP_TEST_REPORT" not in os.environ:
        checks.append(("Tests", run_tests_report))
    if "SKIP_BENCHMARKS" not in os.environ:
        checks.append(("Benchmarks", run_benchmarks))

    files = get_all_h_cpp_files()
    if not files:
        print_error("Could not gather header and cpp files.")
        exit(1)

    print_info(f"Found {len(files)} C++ files to check.")

    succeededChecks = []
    failedChecks = []
    for name, check in checks:
        if check(files):
            succeededChecks.append(name)
            print_success(f"{name} passed.")
        else:
            failedChecks.append(name)

    print(f"\n{BOLD}{'═' * 50}{RESET}")
    print_success(
        f"Succeeded checks: {', '.join(succeededChecks) if succeededChecks else 'none'}"
    )
    print_error(f"Failed checks: {', '.join(failedChecks) if failedChecks else 'none'}")
    if len(failedChecks) == 0:
        print_success("All checks succeeded!")
        print(f"{BOLD}{'═' * 50}{RESET}")
        exit(0)
    else:
        print_error("Some checks failed!")
        print(f"{BOLD}{'═' * 50}{RESET}")
        exit(1)


if __name__ == "__main__":
    main()
