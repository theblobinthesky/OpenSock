import subprocess
import os
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor
from functools import partial

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


def run_command(command_list, cwd=None):
    try:
        result = subprocess.run(
            command_list, capture_output=True, text=True, check=False, cwd=cwd
        )
        if result.returncode != 0:
            print_error("Command failed!")
            print(f"{YELLOW}{'─' * 50}{RESET}")
            if result.stdout.strip():
                print_info("STDOUT:")
                print(result.stdout)
            if result.stderr.strip():
                print_error("STDERR:")
                print(result.stderr)
            print(f"{YELLOW}{'─' * 50}{RESET}")
            return None
        return result.stdout
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


def run_single_clang_tidy(compile_db_path: str, file: str) -> bool:
    return run_command(["clang-tidy", "-p", compile_db_path, "--warnings-as-errors=*", file]) is not None


def run_clang_tidy(files: List[str]) -> bool:
    print_info("→ Running Clang-Tidy...")

    compile_db_path = "build/precommit"
    compile_db_file = os.path.join(compile_db_path, "compile_commands.json")

    if os.path.exists(compile_db_file):
        print_info(f"Using compile_commands.json from {compile_db_path}.")
    else:
        print_error(f"No compile_commands.json found in {compile_db_path}.")
        return False

    with ThreadPoolExecutor(max_workers=24) as executor:
        results = list(
            executor.map(partial(run_single_clang_tidy, compile_db_file), files)
        )
        return all(results)


def run_selected_tests_with_san(_: List[str]) -> bool:
    print_info("→ Building with Sanitizers...")
    if not run_command([f"make", "install-debug-asan"]):
        return False

    print_info("→ Running tests with AddressSanitizer...")
    if not run_command([f"scripts/run_tests_with_asan.sh"]):
        return False
    return True

def run_other_tests(_: List[str]) -> bool:
    pass # TODO 

def main():
    print(f"{BOLD}{'═' * 50}{RESET}")
    print(f"{BOLD}{BLUE}  Pre-commit Checks{RESET}")
    print(f"{BOLD}{'═' * 50}{RESET}")

    checks = []
    if "SKIP_COMPILE" not in os.environ:
        checks.append(("Compile (Debug)", run_precommit_compile))
    if "SKIP_CPPCHECK" not in os.environ:
        checks.append(("CppCheck", run_cpp_check))
    if "SKIP_CLANGTIDY" not in os.environ:
        checks.append(("Clang-Tidy", run_clang_tidy))
    if "SKIP_ASAN_USAN" not in os.environ:
        checks.append(("Selected Tests (ASan, USan)", run_selected_tests_with_san))
    if "SKIP_OTHER_TESTS" not in os.environ:
        checks.append(("Other Tests", run_other_tests))

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
    print_success(f"Succeeded checks: {', '.join(succeededChecks) if succeededChecks else 'none'}")
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
