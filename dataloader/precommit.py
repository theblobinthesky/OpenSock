import subprocess
import os
from pathlib import Path
from typing import List

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


def run_command(command_list):
    try:
        result = subprocess.run(
            command_list, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            print(f"{RED}{BOLD}✗ Command failed!{RESET}")
            print(f"{YELLOW}{'─' * 50}{RESET}")
            if result.stdout.strip():
                print(f"{BLUE}STDOUT:{RESET}\n{result.stdout}")
            if result.stderr.strip():
                print(f"{BLUE}STDERR:{RESET}\n{result.stderr}")
            print(f"{YELLOW}{'─' * 50}{RESET}")
            return None
        return result.stdout
    except FileNotFoundError:
        print(f"{YELLOW}⚠ Error: {command_list[0]} not found. Is it installed?{RESET}")
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
    return files


def run_cpp_check(files: List[str]) -> bool:
    print(f"\n{BLUE}→ Running CppCheck...{RESET}")
    return run_command([
        "cppcheck", 
        "--suppress=missingIncludeSystem",
        "--suppress=unusedFunction",
        "--suppress=missingInclude",
        "--suppress=unusedStructMember",
        "--std=c++20",
        "--enable=all",
        "--error-exitcode=1",
        "--inconclusive"
    ] + files) is not None

def run_clang_tidy(files: List[str]) -> bool:
    print(f"\n{BLUE}→ Running Clang-Tidy...{RESET}")
    return run_command([f"clang-tidy"] + files) is not None

def run_tests_with_asan(_: List[str]) -> bool:
    print(f"\n{BLUE}→ Building with AddressSanitizer...{RESET}")
    if not run_command([f"make", "install-debug-asan"]):
        return False

    print(f"\n{BLUE}→ Running tests with AddressSanitizer...{RESET}")
    if not run_command([f"./scripts/run_tests_with_asan.sh"]):
        return False
    return True


def main():
    print(f"{BOLD}{'═' * 50}{RESET}")
    print(f"{BOLD}{BLUE}  Pre-commit Checks{RESET}")
    print(f"{BOLD}{'═' * 50}{RESET}")

    checks = [
        ("CppCheck", run_cpp_check),
        ("Clang-Tidy", run_clang_tidy),
        ("Tests (ASan)", run_tests_with_asan),
    ]

    files = get_all_h_cpp_files()
    if not files:
        print(f"{RED}✗ Could not gather header and cpp files.{RESET}")
        exit(-1)

    print(f"{BLUE}Found {len(files)} C++ files to check{RESET}")

    failedChecks = []
    for name, check in checks:
        if not check(files):
            failedChecks.append(name)
        else:
            print(f"{GREEN}{BOLD}✓ {name} passed{RESET}")

    print(f"\n{BOLD}{'═' * 50}{RESET}")
    if len(failedChecks) == 0:
        print(f"{GREEN}{BOLD}✓ All checks succeeded!{RESET}")
        print(f"{BOLD}{'═' * 50}{RESET}")
        exit(0)
    else:
        failedChecksStr = ", ".join(failedChecks)
        print(f"{RED}{BOLD}✗ Failed checks: {failedChecksStr}{RESET}")
        print(f"{BOLD}{'═' * 50}{RESET}")
        exit(-1)


if __name__ == "__main__":
    main()
