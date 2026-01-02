from glob import glob
import pyperclip

filters = [
    "tests/*.py",
    "src/*.h",
    "src/*.cpp",
    "scripts/*",
    ".clang-tidy",
    "CMake*",
    "Makefile",
    "pyproject.toml",
    "pytest.ini",
    "*.md"
]

if __name__ == "__main__":
    file_paths = []
    for filter in filters:
        file_paths.extend(glob(filter))

    compaction = "Code context for this codebase:\n"
    for file_path in file_paths:
        with open(file_path, "r") as file:
            content = file.read()

        compaction += f"> File '{file_path}':\n"
        compaction += content
        compaction += "\n"

    pyperclip.copy(compaction)
    print("Copied context to clipboard")
