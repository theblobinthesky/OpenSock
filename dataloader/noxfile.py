from __future__ import annotations
import nox
import glob

nox.options.sessions = ["lint", "tests"]

@nox.session
def lint(session: nox.Session) -> None:
    """
    Run the linter.
    """
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox.session
def tests(session: nox.Session) -> None:
    """
    Run the unit and regular tests.
    """
    session.run("pytest", *session.posargs, "-s")


@nox.session(venv_backend="none")
def dev(session: nox.Session) -> None:
    session.run(
        "pip",
        "install",
        "-e.",
       "-Ccmake.define.CMAKE_EXPORT_COMPILE_COMMANDS=1", # for clang-tidy
       "-Ccmake.define.CMAKE_EXPORT_LINK_COMMANDS=1",
        "-Ccmake.define.CMAKE_VERBOSE_MAKEFILE=on",
        "-Ccmake.define.CMAKE_BUILD_TYPE=Debug",
        "-Cbuild-dir=build"
    )

@nox.session(venv_backend="none")
def release(session: nox.Session) -> None:
    session.run(
        "pip",
        "install",
        "-e.",
        "-Ccmake.define.CMAKE_EXPORT_COMPILE_COMMANDS=1", # for clang-tidy
        "-Ccmake.define.CMAKE_EXPORT_LINK_COMMANDS=1",
        "-Ccmake.define.CMAKE_BUILD_TYPE=Release",
        "-Cbuild-dir=build",
    )

@nox.session(venv_backend="none")
def clang_tidy(session: nox.Session) -> None:
    build_dir = "build"
    sources = glob.glob("src/**/*.cpp", recursive=True)

    if not sources:
        session.error("No C++ Source found in src folder.")

    session.run(
        "clang-tidy",
        "-p", build_dir,
        *sources,
        *session.posargs
    )
