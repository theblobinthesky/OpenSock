from __future__ import annotations
import nox

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
    session.run("pytest", *session.posargs)


@nox.session(venv_backend="none")
def dev(session: nox.Session) -> None:
    session.run(
        "pip",
        "install",
        "-e.",
#        "-Ccmake.define.CMAKE_EXPORT_COMPILE_COMMANDS=1",
#        "-Ccmake.define.CMAKE_EXPORT_LINK_COMMANDS=1",
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
        "-Ccmake.define.CMAKE_EXPORT_COMPILE_COMMANDS=1",
        "-Ccmake.define.CMAKE_EXPORT_LINK_COMMANDS=1",
        "-Ccmake.define.CMAKE_BUILD_TYPE=Release",
        "-Cbuild-dir=build",
    )