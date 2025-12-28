.PHONY: install-debug, install-release, publish, build-debug, build-release

install-debug:
	uv pip install --no-build-isolation --config-settings=editable.rebuild=true -e . -Cbuild-dir=build/debug -Ccmake.args="--preset Debug"
	cmake --preset Debug

install-debug-asan:
	uv pip install -e . -Cbuild-dir=build/debug-asan -Ccmake.args="-DENABLE_ASAN=ON --preset Debug"
	cmake --preset Debug

install-release:
	uv pip install -e . -Cbuild-dir=build/release -Ccmake.args="--preset Release"
	cmake --preset Release

publish:
	uv pip install . -Ccmake.args="--preset Release"

build-debug:
	cmake --build build/debug -- -v

build-release:
	cmake --build build/release
