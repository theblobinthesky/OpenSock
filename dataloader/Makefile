.PHONY: install-debug, install-release, publish, build-debug, build-release

install-debug:
	uv pip install --no-build-isolation --config-settings=editable.rebuild=true -e . -Cbuild-dir=build/debug -Ccmake.args="-DIS_DEBUG=ON --preset Debug"

install-debug-asan-usan:
	uv pip install -e . -Cbuild-dir=build/debug-asan-usan -Ccmake.args="-DIS_DEBUG=ON -DENABLE_ASAN_USAN=ON --preset DebugAsanUsan"

install-release:
	uv pip install -e . -Cbuild-dir=build/release -Ccmake.args="--preset Release"

publish:
	uv pip install . -Ccmake.args="--preset Release"
