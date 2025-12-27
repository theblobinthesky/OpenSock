echo "#'tests/*.py': $(find tests -name "*.py" | xargs cat | wc -l)"
echo "#'src/*.cpp': $(find src -name "*.cpp" | xargs cat | wc -l)"
echo "#'src/*.h': $(find src -name "*.h" | xargs cat | wc -l)"
echo "#'src/*.py': $(find src -name "*.py" | xargs cat | wc -l)"
