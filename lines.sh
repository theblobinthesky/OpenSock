echo "#'src/*.py': $(find src -name *.py | xargs cat | wc -l)"
echo "#'dataloader/*.cpp': $(find dataloader -name *.cpp | xargs cat | wc -l)"
