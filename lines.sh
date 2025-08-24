echo "#'src/*.py': $(find src -name *.py | xargs cat | wc -l)"
echo "#'dataloader/*.cpp': $(find dataloader/src -name *.cpp | xargs cat | wc -l)"
echo "#'dataloader/*.h': $(find dataloader/src -name *.h | xargs cat | wc -l)"
echo "#'dataloader/*.py': $(find dataloader/src -name *.py | xargs cat | wc -l)"
