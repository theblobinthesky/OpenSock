#ifndef VERSION_FILESYSTEMDATASOURCE_H
#define VERSION_FILESYSTEMDATASOURCE_H

#include "dataio.h"
#include <unordered_map>

class FlatDataSource final : public IDataSource {
public:
    explicit FlatDataSource(std::string _rootDirectory, std::unordered_map<std::string, std::string> _subdirToDictName);

    explicit FlatDataSource(std::string _rootDirectory,
                            std::vector<ItemKey> _itemKeys,
                            std::vector<std::vector<std::string> > _entries);

    std::vector<ItemKey> getItemKeys() override;

    std::vector<std::vector<std::string> > getEntries() override;

    CpuAllocation loadItemSliceIntoContigousBatch(BumpAllocator<uint8_t *> alloc,
                                                  const std::vector<std::vector<std::string> > &batchPaths,
                                                  size_t itemKeysIdx) override;

    bool preInitDataset(bool forceInvalidation) override;

    void initDataset() override;

    void splitIntoTwoDataSources(size_t aNumEntries, std::shared_ptr<IDataSource> &dataSourceA,
                                std::shared_ptr<IDataSource> &dataSourceB) override;

private:
    std::string rootDirectory;
    std::unordered_map<std::string, std::string> subdirToDictName;
    std::vector<ItemKey> itemKeys;
    std::vector<std::vector<std::string> > entries;
    std::vector<uint8_t> memoryArena;
    bool initRequired;

    void initDatasetFromRootDirectory();

    void verifyDatasetIsConsistent();
};

#endif
