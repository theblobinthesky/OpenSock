#ifndef VERSION_FILESYSTEMDATASOURCE_H
#define VERSION_FILESYSTEMDATASOURCE_H

#include "dataio.h"

struct SubdirToDictname {
    std::string subdir;
    std::string dictname;

    SubdirToDictname(std::string subdir, std::string dictname);
};

class FlatDataSource final : public IDataSource {
public:
    explicit FlatDataSource(std::string _rootDirectory, std::vector<SubdirToDictname> _subdirsToDictNames);

    explicit FlatDataSource(std::string _rootDirectory,
                            std::vector<ItemKey> _itemKeys,
                            std::vector<std::vector<std::string> > _entries);

    std::vector<ItemKey> getItemKeys() override;

    std::vector<std::vector<std::string> > getEntries() override;

    CpuAllocation loadItemSliceIntoContigousBatch(BumpAllocator<uint8_t *> alloc,
                                                  const std::vector<std::vector<std::string> > &batchPaths,
                                                  size_t itemKeysIdx, uint32_t bufferSize) override;

    bool preInitDataset(bool forceInvalidation) override;

    void initDataset() override;

    void splitIntoTwoDataSources(size_t aNumEntries, std::shared_ptr<IDataSource> &dataSourceA,
                                std::shared_ptr<IDataSource> &dataSourceB) override;

private:
    std::string rootDirectory;
    std::vector<SubdirToDictname> subdirsToDictNames;
    std::vector<ItemKey> itemKeys;
    std::vector<std::vector<std::string> > entries;
    std::vector<uint8_t> memoryArena;
    bool initRequired;

    void initDatasetFromRootDirectory();

    void verifyDatasetIsConsistent();
};

#endif
