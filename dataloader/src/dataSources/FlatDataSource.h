#ifndef VERSION_FILESYSTEMDATASOURCE_H
#define VERSION_FILESYSTEMDATASOURCE_H

#include "dataio.h"

class FlatDataSource final : public IDataSource {
public:
    explicit FlatDataSource(std::string _rootDirectory);

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

    IDataSource *splitIntoTwoDatasetsAB(size_t aNumEntries) override;

private:
    std::string rootDirectory;
    std::vector<ItemKey> itemKeys;
    std::vector<std::vector<std::string> > entries;
    std::vector<uint8_t> memoryArena;
    bool initRequired;

    void initDatasetFromRootDirectory();

    void verifyDatasetIsConsistent();
};

#endif
