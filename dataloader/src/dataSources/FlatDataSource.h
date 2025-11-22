#ifndef VERSION_FILESYSTEMDATASOURCE_H
#define VERSION_FILESYSTEMDATASOURCE_H

#include "dataio.h"

class FlatDataSource final : public IDataSource {
public:
    explicit FlatDataSource(std::string _rootDirectory);

    std::vector<ItemKey> getItemKeys() override;

    std::vector<std::vector<std::string>> getEntries() override;

    void loadFile(uint8_t *&data, size_t &size) override;

    bool preInitDataset(bool forceInvalidation) override;

    void initDataset() override;

private:
    std::string rootDirectory;
    std::vector<ItemKey> itemKeys;
    std::vector<std::vector<std::string> > entries;
    bool initRequired;
};

#endif
