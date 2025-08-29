#ifndef VERSION_FILESYSTEMDATASOURCE_H
#define VERSION_FILESYSTEMDATASOURCE_H

#include "dataio.h"

class FlatDataSource final : public IDataSource {
public:
    explicit FlatDataSource(std::string _rootDirectory);

    std::vector<ItemKey> getItemKeys() override;

    std::vector<Sample> getSamples() override;

    void loadFile(uint8_t *&data, size_t &size) override;

private:
    std::string rootDirectory;
    std::vector<ItemKey> itemKeys;
    std::vector<Sample> samples;
};


#endif
