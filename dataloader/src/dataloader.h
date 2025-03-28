#ifndef DATALOADER_H
#define DATALOADER_H
#include <cstddef>
#include <pybind11/pybind11.h>
#include "dataset.h"

#define CHS 3

// TODO: Align everything!
class MemoryArena {
public:
    MemoryArena();
    explicit MemoryArena(size_t _total_size);
    MemoryArena &operator=(MemoryArena &&arena) noexcept;
    MemoryArena(const MemoryArena &arena) = delete;
    MemoryArena(MemoryArena &&arena) noexcept;
    ~MemoryArena();
    void *allocate(size_t size);

private:
    void *data;
    size_t total_size;
    size_t offset;
};

class DataLoader {
public:
    DataLoader(
        Dataset _dataset,
        int _batchSize,
        pybind11::function _createDatasetFunction
    );

    pybind11::dict getNextBatch();

    [[nodiscard]] size_t getNumberOfBatches() const;

private:
    Dataset dataset;
    size_t batchSize;
    size_t numberOfBatches;
    pybind11::function createDatasetFunction;
    MemoryArena memoryArena;
};

#endif //DATALOADER_H
