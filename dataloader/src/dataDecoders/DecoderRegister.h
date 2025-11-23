#ifndef VERSION_DECODERREGISTER_H
#define VERSION_DECODERREGISTER_H
#include <unordered_map>

#include "dataio.h"

class DecoderRegister {
public:
    static DecoderRegister &getInstance();

    DecoderRegister();

    [[nodiscard]] IDataDecoder *getDataDecoderByExtension(const std::string &ext);

private:
    static DecoderRegister *instance;
    std::unordered_map<std::string, IDataDecoder *> extToDataDecoder;
};

#endif
