#ifndef TYPES_H
#define TYPES_H

#define null nullptr

#if false
#define debugLog(...) std::printf(__VA_ARGS__)
#else
#define debugLog(...)
#endif

inline bool existsEnvVar(const std::string &name) {
    return std::getenv(name.c_str()) != null;
}

#define INVALID_DS_ENV_VAR "INVALIDATE_DATASET"


#endif
