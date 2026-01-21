#ifndef ODI_ERROR_H
#define ODI_ERROR_H

#include "odi_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes */
typedef enum odi_error {
    ODI_SUCCESS = 0,

    /* General errors (1-99) */
    ODI_ERROR_UNKNOWN = 1,
    ODI_ERROR_INVALID_ARGUMENT = 2,
    ODI_ERROR_OUT_OF_MEMORY = 3,
    ODI_ERROR_NOT_IMPLEMENTED = 4,
    ODI_ERROR_INVALID_STATE = 5,
    ODI_ERROR_OPERATION_FAILED = 6,

    /* File/IO errors (100-199) */
    ODI_ERROR_FILE_NOT_FOUND = 100,
    ODI_ERROR_FILE_READ = 101,
    ODI_ERROR_FILE_WRITE = 102,
    ODI_ERROR_FILE_INVALID = 103,
    ODI_ERROR_MMAP_FAILED = 104,

    /* Model errors (200-299) */
    ODI_ERROR_MODEL_INVALID = 200,
    ODI_ERROR_MODEL_UNSUPPORTED_ARCH = 201,
    ODI_ERROR_MODEL_UNSUPPORTED_DTYPE = 202,
    ODI_ERROR_MODEL_CORRUPTED = 203,
    ODI_ERROR_MODEL_VERSION = 204,
    ODI_ERROR_TENSOR_NOT_FOUND = 205,

    /* Backend errors (300-399) */
    ODI_ERROR_BACKEND_UNAVAILABLE = 300,
    ODI_ERROR_BACKEND_INIT_FAILED = 301,
    ODI_ERROR_BACKEND_NOT_SUPPORTED = 302,
    ODI_ERROR_GPU_OUT_OF_MEMORY = 303,

    /* Inference errors (400-499) */
    ODI_ERROR_CONTEXT_TOO_LONG = 400,
    ODI_ERROR_TOKENIZATION_FAILED = 401,
    ODI_ERROR_INFERENCE_FAILED = 402,
    ODI_ERROR_CANCELLED = 403
} odi_error_t;

/* Get human-readable error string */
ODI_API const char* odi_error_string(odi_error_t error);

/* Thread-local last error */
ODI_API odi_error_t odi_get_last_error(void);
ODI_API const char* odi_get_last_error_message(void);

/* Set error (internal use) */
ODI_API void odi_set_error(odi_error_t error, const char* message);
ODI_API void odi_clear_error(void);

#ifdef __cplusplus
}
#endif

#endif /* ODI_ERROR_H */
