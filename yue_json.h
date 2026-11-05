#ifndef yue_json_H
#define yue_json_H

#include <stddef.h>

/* Minimal JSON reader for model configuration files (config.json,
 * model_index.json, scheduler_config.json, safetensors index metadata).
 * Objects preserve insertion order so duplicate keys resolve to the first
 * entry, mirroring typical config semantics closely enough for these
 * single-valued files. */

typedef enum {
    yue_json_NULL = 0,
    yue_json_BOOL,
    yue_json_NUMBER,
    yue_json_STRING,
    yue_json_ARRAY,
    yue_json_OBJECT
} yw_json_kind;

typedef struct yw_json yw_json;

/* Parse NUL-terminated text. Returns 1 and fills *out on success,
 * 0 on failure (with *out set to NULL). */
int yw_json_parse(const char *text, yw_json **out, char *error,
                  size_t error_size);
void yw_json_free(yw_json *value);

yw_json_kind yw_json_type(const yw_json *value);
const yw_json *yw_json_get(const yw_json *value, const char *key);
const yw_json *yw_json_at(const yw_json *value, size_t index);
size_t yw_json_count(const yw_json *value);
/* True bool, or number != 0, or string "true"/"1". */
int yw_json_bool(const yw_json *value);
double yw_json_number(const yw_json *value);
/* NUL-terminated UTF-8 payload of a string value (or ""). */
const char *yw_json_string(const yw_json *value);
/* Convenience: object member as double/string with fallback. */
double yw_json_get_number(const yw_json *value, const char *key,
                          double fallback);
const char *yw_json_get_string(const yw_json *value, const char *key,
                               const char *fallback);
int yw_json_get_bool(const yw_json *value, const char *key, int fallback);

#endif
