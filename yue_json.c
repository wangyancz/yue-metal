#include "yue_json.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct yw_json yw_json;

struct yw_json {
    yw_json_kind kind;
    /* NUMBER */
    double number;
    /* STRING (owned), BOOL (0/1) */
    char *string;
    int boolean;
    /* ARRAY / OBJECT */
    char **keys; /* OBJECT only, owned */
    yw_json **items;
    size_t count;
    size_t capacity;
};

typedef struct {
    const char *cursor;
    char *error;
    size_t error_size;
    int depth; /* recursion guard for parse_value */
} yw_json_parser;

/* serve feeds 64 KB attacker-controlled lines straight into this parser:
 * unbounded {/[ nesting recursed once per level and could smash the
 * stack. Any real config/request nests a handful of levels. */
#define QW_JSON_MAX_DEPTH 128

static yw_json *parse_value(yw_json_parser *parser);

static void fail(yw_json_parser *parser, const char *message) {
    if (parser->error && parser->error_size > 0) {
        snprintf(parser->error, parser->error_size, "%s", message);
    }
}

static void skip_space(yw_json_parser *parser) {
    while (*parser->cursor == ' ' || *parser->cursor == '\t' ||
           *parser->cursor == '\n' || *parser->cursor == '\r') {
        parser->cursor++;
    }
}

static yw_json *value_new(yw_json_kind kind) {
    yw_json *value = calloc(1, sizeof(yw_json));
    if (value) value->kind = kind;
    return value;
}

void yw_json_free(yw_json *value) {
    if (!value) return;
    free(value->string);
    for (size_t i = 0; i < value->count; i++) {
        if (value->keys) free(value->keys[i]);
        yw_json_free(value->items[i]);
    }
    free(value->keys);
    free(value->items);
    free(value);
}

static int append_item(yw_json *value, char *key, yw_json *item) {
    if (value->count == value->capacity) {
        size_t capacity = value->capacity < 8 ? 8 : value->capacity * 2;
        yw_json **items = realloc(value->items, sizeof(yw_json *) * capacity);
        if (!items) return 0;
        value->items = items;
        char **keys = realloc(value->keys, sizeof(char *) * capacity);
        if (!keys) return 0;
        value->keys = keys;
        value->capacity = capacity;
    }
    value->keys[value->count] = key;
    value->items[value->count] = item;
    value->count++;
    return 1;
}

static int parse_hex4(const char *text, unsigned *out) {
    unsigned value = 0;
    for (int i = 0; i < 4; i++) {
        char c = text[i];
        value <<= 4;
        if (c >= '0' && c <= '9') value |= (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') value |= (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') value |= (unsigned)(c - 'A' + 10);
        else return 0;
    }
    *out = value;
    return 1;
}

static size_t utf8_encode(unsigned code, char *out) {
    if (code < 0x80) {
        out[0] = (char)code;
        return 1;
    }
    if (code < 0x800) {
        out[0] = (char)(0xC0 | (code >> 6));
        out[1] = (char)(0x80 | (code & 0x3F));
        return 2;
    }
    if (code < 0x10000) {
        out[0] = (char)(0xE0 | (code >> 12));
        out[1] = (char)(0x80 | ((code >> 6) & 0x3F));
        out[2] = (char)(0x80 | (code & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (code >> 18));
    out[1] = (char)(0x80 | ((code >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((code >> 6) & 0x3F));
    out[3] = (char)(0x80 | (code & 0x3F));
    return 4;
}

/* Parses a string body after the opening quote; cursor ends past the
 * closing quote. Returns an owned UTF-8 buffer or NULL. */
static char *parse_string(yw_json_parser *parser) {
    const char *start = ++parser->cursor; /* past the opening quote */
    const char *scan = start;
    while (*scan != '"') {
        if (*scan == '\0') {
            fail(parser, "unterminated string");
            return NULL;
        }
        if (*scan == '\\') {
            scan++;
            if (*scan == '\0') {
                fail(parser, "unterminated escape");
                return NULL;
            }
        }
        scan++;
    }

    /* Fast path: no escapes, slice directly. */
    if (memchr(start, '\\', (size_t)(scan - start)) == NULL) {
        size_t length = (size_t)(scan - start);
        char *out = malloc(length + 1);
        if (!out) {
            fail(parser, "out of memory");
            return NULL;
        }
        memcpy(out, start, length);
        out[length] = '\0';
        parser->cursor = scan + 1;
        return out;
    }

    size_t capacity = (size_t)(scan - start) + 16;
    char *out = malloc(capacity);
    if (!out) {
        fail(parser, "out of memory");
        return NULL;
    }
    size_t length = 0;
    const char *p = start;
    while (p < scan) {
        if (*p != '\\') {
            out[length++] = *p++;
            continue;
        }
        p++;
        char c = *p++;
        switch (c) {
            case '"': out[length++] = '"'; break;
            case '\\': out[length++] = '\\'; break;
            case '/': out[length++] = '/'; break;
            case 'b': out[length++] = '\b'; break;
            case 'f': out[length++] = '\f'; break;
            case 'n': out[length++] = '\n'; break;
            case 'r': out[length++] = '\r'; break;
            case 't': out[length++] = '\t'; break;
            case 'u': {
                unsigned code = 0;
                if (!parse_hex4(p, &code)) {
                    free(out);
                    fail(parser, "bad \\u escape");
                    return NULL;
                }
                p += 4;
                if (code == 0) {
                    /* An embedded NUL would truncate every downstream
                     * strlen-based consumer. */
                    free(out);
                    fail(parser, "\\u0000 is not allowed in strings");
                    return NULL;
                }
                if (code >= 0xD800 && code <= 0xDBFF && p[0] == '\\' &&
                    p[1] == 'u') {
                    unsigned low = 0;
                    if (parse_hex4(p + 2, &low) && low >= 0xDC00 &&
                        low <= 0xDFFF) {
                        code = 0x10000 + ((code - 0xD800) << 10) +
                               (low - 0xDC00);
                        p += 6;
                    }
                }
                if (code >= 0xD800 && code <= 0xDFFF) {
                    /* Lone surrogate: encoding it would emit invalid
                     * CESU-8; use the replacement character instead. */
                    code = 0xFFFD;
                }
                length += utf8_encode(code, out + length);
                break;
            }
            default:
                free(out);
                fail(parser, "bad escape");
                return NULL;
        }
    }
    out[length] = '\0';
    parser->cursor = scan + 1;
    return out;
}

static yw_json *parse_value_inner(yw_json_parser *parser) {
    skip_space(parser);
    char c = *parser->cursor;
    if (c == '{') {
        parser->cursor++;
        yw_json *value = value_new(yue_json_OBJECT);
        if (!value) {
            fail(parser, "out of memory");
            return NULL;
        }
        skip_space(parser);
        if (*parser->cursor == '}') {
            parser->cursor++;
            return value;
        }
        while (1) {
            skip_space(parser);
            if (*parser->cursor != '"') {
                yw_json_free(value);
                fail(parser, "expected object key");
                return NULL;
            }
            char *key = parse_string(parser);
            if (!key) {
                yw_json_free(value);
                return NULL;
            }
            skip_space(parser);
            if (*parser->cursor != ':') {
                free(key);
                yw_json_free(value);
                fail(parser, "expected ':'");
                return NULL;
            }
            parser->cursor++;
            yw_json *item = parse_value(parser);
            if (!item) {
                free(key);
                yw_json_free(value);
                return NULL;
            }
            if (!append_item(value, key, item)) {
                free(key);
                yw_json_free(item);
                yw_json_free(value);
                fail(parser, "out of memory");
                return NULL;
            }
            skip_space(parser);
            if (*parser->cursor == ',') {
                parser->cursor++;
                continue;
            }
            if (*parser->cursor == '}') {
                parser->cursor++;
                return value;
            }
            yw_json_free(value);
            fail(parser, "expected ',' or '}'");
            return NULL;
        }
    }
    if (c == '[') {
        parser->cursor++;
        yw_json *value = value_new(yue_json_ARRAY);
        if (!value) {
            fail(parser, "out of memory");
            return NULL;
        }
        skip_space(parser);
        if (*parser->cursor == ']') {
            parser->cursor++;
            return value;
        }
        while (1) {
            yw_json *item = parse_value(parser);
            if (!item) {
                yw_json_free(value);
                return NULL;
            }
            if (!append_item(value, NULL, item)) {
                yw_json_free(item);
                yw_json_free(value);
                fail(parser, "out of memory");
                return NULL;
            }
            skip_space(parser);
            if (*parser->cursor == ',') {
                parser->cursor++;
                continue;
            }
            if (*parser->cursor == ']') {
                parser->cursor++;
                return value;
            }
            yw_json_free(value);
            fail(parser, "expected ',' or ']'");
            return NULL;
        }
    }
    if (c == '"') {
        char *text = parse_string(parser);
        if (!text) return NULL;
        yw_json *value = value_new(yue_json_STRING);
        if (!value) {
            free(text);
            fail(parser, "out of memory");
            return NULL;
        }
        value->string = text;
        return value;
    }
    if (c == 't' && strncmp(parser->cursor, "true", 4) == 0) {
        parser->cursor += 4;
        yw_json *value = value_new(yue_json_BOOL);
        if (value) value->boolean = 1;
        else fail(parser, "out of memory");
        return value;
    }
    if (c == 'f' && strncmp(parser->cursor, "false", 5) == 0) {
        parser->cursor += 5;
        yw_json *value = value_new(yue_json_BOOL);
        if (!value) fail(parser, "out of memory");
        return value;
    }
    if (c == 'n' && strncmp(parser->cursor, "null", 4) == 0) {
        parser->cursor += 4;
        yw_json *value = value_new(yue_json_NULL);
        if (!value) fail(parser, "out of memory");
        return value;
    }
    if (c == '-' || (c >= '0' && c <= '9')) {
        char *end = NULL;
        double number = strtod(parser->cursor, &end);
        if (end == parser->cursor || !isfinite(number)) {
            fail(parser, "bad number");
            return NULL;
        }
        parser->cursor = end;
        yw_json *value = value_new(yue_json_NUMBER);
        if (value) value->number = number;
        else fail(parser, "out of memory");
        return value;
    }
    fail(parser, "unexpected character");
    return NULL;
}

/* Depth guard wrapper: the real body is parse_value_inner. Unbounded
 * {[ nesting used to recurse once per level and could smash the stack on
 * a hostile request line. Real configs nest a handful of levels. */
static yw_json *parse_value_inner(yw_json_parser *parser);

static yw_json *parse_value(yw_json_parser *parser) {
    if (parser->depth >= QW_JSON_MAX_DEPTH) {
        fail(parser, "nesting too deep");
        return NULL;
    }
    parser->depth++;
    yw_json *value = parse_value_inner(parser);
    parser->depth--;
    return value;
}

int yw_json_parse(const char *text, yw_json **out, char *error,
                  size_t error_size) {
    if (error && error_size > 0) error[0] = '\0';
    if (!text || !out) return 0;
    yw_json_parser parser = {text, error, error_size, 0};
    yw_json *value = parse_value(&parser);
    if (!value) return 0;
    skip_space(&parser);
    if (*parser.cursor != '\0') {
        yw_json_free(value);
        if (error && error_size > 0) {
            snprintf(error, error_size, "trailing characters");
        }
        return 0;
    }
    *out = value;
    return 1;
}

yw_json_kind yw_json_type(const yw_json *value) {
    return value ? value->kind : yue_json_NULL;
}

const yw_json *yw_json_get(const yw_json *value, const char *key) {
    if (!value || value->kind != yue_json_OBJECT || !key) return NULL;
    for (size_t i = 0; i < value->count; i++) {
        if (strcmp(value->keys[i], key) == 0) return value->items[i];
    }
    return NULL;
}

const yw_json *yw_json_at(const yw_json *value, size_t index) {
    if (!value || index >= value->count) return NULL;
    return value->items[index];
}

size_t yw_json_count(const yw_json *value) {
    return value ? value->count : 0;
}

int yw_json_bool(const yw_json *value) {
    if (!value) return 0;
    if (value->kind == yue_json_BOOL) return value->boolean;
    if (value->kind == yue_json_NUMBER) return value->number != 0.0;
    if (value->kind == yue_json_STRING) {
        return strcmp(value->string, "true") == 0 ||
               strcmp(value->string, "1") == 0;
    }
    return 0;
}

double yw_json_number(const yw_json *value) {
    if (!value || value->kind != yue_json_NUMBER) return 0.0;
    return value->number;
}

const char *yw_json_string(const yw_json *value) {
    if (!value || value->kind != yue_json_STRING) return "";
    return value->string;
}

double yw_json_get_number(const yw_json *value, const char *key,
                          double fallback) {
    const yw_json *item = yw_json_get(value, key);
    if (!item || item->kind != yue_json_NUMBER) return fallback;
    return item->number;
}

const char *yw_json_get_string(const yw_json *value, const char *key,
                               const char *fallback) {
    const yw_json *item = yw_json_get(value, key);
    if (!item || item->kind != yue_json_STRING) return fallback;
    return item->string;
}

int yw_json_get_bool(const yw_json *value, const char *key, int fallback) {
    const yw_json *item = yw_json_get(value, key);
    /* An explicit null falls back too: yw_json_bool(NULL) is 0, which is
     * not the caller's default. */
    if (!item || item->kind == yue_json_NULL) return fallback;
    return yw_json_bool(item);
}
