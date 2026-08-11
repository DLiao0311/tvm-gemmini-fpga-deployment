#ifndef TVM_GEMMINI_RUNTIME_COMMON_H_
#define TVM_GEMMINI_RUNTIME_COMMON_H_

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define INPUT_H 512
#define INPUT_W 512
#define INPUT_C 3
#define OUTPUT_H 128
#define OUTPUT_W 128
#define OUTPUT_C 1

#define INPUT_ELEMS (INPUT_H * INPUT_W * INPUT_C)
#define OUTPUT_ELEMS (OUTPUT_H * OUTPUT_W * OUTPUT_C)
#define OUTPUT_DIVISOR 1000.0f

struct tvmgen_default_inputs {
    void* input;
};

struct tvmgen_default_outputs {
    void* output;
};

int32_t tvmgen_default_run(struct tvmgen_default_inputs* inputs,
                           struct tvmgen_default_outputs* outputs);

typedef void (*preprocess_fn)(const uint8_t* image, void* input);

struct run_options {
    const char* image_path;
    int warmup;
    int iterations;
};

static void print_usage(const char* program) {
    fprintf(stderr,
            "usage: %s <input-image> [--warmup N] [--iterations N]\n",
            program);
}

static int parse_nonnegative_int(const char* text, int allow_zero, int* value) {
    char* end = NULL;
    errno = 0;
    long parsed = strtol(text, &end, 10);
    if (errno != 0 || *text == '\0' || *end != '\0' ||
        parsed < (allow_zero ? 0 : 1) || parsed > INT32_MAX) {
        return -1;
    }
    *value = (int)parsed;
    return 0;
}

static int parse_options(int argc, char** argv, struct run_options* options) {
    if (argc < 2) {
        print_usage(argv[0]);
        return -1;
    }

    options->image_path = argv[1];
    options->warmup = 0;
    options->iterations = 1;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            if (parse_nonnegative_int(argv[++i], 1, &options->warmup) != 0) {
                fprintf(stderr, "invalid warmup count: %s\n", argv[i]);
                return -1;
            }
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            if (parse_nonnegative_int(argv[++i], 0, &options->iterations) != 0) {
                fprintf(stderr, "invalid iteration count: %s\n", argv[i]);
                return -1;
            }
        } else {
            fprintf(stderr, "unknown or incomplete option: %s\n", argv[i]);
            print_usage(argv[0]);
            return -1;
        }
    }
    return 0;
}

static uint8_t* resize_rgb_bilinear(const uint8_t* source,
                                    int source_width,
                                    int source_height) {
    size_t output_bytes = (size_t)INPUT_W * INPUT_H * INPUT_C;
    uint8_t* resized = (uint8_t*)malloc(output_bytes);
    if (resized == NULL) {
        return NULL;
    }

    const float scale_x = (float)source_width / INPUT_W;
    const float scale_y = (float)source_height / INPUT_H;
    for (int y = 0; y < INPUT_H; ++y) {
        float source_y = ((float)y + 0.5f) * scale_y - 0.5f;
        int y0 = (int)source_y;
        float y_weight = source_y - y0;
        if (source_y <= 0.0f) {
            y0 = 0;
            y_weight = 0.0f;
        }
        int y1 = y0 + 1;
        if (y1 >= source_height) {
            y1 = source_height - 1;
        }

        for (int x = 0; x < INPUT_W; ++x) {
            float source_x = ((float)x + 0.5f) * scale_x - 0.5f;
            int x0 = (int)source_x;
            float x_weight = source_x - x0;
            if (source_x <= 0.0f) {
                x0 = 0;
                x_weight = 0.0f;
            }
            int x1 = x0 + 1;
            if (x1 >= source_width) {
                x1 = source_width - 1;
            }

            for (int channel = 0; channel < INPUT_C; ++channel) {
                float top_left = source[(y0 * source_width + x0) * INPUT_C + channel];
                float top_right = source[(y0 * source_width + x1) * INPUT_C + channel];
                float bottom_left = source[(y1 * source_width + x0) * INPUT_C + channel];
                float bottom_right = source[(y1 * source_width + x1) * INPUT_C + channel];
                float top = top_left + (top_right - top_left) * x_weight;
                float bottom = bottom_left + (bottom_right - bottom_left) * x_weight;
                float value = top + (bottom - top) * y_weight;
                resized[(y * INPUT_W + x) * INPUT_C + channel] =
                    (uint8_t)(value + 0.5f);
            }
        }
    }
    return resized;
}

static uint8_t* load_rgb_512(const char* path) {
    int width = 0;
    int height = 0;
    int channels = 0;
    uint8_t* image = stbi_load(path, &width, &height, &channels, INPUT_C);
    if (image == NULL) {
        fprintf(stderr, "failed to load %s: %s\n", path, stbi_failure_reason());
        return NULL;
    }
    if (width == INPUT_W && height == INPUT_H) {
        return image;
    }

    uint8_t* resized = resize_rgb_bilinear(image, width, height);
    stbi_image_free(image);
    if (resized == NULL) {
        fprintf(stderr, "failed to resize %s to %dx%d\n", path, INPUT_W, INPUT_H);
        return NULL;
    }
    printf("resized image: %dx%d -> %dx%d\n", width, height, INPUT_W, INPUT_H);
    return resized;
}

static double now_ms(void) {
    struct timespec timestamp;
    clock_gettime(CLOCK_MONOTONIC, &timestamp);
    return timestamp.tv_sec * 1000.0 + timestamp.tv_nsec * 1e-6;
}

static int compare_double(const void* lhs, const void* rhs) {
    double a = *(const double*)lhs;
    double b = *(const double*)rhs;
    return (a > b) - (a < b);
}

static double percentile(const double* sorted, int count, double fraction) {
    int index = (int)((count - 1) * fraction);
    return sorted[index];
}

static int run_model(const struct run_options* options,
                     size_t input_bytes,
                     preprocess_fn preprocess,
                     const char* mode) {
    int status = 1;
    uint8_t* image = NULL;
    void* input = NULL;
    float* output = NULL;
    double* latencies = NULL;

    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        fprintf(stderr, "mlockall failed (ignored): %s\n", strerror(errno));
    }

    image = load_rgb_512(options->image_path);
    if (image == NULL) {
        goto cleanup;
    }

    input = aligned_alloc(64, input_bytes);
    output = (float*)aligned_alloc(64, OUTPUT_ELEMS * sizeof(float));
    latencies = (double*)malloc((size_t)options->iterations * sizeof(double));
    if (input == NULL || output == NULL || latencies == NULL) {
        fprintf(stderr, "memory allocation failed\n");
        goto cleanup;
    }
    memset(output, 0, OUTPUT_ELEMS * sizeof(float));

    struct tvmgen_default_inputs inputs = {.input = input};
    struct tvmgen_default_outputs outputs = {.output = output};

    printf("warming up (%d iters)...\n", options->warmup);
    for (int i = 0; i < options->warmup; ++i) {
        preprocess(image, input);
        if (tvmgen_default_run(&inputs, &outputs) != 0) {
            fprintf(stderr, "model execution failed during warmup\n");
            goto cleanup;
        }
    }

    printf("benchmarking (%d iters)...\n", options->iterations);
    double wall_start = now_ms();
    for (int i = 0; i < options->iterations; ++i) {
        double start = now_ms();
        preprocess(image, input);
        if (tvmgen_default_run(&inputs, &outputs) != 0) {
            fprintf(stderr, "model execution failed at iteration %d\n", i + 1);
            goto cleanup;
        }
        latencies[i] = now_ms() - start;
    }
    double wall_ms = now_ms() - wall_start;

    qsort(latencies, (size_t)options->iterations, sizeof(double), compare_double);
    double latency_sum = 0.0;
    for (int i = 0; i < options->iterations; ++i) {
        latency_sum += latencies[i];
    }

    float density_sum = 0.0f;
    for (int i = 0; i < OUTPUT_ELEMS; ++i) {
        density_sum += output[i] / OUTPUT_DIVISOR;
    }

    printf("\n=== %s ===\n", mode);
    printf("iterations   : %d\n", options->iterations);
    printf("total wall   : %.2f ms\n", wall_ms);
    printf("throughput   : %.2f img/s\n",
           options->iterations / (wall_ms / 1000.0));
    printf("latency mean : %.3f ms\n", latency_sum / options->iterations);
    printf("latency min  : %.3f ms\n", latencies[0]);
    printf("latency p50  : %.3f ms\n",
           percentile(latencies, options->iterations, 0.50));
    printf("latency p95  : %.3f ms\n",
           percentile(latencies, options->iterations, 0.95));
    printf("latency p99  : %.3f ms\n",
           percentile(latencies, options->iterations, 0.99));
    printf("latency max  : %.3f ms\n", latencies[options->iterations - 1]);
    printf("density sum  : %.6f\n", density_sum);
    status = 0;

cleanup:
    stbi_image_free(image);
    free(input);
    free(output);
    free(latencies);
    return status;
}

#endif
