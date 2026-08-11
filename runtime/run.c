#include "runtime_common.h"

static const float k_mean[INPUT_C] = {0.485f, 0.456f, 0.406f};
static const float k_std[INPUT_C] = {0.229f, 0.224f, 0.225f};

static void normalize_rgb_nhwc(const uint8_t* image, void* input) {
    float* buffer = (float*)input;
    for (int i = 0; i < INPUT_H * INPUT_W; ++i) {
        for (int channel = 0; channel < INPUT_C; ++channel) {
            int index = i * INPUT_C + channel;
            float pixel = (float)image[index] / 255.0f;
            buffer[index] = (pixel - k_mean[channel]) / k_std[channel];
        }
    }
}

int main(int argc, char** argv) {
    struct run_options options;
    if (parse_options(argc, argv, &options) != 0) {
        return 1;
    }
    return run_model(&options,
                     INPUT_ELEMS * sizeof(float),
                     normalize_rgb_nhwc,
                     "Float-input benchmark");
}
