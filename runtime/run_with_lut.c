#include <quant_lut.h>

#include "runtime_common.h"

static void apply_lut_nhwc(const uint8_t* image, void* input) {
    int8_t* buffer = (int8_t*)input;
    for (int i = 0; i < INPUT_H * INPUT_W; ++i) {
        for (int channel = 0; channel < INPUT_C; ++channel) {
            int index = i * INPUT_C + channel;
            buffer[index] = quant_lut[channel][image[index]];
        }
    }
}

int main(int argc, char** argv) {
    struct run_options options;
    if (parse_options(argc, argv, &options) != 0) {
        return 1;
    }
    return run_model(&options,
                     INPUT_ELEMS * sizeof(int8_t),
                     apply_lut_nhwc,
                     "INT8-input LUT benchmark");
}
