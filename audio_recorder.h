#pragma once

#include "mbed.h"
#include "mbed-os/targets/TARGET_STM/TARGET_STM32F7/device/stm32f7xx_hal_i2s.h"
#include "tensorflow/lite/core/api/error_reporter.h"

class AudioRecorder {
private:
    I2S_HandleTypeDef hi2s1;
public:
    AudioRecorder(tflite::ErrorReporter* error_reporter);
    void record(int16_t *output, int32_t length);
};
