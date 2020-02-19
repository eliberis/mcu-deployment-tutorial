#pragma once

#include "mbed.h"
#include "mbed-os/targets/TARGET_STM/TARGET_STM32F7/device/stm32f7xx_hal_i2s.h"

int audio_init();
void audio_record(int16_t *output, int32_t length);
