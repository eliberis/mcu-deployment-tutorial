#include "audio_recorder.h"

AudioRecorder::AudioRecorder(tflite::ErrorReporter* error_reporter) {
    // Initialise the I2S subsystem

    // Set up clocks
    RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};
    PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_I2S;
    PeriphClkInitStruct.PLLI2S.PLLI2SN = 192;
    PeriphClkInitStruct.PLLI2S.PLLI2SP = RCC_PLLP_DIV2;
    PeriphClkInitStruct.PLLI2S.PLLI2SR = 2;
    PeriphClkInitStruct.PLLI2S.PLLI2SQ = 2;
    PeriphClkInitStruct.PLLI2SDivQ = 1;
    PeriphClkInitStruct.I2sClockSelection = RCC_I2SCLKSOURCE_PLLI2S;
    if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK) {
        error_reporter->Report("HAL_RCCEx_PeriphCLKConfig fail");
    }

    // Enable GPIO port clock
    __SPI1_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOC_CLK_ENABLE();

    pin_function(PA_7, STM_PIN_DATA(STM_MODE_AF_PP, GPIO_NOPULL, GPIO_AF5_SPI3));
    pin_function(PA_5, STM_PIN_DATA(STM_MODE_AF_PP, GPIO_NOPULL, GPIO_AF5_SPI3));
    pin_function(PA_4, STM_PIN_DATA(STM_MODE_AF_PP, GPIO_NOPULL, GPIO_AF5_SPI3));

    // Set up I2S handles
    hi2s1.Instance = SPI1;
    hi2s1.Init.Mode = I2S_MODE_MASTER_RX;
    hi2s1.Init.Standard = I2S_STANDARD_PHILIPS;
    hi2s1.Init.DataFormat = I2S_DATAFORMAT_24B;
    hi2s1.Init.MCLKOutput = I2S_MCLKOUTPUT_ENABLE;
    hi2s1.Init.AudioFreq = I2S_AUDIOFREQ_16K;
    hi2s1.Init.CPOL = I2S_CPOL_LOW;
    hi2s1.Init.ClockSource = I2S_CLOCK_PLL;
    if (HAL_I2S_Init(&hi2s1) != HAL_OK) {
        error_reporter->Report("HAL_I2S_Init fail");
    }
}

void AudioRecorder::record(int16_t *output, int32_t length) {
    uint16_t data_in[4];
    int i = 0;
    while (i < length) {
        HAL_StatusTypeDef result = HAL_I2S_Receive(&hi2s1, data_in, 2, 100);
        if (result == HAL_OK) {
            // (we discarded 2 bits by keeping 16 out of 18 data bits)
            output[i] = ((int16_t)data_in[0]) << 2;
            i++;
        }
    }
}
