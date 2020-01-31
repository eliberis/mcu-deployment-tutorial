/* mbed Microcontroller Library
 * Copyright (c) 2019 ARM Limited
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mbed.h"
#include "platform/mbed_thread.h"


// Blinking rate in milliseconds
#define BLINKING_RATE_MS                                                    500


Serial pc(USBTX, USBRX);
I2S_HandleTypeDef hi2s1;


static void MX_I2S1_Init(void)
{

  /* USER CODE BEGIN I2S1_Init 0 */

  /* USER CODE END I2S1_Init 0 */

  /* USER CODE BEGIN I2S1_Init 1 */

  /* USER CODE END I2S1_Init 1 */
  hi2s1.Instance = SPI1;
  hi2s1.Init.Mode = I2S_MODE_MASTER_RX;
  hi2s1.Init.Standard = I2S_STANDARD_PHILIPS;
  hi2s1.Init.DataFormat = I2S_DATAFORMAT_24B;
  hi2s1.Init.MCLKOutput = I2S_MCLKOUTPUT_DISABLE; // TODO: ENABLE?
  hi2s1.Init.AudioFreq = I2S_AUDIOFREQ_16K;
  hi2s1.Init.CPOL = I2S_CPOL_LOW;
  hi2s1.Init.FirstBit = I2S_FIRSTBIT_MSB;
  hi2s1.Init.WSInversion = I2S_WS_INVERSION_DISABLE;
  hi2s1.Init.Data24BitAlignment = I2S_DATA_24BIT_ALIGNMENT_RIGHT;
  hi2s1.Init.MasterKeepIOState = I2S_MASTER_KEEP_IO_STATE_DISABLE;
  if (HAL_I2S_Init(&hi2s1) != HAL_OK)
  {
    pc.printf("HAL_I2S_Init error.\r\n");
  }
  /* USER CODE BEGIN I2S1_Init 2 */

  /* USER CODE END I2S1_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOA_CLK_ENABLE();

}


int main()
{
	pc.printf("main() begun\r\n");

	MX_GPIO_Init();
  	MX_I2S1_Init();
    // Initialise the digital pin LEDX as an output
    DigitalOut led(LED2);

    pc.printf("System initialised\r\n");



    while (true) {
        led = !led;
        pc.printf("LED state changed\r\n");
        thread_sleep_for(BLINKING_RATE_MS);

        uint16_t data_in[2];
        HAL_StatusTypeDef result = HAL_I2S_Receive(&hi2s1, data_in, 2, 100);
        //pc.printf("Status %d\r\n", result);
		if (result == HAL_OK) {
			volatile int32_t data_full = (int32_t) data_in[0] << 16 | data_in[1];
			// volatile int16_t data_short = (int16_t) data_in[0];
			pc.printf("%d \r\n", data_full);
		}
    }
}
