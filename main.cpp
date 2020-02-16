/* mbed Microcontroller Library
 * Copyright (c) 2019 ARM Limited
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mbed.h"
#include "platform/mbed_thread.h"
#include "mbed-os/targets/TARGET_STM/TARGET_STM32F7/device/stm32f7xx_hal_i2s.h"

#include "tensorflow/lite/micro/examples/micro_speech/audio_provider.h"
#include "tensorflow/lite/micro/examples/micro_speech/command_responder.h"
#include "tensorflow/lite/micro/examples/micro_speech/feature_provider.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/tiny_conv_micro_features_model_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/recognize_commands.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace {
I2S_HandleTypeDef hi2s1;
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
FeatureProvider* feature_provider = nullptr;
RecognizeCommands* recognizer = nullptr;
int32_t previous_time = 0;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 50 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

}  // namespace


void setup() {
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};
  // PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_I2S;
  // PeriphClkInitStruct.PLLI2S.PLLI2SN = 192;
  // PeriphClkInitStruct.PLLI2S.PLLI2SP = RCC_PLLP_DIV2;
  // PeriphClkInitStruct.PLLI2S.PLLI2SR = 2;
  // PeriphClkInitStruct.PLLI2S.PLLI2SQ = 2;
  // PeriphClkInitStruct.PLLI2SDivQ = 1;
  // PeriphClkInitStruct.I2sClockSelection = RCC_I2SCLKSOURCE_PLLI2S;
  // if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
  // {
  //   error_reporter->Report("HAL_RCCEx_PeriphCLKConfig");
  // }

  //   // Enable GPIO port clock
  // // __HAL_RCC_GPIOA_CLK_ENABLE();
  // __HAL_RCC_GPIOA_CLK_ENABLE();
  // pin_function(PA_7, STM_PIN_DATA(STM_MODE_AF_PP, GPIO_NOPULL, GPIO));
  // pin_mode(PA_7, PullNone);

  
  // pin_function(PA_5, STM_PIN_DATA(STM_MODE_AF_PP, GPIO_NOPULL, GPIO_AF14_USB));
  // __HAL_RCC_GPIOC_CLK_ENABLE();


  // Set up I2S handles
  // hi2s1.Instance = SPI1;
  // hi2s1.Init.Mode = I2S_MODE_MASTER_RX;
  // hi2s1.Init.Standard = I2S_STANDARD_PHILIPS;
  // hi2s1.Init.DataFormat = I2S_DATAFORMAT_24B;
  // hi2s1.Init.MCLKOutput = I2S_MCLKOUTPUT_DISABLE; // TODO: ENABLE?
  // hi2s1.Init.AudioFreq = I2S_AUDIOFREQ_16K;
  // hi2s1.Init.CPOL = I2S_CPOL_LOW;
  // hi2s1.Init.FirstBit = I2S_FIRSTBIT_MSB;
  // hi2s1.Init.WSInversion = I2S_WS_INVERSION_DISABLE;
  // hi2s1.Init.Data24BitAlignment = I2S_DATA_24BIT_ALIGNMENT_RIGHT;
  // hi2s1.Init.MasterKeepIOState = I2S_MASTER_KEEP_IO_STATE_DISABLE;
  // hi2s1.Instance = SPI1;
  // hi2s1.Init.Mode = I2S_MODE_MASTER_RX;
  // hi2s1.Init.Standard = I2S_STANDARD_PHILIPS;
  // hi2s1.Init.DataFormat = I2S_DATAFORMAT_24B;
  // hi2s1.Init.MCLKOutput = I2S_MCLKOUTPUT_ENABLE;
  // hi2s1.Init.AudioFreq = I2S_AUDIOFREQ_16K;
  // hi2s1.Init.CPOL = I2S_CPOL_LOW;
  // hi2s1.Init.ClockSource = I2S_CLOCK_PLL;
  // if (HAL_I2S_Init(&hi2s1) != HAL_OK) {
  //   error_reporter->Report("HAL_I2S_Init error.");
  // }



  // Set up Tensorflow Lite Micro
  model = tflite::GetModel(g_tiny_conv_micro_features_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::ops::micro::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_CONV_2D,
      tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(
          tflite::BuiltinOperator_MAX_POOL_2D,
          tflite::ops::micro::Register_MAX_POOL_2D());
  micro_op_resolver.AddBuiltin(
          tflite::BuiltinOperator_RESHAPE,
          tflite::ops::micro::Register_RESHAPE());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                               tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != kFeatureSliceCount) ||
      (model_input->dims->data[2] != kFeatureSliceSize) ||
      (model_input->type != kTfLiteUInt8)) {
    error_reporter->Report("Bad input tensor parameters in model");
    return;
  }

  // Prepare to access the audio spectrograms from a microphone or other source
  // that will provide the inputs to the neural network.
  // NOLINTNEXTLINE(runtime-global-variables)
  static FeatureProvider static_feature_provider(kFeatureElementCount,
                                                 model_input->data.uint8);
  feature_provider = &static_feature_provider;

  static RecognizeCommands static_recognizer(error_reporter);
  recognizer = &static_recognizer;

  previous_time = 0;
}

// The name of this function is important for Arduino compatibility.
void loop() {
  // Fetch the spectrogram for the current time.
  const int32_t current_time = LatestAudioTimestamp();
  int how_many_new_slices = 0;
  TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
      error_reporter, previous_time, current_time, &how_many_new_slices);
  if (feature_status != kTfLiteOk) {
    error_reporter->Report("Feature generation failed");
    return;
  }
  previous_time = current_time;
  // If no new audio samples have been received since last time, don't bother
  // running the network model.
  if (how_many_new_slices == 0) {
    return;
  }

  // Run the model on the spectrogram input and make sure it succeeds.
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    return;
  }

  // Obtain a pointer to the output tensor
  TfLiteTensor* output = interpreter->output(0);
  // Determine whether a command was recognized based on the output of inference
  const char* found_command = nullptr;
  uint8_t score = 0;
  bool is_new_command = false;
  TfLiteStatus process_status = recognizer->ProcessLatestResults(
      output, current_time, &found_command, &score, &is_new_command);
  if (process_status != kTfLiteOk) {
    error_reporter->Report("RecognizeCommands::ProcessLatestResults() failed");
    return;
  }
  // Do something based on the recognized command. The default implementation
  // just prints to the error console, but you should replace this with your
  // own function for a real application.
  RespondToCommand(error_reporter, current_time, found_command, score,
                   is_new_command);
}

namespace {
int16_t g_dummy_audio_data[kMaxAudioSampleSize];
int32_t g_latest_audio_timestamp = 0;
}  // namespace

TfLiteStatus GetAudioSamples(tflite::ErrorReporter* error_reporter,
                             int start_ms, int duration_ms,
                             int* audio_samples_size, int16_t** audio_samples) {
  for (int i = 0; i < kMaxAudioSampleSize; ++i) {
    g_dummy_audio_data[i] = 0;
  }
  *audio_samples_size = kMaxAudioSampleSize;
  *audio_samples = g_dummy_audio_data;
  return kTfLiteOk;
}

int32_t LatestAudioTimestamp() {
  g_latest_audio_timestamp += 100;
  return g_latest_audio_timestamp;
}

void RespondToCommand(tflite::ErrorReporter* error_reporter,
                      int32_t current_time, const char* found_command,
                      uint8_t score, bool is_new_command) {
  //if (is_new_command) {
    error_reporter->Report("Heard %s (%d) @%dms", found_command, score,
                           current_time);
  //}
}


void SystemClock_Config(void)
{
  // RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  // RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  // RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};

  // /** Macro to configure the PLL multiplication factor 
  // */
  // __HAL_RCC_PLL_PLLM_CONFIG(16);
  // /** Macro to configure the PLL clock source 
  // */
  // __HAL_RCC_PLL_PLLSOURCE_CONFIG(RCC_PLLSOURCE_HSI);
  // /** Configure the main internal regulator output voltage 
  // */
  // __HAL_RCC_PWR_CLK_ENABLE();
  // __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);
  // /** Initializes the CPU, AHB and APB busses clocks 
  // */
  // RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  // RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  // RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  // RCC_OscInitStruct.PLL.PLLState = RCC_PLL_NONE;
  // RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  // RCC_OscInitStruct.PLL.PLLM = 16;
  // if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  // {
  //   error_reporter->Report("HAL_RCC_OscConfig");
  // }
  // /** Initializes the CPU, AHB and APB busses clocks 
  // */
  // RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
  //                             |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  // RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
  // RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  // RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  // RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  // if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_0) != HAL_OK)
  // {
  //   error_reporter->Report("HAL_RCC_ClockConfig");
  // }
  // PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_I2S;
  // PeriphClkInitStruct.PLLI2S.PLLI2SN = 192;
  // PeriphClkInitStruct.PLLI2S.PLLI2SP = RCC_PLLP_DIV2;
  // PeriphClkInitStruct.PLLI2S.PLLI2SR = 2;
  // PeriphClkInitStruct.PLLI2S.PLLI2SQ = 2;
  // PeriphClkInitStruct.PLLI2SDivQ = 1;
  // PeriphClkInitStruct.I2sClockSelection = RCC_I2SCLKSOURCE_PLLI2S;
  // if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
  // {
  //   error_reporter->Report("HAL_RCCEx_PeriphCLKConfig");
  // }
}

int main() {
  //SystemClock_Config();
  setup();

  while (true) {
      loop();

      // uint16_t data_in[2];
      // HAL_StatusTypeDef result = HAL_I2S_Receive(&hi2s1, data_in, 2, 100);
      // error_reporter->Report("Status %d\r\n", result);
      // if (result == HAL_OK) {
      //   volatile int32_t data_full = (int32_t) data_in[0] << 16 | data_in[1];
      //   // volatile int16_t data_short = (int16_t) data_in[0];
      //   error_reporter->Report("%d \r\n", data_full);
      // }
  }
}
