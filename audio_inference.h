#pragma once

#include "tensorflow/lite/micro/examples/micro_speech/audio_provider.h"
#include "tensorflow/lite/micro/examples/micro_speech/feature_provider.h"
#include "model_settings.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

class AudioInference {
private:
	const tflite::Model* model = nullptr;
	tflite::MicroInterpreter* interpreter = nullptr;
	TfLiteTensor* model_input = nullptr;
	FeatureProvider* feature_provider = nullptr;
	tflite::ErrorReporter* error_reporter = nullptr;

public:
	AudioInference(const unsigned char* model_data, 
                   uint8_t* tensor_arena, const int kTensorArenaSize, 
                   tflite::ErrorReporter* error_reporter);
	uint8_t* invoke();
};
