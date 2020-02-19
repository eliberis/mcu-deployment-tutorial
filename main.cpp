#include "mbed.h"
#include "platform/mbed_thread.h"

#include "model_data.h"
#include "audio_recorder.h"
#include "tensorflow/lite/micro/examples/micro_speech/audio_provider.h"
#include "tensorflow/lite/micro/examples/micro_speech/command_responder.h"
#include "tensorflow/lite/micro/examples/micro_speech/feature_provider.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace {

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
FeatureProvider* feature_provider = nullptr;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 50 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// We record for 1.2s, and discard the first 200ms 
// (would be corrupted by board vibrations / human delay)
const int recording_size = kAudioSampleFrequency * 6 / 5; 
int16_t recording[recording_size];
int16_t g_dummy_audio_data[kMaxAudioSampleSize];

DigitalOut setup_led(LED1);
DigitalOut recording_led(LED2);
DigitalOut computing_led(LED3);

}  // namespace

void setup() {
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    for (int i = 0; i < kMaxAudioSampleSize; i++) {
        g_dummy_audio_data[i] = 0;
    }

    audio_init();

    // Set up Tensorflow Lite Micro
    model = tflite::GetModel(g_model_data);
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
    micro_op_resolver.AddBuiltin(
            tflite::BuiltinOperator_FULLY_CONNECTED,
            tflite::ops::micro::Register_FULLY_CONNECTED());
    micro_op_resolver.AddBuiltin(
            tflite::BuiltinOperator_SOFTMAX,
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
    static FeatureProvider static_feature_provider(kFeatureElementCount, model_input->data.uint8);
    feature_provider = &static_feature_provider;
}

void infer() {
    // Fetch the spectrogram.
    int how_many_new_slices = 0;
    TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
            error_reporter, 0, 1000, &how_many_new_slices);
    if (feature_status != kTfLiteOk) {
        error_reporter->Report("Feature generation failed");
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

    if ((output->dims->size != 2) ||
            (output->dims->data[0] != 1) ||
            (output->dims->data[1] != kCategoryCount)) {
        error_reporter->Report(
                "The results for recognition should contain %d elements, but there are "
                "%d in an %d-dimensional shape",
                kCategoryCount, output->dims->data[1],
                output->dims->size);
        return;
    }

    if (output->type != kTfLiteUInt8) {
        error_reporter->Report(
                "The results for recognition should be uint8 elements, but are %d",
                output->type);
        return;
    }

    int current_top_index = 0;
    int32_t current_top_score = 0;
    for (int i = 0; i < kCategoryCount; ++i) {
        uint8_t score = output->data.uint8[i];
        error_reporter->Report("%s: %d", kCategoryLabels[i], score);
        if (score > current_top_score) {
            current_top_score = score;
            current_top_index = i;
        }
    }
    const char* current_top_label = kCategoryLabels[current_top_index];
    error_reporter->Report("Heard: %s", current_top_label);
}


TfLiteStatus GetAudioSamples(tflite::ErrorReporter* error_reporter,
                             int start_ms, int duration_ms,
                             int* audio_samples_size, int16_t** audio_samples) {
    int start_idx = 16 * start_ms + (kAudioSampleFrequency / 5);
    for (int i = 0; i < 16 * duration_ms; i++) {
        g_dummy_audio_data[i] = recording[start_idx + i];
    }
    *audio_samples_size = kMaxAudioSampleSize;
    *audio_samples = g_dummy_audio_data;
    return kTfLiteOk;
}

int main() {
    setup_led = true;
    setup();
    setup_led = false;

    recording_led = true;
    audio_record(recording, recording_size);
    recording_led = false;

    computing_led = true;
    infer();
    computing_led = false;

    // Busy loop until reset
    while(true) {}

    return 0;
}
