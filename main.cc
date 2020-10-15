#include "mbed.h"
#include "platform/mbed_thread.h"

#include "tensorflow/lite/core/api/error_reporter.h"

#include "model_data.h"
#include "audio_recorder.h"
#include "audio_inference.h"
#include "model_settings.h"

namespace {

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 400 * 1024;
__attribute__((aligned(16))) uint8_t tensor_arena[kTensorArenaSize];

tflite::ErrorReporter* error_reporter = nullptr;
AudioInference *engine = nullptr;
AudioRecorder *recorder = nullptr;


// We record for 1.2s, and discard the first 200ms 
// (would be corrupted by board vibrations / human delay)
const int recording_size = kAudioSampleFrequency * 6 / 5; 
int16_t recording[recording_size];
int16_t audio_buffer[kMaxAudioSampleSize];

// Status LEDs
DigitalOut setup_led(LED1);
DigitalOut recording_led(LED2);
DigitalOut computing_led(LED3);

}  // namespace

void setup() {
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    for (int i = 0; i < kMaxAudioSampleSize; i++) {
        audio_buffer[i] = 0;
    }

    static AudioInference static_engine(g_model_data, tensor_arena, kTensorArenaSize, error_reporter);
    engine = &static_engine;

    static AudioRecorder static_recorder(error_reporter);
    recorder = &static_recorder;
}

// This callback will feed audio samples into TensorFlow as requested
TfLiteStatus GetAudioSamples(tflite::ErrorReporter* error_reporter,
                             int start_ms, int duration_ms,
                             int* audio_samples_size, int16_t** audio_samples) {
    const int samples_per_ms = kAudioSampleFrequency / 1000;
    int start_idx = samples_per_ms * start_ms + (kAudioSampleFrequency / 5); // Discard first 200ms
    for (int i = 0; i < samples_per_ms * duration_ms; i++) {
        audio_buffer[i] = recording[start_idx + i];
    }
    *audio_samples_size = kMaxAudioSampleSize;
    *audio_samples = audio_buffer;
    return kTfLiteOk;
}

void infer() {
    int8_t *output = engine->invoke();
    if (output == nullptr) {
        return;
    }

    // Print out all scores and find the maximum-scoring label
    int current_top_index = 0;
    int32_t current_top_score = 0;
    for (int i = 0; i < kCategoryCount; ++i) {
        int8_t score = output[i];
        error_reporter->Report("%s: %d", kCategoryLabels[i], score);
        if (score > current_top_score) {
            current_top_score = score;
            current_top_index = i;
        }
    }

    const char* current_top_label = kCategoryLabels[current_top_index];
    error_reporter->Report("Heard: %s", current_top_label);
}


int main() {
    setup_led = true;
    setup();
    setup_led = false;

    recording_led = true;
    recorder->record(recording, recording_size);
    recording_led = false;

    computing_led = true;
    infer();
    computing_led = false;

    // Busy loop until reset
    while(true) {}

    return 0;
}
