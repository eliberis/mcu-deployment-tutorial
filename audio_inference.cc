#include "audio_inference.h"

AudioInference::AudioInference(const unsigned char* model_data, 
                               uint8_t* tensor_arena, const int kTensorArenaSize, 
                               tflite::ErrorReporter* error_reporter) {
    this->error_reporter = error_reporter;

    this->model = tflite::GetModel(model_data);
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
    this->interpreter = &static_interpreter;

    // Allocate memory from the tensor_arena for the model's tensors.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        error_reporter->Report("AllocateTensors() failed");
        return;
    }

    // Get information about the memory area to use for the model's input.
    this->model_input = interpreter->input(0);
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
    this->feature_provider = &static_feature_provider;
}


uint8_t* AudioInference::invoke() {
	// Fetch the spectrogram.
    int how_many_new_slices = 0;
    TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
            error_reporter, 0, 1000, &how_many_new_slices);
    if (feature_status != kTfLiteOk) {
        error_reporter->Report("Feature generation failed");
        return nullptr;
    }

    // Run the model on the spectrogram input and make sure it succeeds.
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        error_reporter->Report("Invoke failed");
        return nullptr;
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
        return nullptr;
    }

    if (output->type != kTfLiteUInt8) {
        error_reporter->Report(
                "The results for recognition should be uint8 elements, but are %d",
                output->type);
        return nullptr;
    }

    return output->data.uint8;
}
