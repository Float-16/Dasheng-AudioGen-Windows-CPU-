# Dasheng-AudioGen Windows Build (audiogen.cpp)

A Windows-compatible build of [audiogen.cpp](https://github.com/audiohacking/audiogen.cpp) for running the [Dasheng-AudioGen](https://huggingface.co/audiohacking/dasheng-audiogen-gguf) model.

**Disclaimer:** I am not the original author. All credit belongs to the original developers. This repository provides the Windows compilation fixes and pre-compiled binaries.

## Requirements
Download the following model files from [Hugging Face](https://huggingface.co/audiohacking/dasheng-audiogen-gguf):
- `t5_encoder.gguf`
- `dit.gguf` (Must be the F16 version)
- `vocoder.gguf`
- `spiece.model`

## Usage
Download the latest release `.zip` and extract it. Run the following command in Command Prompt:
```cmd
.\dasheng-audiogen.exe "<path_to_models>\t5_encoder.gguf" "<path_to_models>\dit.gguf" "<path_to_models>\vocoder.gguf" "<path_to_models>\spiece.model" --caption "<your_description>" --output "output.wav"