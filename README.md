# **FLUX-REALISM [FLUX Super Realism Image Generator]**

> FLUX.1-Krea-dev + FLUX.1-Krea-dev

A Gradio-based web application for generating hyper-realistic images using FLUX.1-dev with Super Realism LoRA enhancement. This application provides an intuitive interface for creating high-quality, photorealistic images with customizable parameters and styles.

## Features

- **Hyper-realistic image generation** using FLUX.1-dev with Super Realism LoRA
- **Multiple quality presets** (8K, 4K, 2K, and Style Zero)
- **Customizable parameters** including dimensions, guidance scale, and inference steps
- **Batch generation** support for up to 5 images simultaneously
- **ZIP download** option for generated image batches
- **Seed control** with randomization option for reproducible results
- **Negative prompting** support for better content control
- **Real-time generation tracking** with timing information

## Requirements

### System Requirements
- CUDA-compatible GPU (recommended: 8GB+ VRAM)
- Python 3.8+
- CUDA toolkit installed

### Dependencies
```
torch
gradio
diffusers
Pillow
numpy
spaces
```

## Installation

1. Clone or download the application files
2. Install required dependencies:
```bash
pip install torch gradio diffusers Pillow numpy spaces
```
3. Ensure CUDA is properly configured for GPU acceleration

## Usage

### Running the Application

Execute the main script to start the Gradio interface:
```bash
python app.py
```

The application will launch a web interface accessible at `http://localhost:7860`

### Interface Components

#### Main Controls
- **Prompt Input**: Enter your image description
- **Run Button**: Generate images based on current settings

#### Quality Styles
- **3840 x 2160**: 8K hyper-realistic output
- **2560 x 1440**: 4K hyper-realistic output  
- **HD+**: 2K hyper-realistic output
- **Style Zero**: Basic prompt without enhancement

#### Advanced Options
- **Negative Prompt**: Specify elements to exclude from generation
- **Seed Control**: Set specific seed or use randomization
- **Dimensions**: Adjust width and height (512-2048px)
- **Guidance Scale**: Control adherence to prompt (0.1-20.0)
- **Inference Steps**: Quality vs speed trade-off (1-40 steps)
- **Batch Size**: Generate 1-5 images simultaneously
- **ZIP Export**: Download all generated images as archive

### Example Prompts

The application includes several example prompts demonstrating effective usage:

1. Professional portrait photography
2. Environmental character shots
3. Studio lighting setups
4. Artistic portrait compositions

## Model Information

- **Base Model**: black-forest-labs/FLUX.1-dev
- **LoRA Enhancement**: strangerzonehf/Flux-Super-Realism-LoRA
- **Trigger Word**: "Super Realism" (automatically prepended)
- **Precision**: bfloat16 for optimal performance

## Configuration

### Style Presets
The application includes predefined style templates that automatically enhance prompts with quality descriptors:
- Ultra-detailed rendering
- Lifelike textures
- High-resolution output
- Sharp focus and vibrant colors
- Photorealistic results

### Default Settings
- Resolution: 1024x1024px
- Guidance Scale: 3.0
- Inference Steps: 30
- Randomized seed enabled
- Single image generation

## Performance Notes

- GPU acceleration required for practical usage
- Generation time varies based on resolution and step count
- Higher inference steps improve quality but increase processing time
- Batch generation processes images sequentially

## File Management

Generated images are automatically saved with unique UUID filenames in PNG format. The ZIP export feature creates compressed archives containing all generated images with sequential naming.

## Troubleshooting

### Common Issues
- **CUDA out of memory**: Reduce image dimensions or batch size
- **Slow generation**: Decrease inference steps or resolution
- **Model loading errors**: Ensure stable internet connection for initial model download

### Performance Optimization
- Use lower guidance scales (2.0-4.0) for faster generation
- Reduce inference steps for quicker results
- Monitor VRAM usage with multiple image generation

## License

This application uses models and components with their respective licenses:
- FLUX.1-dev model licensing applies
- LoRA weights subject to their repository terms
- Application code available for modification and redistribution

## Support

For technical issues or feature requests, refer to the respective model repositories:
- FLUX.1-dev: black-forest-labs/FLUX.1-dev
- Super Realism LoRA: strangerzonehf/Flux-Super-Realism-LoRA

