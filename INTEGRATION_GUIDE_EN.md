# DiffV2IR Frontend Integration Guide

## Project Overview

DiffV2IR is a diffusion-model-based visible-light-to-infrared image conversion system that achieves high-quality image conversion through vision-language understanding. The system contains two core modules:
- **Progressive Learning Module (PLM)**: uses multi-stage knowledge learning to achieve infrared conversion from the full spectrum to the target wavelength
- **Vision-Language Understanding Module (VLUM)**: combines language descriptions and segmentation maps to enhance semantic perception

## Environment Setup

### 1. System Requirements
- Python 3.10.15
- A GPU with CUDA 12.8 support (recommended VRAM >= 8GB)
- Linux/Unix system (Ubuntu 20.04+ recommended)
- Git
- At least 20GB of free disk space (for models and data)

### 2. Create a Virtual Environment
```bash
# Create the environment with conda (recommended)
conda create -n DiffV2IR python=3.10.15
conda activate DiffV2IR
```

### 3. Clone the Project Code
```bash
git clone https://github.com/Jian-Zhang-3DV/DiffV2IR.git
cd DiffV2IR
```

### 4. Install Dependencies

```bash
# Install PyTorch 2.8.0 with CUDA 12.8
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# Install all other dependencies (requirements.txt has been updated to the correct versions)
pip install -r requirements.txt
```

### 5. Verify the Environment Setup

```bash
# Verify the Python version
python --version  # should show Python 3.10.15

# Verify PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Verify the key packages
python -c "import clip, diffusers, transformers, einops, omegaconf; print('All packages imported successfully')"
```

The main dependencies include:
- torch==2.8.0 (using CUDA 12.8)
- torchvision==0.23.0
- numpy==1.26.0
- transformers==4.26.1
- diffusers==0.35.1
- pytorch-lightning==1.4.2
- opencv-python==4.10.0.84
- CLIP (OpenAI vision-language model)
- k-diffusion (sampling algorithms)
- taming-transformers (vector quantization module)

### 5. Download Pretrained Models

#### Model Management System
DiffV2IR now supports custom model paths, avoiding downloads into the system cache directory. All models can be managed in a unified way.

#### Automatically Download Required Models
Use the provided download script to automatically download the BLIP and CLIP models:
```bash
# Download all required models into the models/ directory
python download_models.py

# Or specify a custom directory
python download_models.py --model-dir /path/to/your/models

# Download only specific models
python download_models.py --models blip clip

# Skip models that already exist
python download_models.py --skip-existing
```

#### DiffV2IR Main Model Weights
Download the DiffV2IR pretrained model from one of the following links:

**Option 1: Quark NetDisk**
- Link: https://pan.quark.cn/s/e2f28304ee90
- Access code: EWCz

**Option 2: HuggingFace**
- Link: https://huggingface.co/datasets/Lidong26/IR-500K/tree/main

Configure after downloading:
```bash
# Create the directory structure
mkdir -p pretrained/DiffV2IR/IR-500k/finetuned_checkpoints

# Move the downloaded file to the correct location
mv ~/Downloads/after_phase_2.ckpt pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/

# Verify the file (should be about 5.7GB)
ls -lh pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt
```

#### Model File Description
| Model | Purpose | Size | Default path |
|------|------|------|----------|
| after_phase_2.ckpt | DiffV2IR main model | ~5.7GB | pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/ |
| BLIP | Image caption generation | ~1.4GB | models/blip/model_base_caption_capfilt_large.pth |
| CLIP | Vision-language understanding | ~900MB | models/clip/ViT-L-14.pt |
| SAM (optional) | Segmentation map generation | ~2.4GB | models/sam/sam_vit_h_4b8939.pth |

#### Custom Model Paths
If you want to use a different model storage location, you can set an environment variable:
```bash
# Set the base model directory
export DIFFV2IR_MODEL_DIR=/your/custom/path/models

# Then run inference
python infer.py --input test_input --output test_output ...
```

Or modify the path configuration in `model_paths.py`:
```python
MODEL_PATHS = {
    'blip': '/your/path/to/blip_model.pth',
    'clip': '/your/path/to/clip_model.pt',
    # ...
}
```

## Quick Start - Inference Test

### Prepare Test Data
```bash
# Create test directories
mkdir -p test_input test_input_seg test_output

# Put your visible-light images into the test_input directory
# e.g. cp your_visible_image.jpg test_input/

# Generate segmentation maps (if you don't have any)
# Option 1: auto-generate with SAM (requires downloading the SAM model first)
python generate_sam_masks.py --input test_input --output test_input_seg

# Option 2: manually create simple segmentation maps (all-white images)
# For each input image, create a same-named PNG segmentation map
for img in test_input/*; do
    filename=$(basename "$img" | cut -d. -f1)
    convert -size 512x512 xc:white test_input_seg/${filename}.png
done
```

### Run Inference

#### Basic Command
```bash
# Run inference using the after_phase_2.ckpt model
python infer.py \
    --input test_input \
    --output test_output \
    --ckpt pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt \
    --steps 50 \
    --config configs/generate.yaml

```

#### Advanced Parameter Tuning
```bash
python infer.py \
    --input test_input \
    --output test_output \
    --ckpt pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt \
    --steps 100 \              # Increase steps for higher quality (default 100)
    --resolution 512 \         # Processing resolution (default 512)
    --cfg-text 7.5 \          # Text guidance strength (default 7.5)
    --cfg-image 1.5 \         # Image-conditioning strength (default 1.5)
    --cfg-seg 1.5 \           # Segmentation-map guidance strength (default 1.5)
    --seed 42 \               # Fixed seed for reproducible results
    --config configs/generate.yaml
```

### Verify Output
```bash
# List the generated infrared images
ls -la test_output/

# View the results with an image viewer
# eog test_output/*.png  # Ubuntu
# open test_output/*.png # macOS
```

## API Interface Design

### 1. Basic Inference Interface

```python
# Full command-line interface parameter reference
python infer.py \
    --input <input_image_folder> \        # Required: folder containing the visible-light images
    --output <output_image_folder> \       # Required: folder to save the infrared images
    --ckpt <model_weight_path> \           # Required: path to the pretrained model
    --steps <num_denoising_steps> \              # Optional: default 100, range 20-200
    --resolution <resolution> \           # Optional: default 512
    --cfg-text <text_guidance_strength> \       # Optional: default 7.5
    --cfg-image <image_conditioning_strength> \      # Optional: default 1.5
    --cfg-seg <seg_guidance_strength> \        # Optional: default 1.5
    --seed <random_seed> \               # Optional: for reproducing results
    --config configs/generate.yaml    # Required: path to the config file
```

**Notes:**
1. The input folder must contain visible-light images (jpg, png formats supported)
2. A corresponding segmentation-map folder is required, named `<input>_seg`
3. Segmentation maps must be in PNG format, with filenames matching the input images (extension removed)

### 2. Python API Wrapper (diffv2ir_api.py)

The project provides a wrapped Python API (`diffv2ir_api.py`) to simplify integration.

#### Simple Function Interface

```python
from diffv2ir_api import diffv2ir_convert

# The simplest way to call it
output_path = diffv2ir_convert(
    input_path="input.jpg",
    output_path="output.png",  # Optional; if omitted, *_infrared.png is auto-generated
    checkpoint_path="pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt",
    resolution=512,
    steps=50,
    use_fp16=True,  # Use half precision to reduce VRAM
    seed=42  # Fixed seed for reproducibility
)
print(f"Output path: {output_path}")
```

#### Class Interface (More Control)

```python
from diffv2ir_api import DiffV2IR

# Initialize the model (only once)
model = DiffV2IR(
    config_path="configs/generate.yaml",
    checkpoint_path="pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt",
    device="cuda",
    use_fp16=True,  # Enable half precision
    load_blip=True   # Load BLIP for image captioning
)

# Single-image conversion
output_path = model.convert(
    input_path="input.jpg",
    output_path="output.png",  # Optional
    mask_path="mask.png",      # Optional segmentation map
    resolution=512,
    steps=50,
    cfg_text=7.5,     # Text guidance strength
    cfg_image=1.5,    # Image-conditioning strength
    cfg_seg=1.5,      # Segmentation-map guidance strength
    seed=None,        # None means random
    edit_prompt=None  # Custom prompt; None means auto-generated
)

# Batch conversion
results = model.batch_convert(
    input_folder="input_folder",
    output_folder="output_folder",
    mask_folder="mask_folder",  # Optional
    resolution=512,
    steps=50
)

# results is a dict: {filename: output_path}
for input_name, output_path in results.items():
    if output_path:
        print(f"✓ {input_name} -> {output_path}")
    else:
        print(f"✗ {input_name} conversion failed")
```

#### API Features

- **Automatic output-path generation**: if `output_path` is not specified, `*_infrared.png` is auto-generated
- **Returns the output path**: all functions return the generated image path for easy downstream processing
- **Memory management**: automatically frees VRAM and supports FP16 mode to reduce memory usage
- **Flexible parameters**: all parameters are configurable with sensible defaults
- **Error handling**: includes complete error handling and informative messages

### 3. REST API Service

Create `api_server.py`:

```python
from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)
model = None

@app.route('/init', methods=['POST'])
def init_model():
    global model
    config = request.json
    model = DiffV2IR(
        config_path=config.get('config_path', 'configs/generate.yaml'),
        checkpoint_path=config.get('checkpoint_path'),
        device=config.get('device', 'cuda')
    )
    return jsonify({"status": "success", "message": "Model initialized"})

@app.route('/convert', methods=['POST'])
def convert_image():
    if model is None:
        return jsonify({"error": "Model not initialized"}), 400

    # Receive the base64-encoded image
    data = request.json
    image_base64 = data['image']
    steps = data.get('steps', 50)

    # Decode the image
    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data))

    # Save a temporary file
    temp_input = "temp_input.png"
    temp_output = "temp_output.png"
    image.save(temp_input)

    # Run the conversion
    model.convert(temp_input, temp_output, steps=steps)

    # Read the result and encode it
    with open(temp_output, 'rb') as f:
        result_base64 = base64.b64encode(f.read()).decode()

    # Clean up the temporary files
    os.remove(temp_input)
    os.remove(temp_output)

    return jsonify({
        "status": "success",
        "result": result_base64
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Input/Output Specification

### Input Requirements

1. **Image format**: PNG, JPG, JPEG supported
2. **Image size**:
   - Recommended sizes: 512x512 or 256x256
   - Any size is supported (auto-adjusted to the nearest multiple of 64)
3. **File structure**:
```
input_folder/
├── image1.jpg
├── image2.png
└── ...
```

4. **Segmentation maps (optional)**:
To use segmentation enhancement, prepare the corresponding segmentation maps:
```
input_folder_seg/
├── image1.png  # Segmentation map for input_folder/image1.jpg
├── image2.png
└── ...
```

### Output Format

1. **Image format**: PNG (lossless)
2. **Image size**: same as the adjusted input image size
3. **File naming**: keeps the same filename as the input file

## Parameter Tuning Guide

### Core Parameters

| Parameter | Default | Suggested range | Description |
|------|--------|----------|------|
| `steps` | 100 | 20-200 | Number of denoising steps; more = better quality but slower |
| `cfg_text` | 7.5 | 5.0-10.0 | Text guidance strength |
| `cfg_image` | 1.5 | 1.0-3.0 | Image-conditioning strength |
| `cfg_seg` | 1.5 | 1.0-3.0 | Segmentation-map guidance strength |
| `resolution` | 512 | 256-1024 | Processing resolution |
| `seed` | None | Any integer | Random seed, for reproducing results |

### Performance Optimization Tips

1. **Quick preview**: use `steps=20-30` for a fast preview
2. **High-quality output**: use `steps=100-150` for the best quality
3. **Batch processing**: keep the batch size at or below 4 images (depending on GPU VRAM)

### VRAM Optimization and Recommended Configuration

#### VRAM Usage (measured data)

**FP32 (standard precision) VRAM usage:**
| Resolution | Peak VRAM | Recommended GPU VRAM |
|--------|----------|-------------|
| 256×256 | ~13GB | 16GB+ |
| 384×384 | ~14GB | 16GB+ |
| 512×512 | ~18GB | 24GB+ |

**FP16 (half precision) VRAM usage:**
| Resolution | Peak VRAM | VRAM saved | Recommended GPU VRAM |
|--------|----------|----------|-------------|
| 256×256 | ~4.5GB | 75% | 6GB+ |
| 384×384 | ~5.7GB | 68% | 8GB+ |
| 512×512 | ~8.8GB | 51% | 12GB+ |

#### Recommended GPU VRAM Configuration

| GPU VRAM | Recommended setting | Example command |
|---------|----------|----------|
| **6-8GB** | 256×256 + FP16 | `python infer_optimized.py --resolution 256 --fp16` |
| **8-12GB** | 384×384 + FP16, or 256×256 FP32 | `python infer_optimized.py --resolution 384 --fp16` |
| **12-16GB** | 512×512 + FP16, or 384×384 FP32 | `python infer_optimized.py --resolution 512 --fp16` |
| **16-24GB** | 512×512 FP32 (full precision) | `python infer.py --resolution 512` |
| **24GB+** | 512×512 FP32 + batch processing | Supports parallel multi-image processing |

#### Using the Optimized Inference Script

The project provides an optimized inference script, `infer_optimized.py`, supporting the following options:

```bash
# FP16 half-precision inference (reduces VRAM by 50-75%)
python infer_optimized.py \
    --input test_input \
    --output test_output \
    --ckpt pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt \
    --resolution 512 \
    --steps 50 \
    --fp16  # Enable FP16

# Disable BLIP caption generation (saves a small amount of VRAM)
python infer_optimized.py \
    --input test_input \
    --output test_output \
    --ckpt pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt \
    --no-blip  # Disable BLIP
```

#### Optimization Advice

1. **Priority order when VRAM is insufficient:**
   - First: enable FP16 (`--fp16`)
   - Second: lower the resolution (`--resolution 256`)
   - Last: disable BLIP (`--no-blip`)

2. **Balancing quality and performance:**
   - FP16 has a negligible impact on output quality; it is recommended to use it preferentially
   - The number of steps does not affect VRAM usage; adjust freely based on quality needs
   - Resolution affects quality significantly; keep the original resolution where possible

3. **Batch processing strategy:**
   - 24GB VRAM: process 2-3 512×512 images at once with FP16
   - 16GB VRAM: process 2 384×384 images at once with FP16
   - 8GB VRAM: process one image at a time

## Frontend Integration Example

### React Integration Example

```javascript
// DiffV2IRService.js
class DiffV2IRService {
    constructor(apiUrl) {
        this.apiUrl = apiUrl;
        this.initialized = false;
    }

    async initialize(config) {
        const response = await fetch(`${this.apiUrl}/init`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(config)
        });
        const result = await response.json();
        this.initialized = true;
        return result;
    }

    async convertImage(imageFile, options = {}) {
        if (!this.initialized) {
            throw new Error('Model not initialized');
        }

        // Read the image file
        const base64 = await this.fileToBase64(imageFile);

        // Send the request
        const response = await fetch(`${this.apiUrl}/convert`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                image: base64,
                steps: options.steps || 50
            })
        });

        const result = await response.json();
        if (result.status === 'success') {
            return `data:image/png;base64,${result.result}`;
        }
        throw new Error(result.error);
    }

    fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => {
                const base64 = reader.result.split(',')[1];
                resolve(base64);
            };
            reader.onerror = reject;
        });
    }
}

// React component
function ImageConverter() {
    const [service] = useState(new DiffV2IRService('http://localhost:5000'));
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        service.initialize({
            checkpoint_path: 'pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt'
        });
    }, []);

    const handleConvert = async (file) => {
        setLoading(true);
        try {
            const infraredImage = await service.convertImage(file, {steps: 50});
            setResult(infraredImage);
        } catch (error) {
            console.error('Conversion failed:', error);
        }
        setLoading(false);
    };

    return (
        <div>
            <input type="file" onChange={(e) => handleConvert(e.target.files[0])} />
            {loading && <div>Converting...</div>}
            {result && <img src={result} alt="Infrared result" />}
        </div>
    );
}
```

## Complete Example - From Scratch

Below is a complete example of setting up and running DiffV2IR from scratch:

```bash
# 1. Create the working directory
mkdir ~/diffv2ir_workspace
cd ~/diffv2ir_workspace

# 2. Clone the code
git clone https://github.com/Jian-Zhang-3DV/DiffV2IR.git
cd DiffV2IR

# 3. Create and activate the virtual environment
conda create -n DiffV2IR python=3.10.15 -y
conda activate DiffV2IR

# 4. Install PyTorch (CUDA 12.8)
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# 5. Install all dependencies
pip install -r requirements.txt

# 6. Download the BLIP and CLIP models locally
python download_models.py

# 7. Create the model directory
mkdir -p pretrained/DiffV2IR/IR-500k/finetuned_checkpoints

# 8. Download the DiffV2IR main model (manually from Quark NetDisk or HuggingFace)
# Assuming it was downloaded to ~/Downloads
mv ~/Downloads/after_phase_2.ckpt pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/

# 9. Prepare test images
mkdir -p test_input test_input_seg test_output
# Copy some test images into test_input
cp ~/sample_images/*.jpg test_input/

# 10. Create simple segmentation maps (if you don't have SAM)
for img in test_input/*; do
    filename=$(basename "$img" | cut -d. -f1)
    # Use Python PIL to create a white image
    python -c "
from PIL import Image
img = Image.open('$img')
seg = Image.new('RGB', img.size, (255, 255, 255))
seg.save('test_input_seg/${filename}.png')
"
done

# 11. Run inference
python infer.py \
    --input test_input \
    --output test_output \
    --ckpt pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt \
    --steps 50 \
    --config configs/generate.yaml

# 12. View the results
ls -la test_output/
```

## Error Handling

### Common Errors and Solutions

1. **CUDA Out of Memory**
   ```bash
   # Error message: RuntimeError: CUDA out of memory

   # Solutions:
   # 1. Reduce the resolution
   python infer.py --resolution 256 ...

   # 2. Reduce batching (process one image at a time)
   # 3. Clear the GPU cache
   python -c "import torch; torch.cuda.empty_cache()"

   # 4. Use fewer steps
   python infer.py --steps 20 ...
   ```

2. **Model loading failure**
   ```bash
   # Error message: FileNotFoundError or RuntimeError: Error(s) in loading state_dict

   # Check the model file
   ls -lh pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/

   # Verify model integrity (should be about 5.7GB)
   du -h pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/*.ckpt

   # Confirm the PyTorch version
   python -c "import torch; print(torch.__version__)"  # should be 2.8.0
   ```

3. **Dependency package conflicts**
   ```bash
   # Error message: ImportError or ModuleNotFoundError

   # Recreate a clean environment
   conda deactivate
   conda env remove -n DiffV2IR
   conda create -n DiffV2IR python=3.10.15 -y
   conda activate DiffV2IR
   # Reinstall all dependencies...
   ```

4. **Missing segmentation maps**
   ```bash
   # Error message: FileNotFoundError: [Errno 2] No such file or directory: 'xxx_seg/xxx.png'

   # Solution 1: create simple white segmentation maps
   python -c "
import os
from PIL import Image
import glob

input_dir = 'test_input'
seg_dir = 'test_input_seg'
os.makedirs(seg_dir, exist_ok=True)

for img_path in glob.glob(f'{input_dir}/*'):
    img = Image.open(img_path)
    filename = os.path.splitext(os.path.basename(img_path))[0]
    seg = Image.new('RGB', img.size, (255, 255, 255))
    seg.save(f'{seg_dir}/{filename}.png')
    print(f'Created segmentation for {filename}')
"

   # Solution 2: generate with SAM (requires downloading the SAM model first)
   python generate_sam_masks.py --input test_input --output test_input_seg
   ```

5. **CLIP model download failure**
   ```bash
   # Error message: HTTPError or connection timeout

   # Download the CLIP model manually
   mkdir -p ~/.cache/clip
   wget -O ~/.cache/clip/ViT-L-14.pt \
     https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
   ```

6. **BLIP model download failure**
   ```bash
   # Error message: URLError or HTTPError

   # Use the download script to download into a local directory
   python download_models.py --models blip

   # Or manually download to the specified location
   mkdir -p models/blip
   wget -O models/blip/model_base_caption_capfilt_large.pth \
     https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth
   ```

7. **Model path issues**
   ```bash
   # If the model is not in the default location, set the environment variable
   export DIFFV2IR_MODEL_DIR=/your/models/directory

   # Or modify the path configuration in model_paths.py
   ```

## Monitoring and Logging

### Performance Monitoring
```python
import time
import psutil
import GPUtil

def monitor_conversion(input_path, output_path):
    start_time = time.time()

    # Get the initial resource usage
    cpu_before = psutil.cpu_percent()
    mem_before = psutil.virtual_memory().percent
    gpu_before = GPUtil.getGPUs()[0].memoryUsed

    # Run the conversion
    model.convert(input_path, output_path)

    # Compute the resource usage
    duration = time.time() - start_time
    cpu_usage = psutil.cpu_percent() - cpu_before
    mem_usage = psutil.virtual_memory().percent - mem_before
    gpu_usage = GPUtil.getGPUs()[0].memoryUsed - gpu_before

    return {
        'duration': duration,
        'cpu_usage': cpu_usage,
        'memory_usage': mem_usage,
        'gpu_memory': gpu_usage
    }
```

### Logging Configuration
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('diffv2ir.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('DiffV2IR')
```

## Deployment Recommendations

### Docker Deployment

```dockerfile
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy the project files
COPY . /app/DiffV2IR

# Install the Python dependencies
RUN pip install -r DiffV2IR/requirements.txt

# Download the model (optional; you can also mount it)
# RUN wget -O /app/models/after_phase_2.ckpt <model_url>

# Expose the port
EXPOSE 5000

# Start the service
CMD ["python", "DiffV2IR/api_server.py"]
```

### Production Configuration

1. **Deploy the Flask app with Gunicorn**
```bash
gunicorn -w 4 -b 0.0.0.0:5000 api_server:app
```

2. **Use an Nginx reverse proxy**
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location /api/diffv2ir {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 50M;
    }
}
```

3. **Load-balancing configuration** (multi-GPU servers)
```python
# Use Celery for task-queue management
from celery import Celery

app = Celery('diffv2ir', broker='redis://localhost:6379')

@app.task
def convert_image_task(input_path, output_path, **kwargs):
    return model.convert(input_path, output_path, **kwargs)
```

## Testing and Verification

### Functional Tests

The project provides test scripts to verify the API functionality:

```bash
# Test the API wrapper
python test_api.py

# Simple functional test
python test_simple.py

# Direct inference test
python test_direct.py
```

### Test Data

The project includes test data in the `data_for_diffv2ir/` directory:
- `input/` - visible-light images
- `input_seg/` - corresponding segmentation maps
- `ground_truth/` - reference infrared images

### Quick Verification

```python
# Quickly test a single image
from diffv2ir_api import diffv2ir_convert

output = diffv2ir_convert(
    input_path="data_for_diffv2ir/input/DJI_0061.png",
    checkpoint_path="pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt",
    steps=20,  # A small number of steps for a quick test
    resolution=256,  # Low resolution to speed things up
    use_fp16=True
)
print(f"Generated successfully: {output}")
```

## Best Practices

1. **Model warm-up**: run one inference after the service starts to avoid first-request latency
2. **Caching**: cache results for identical inputs to avoid redundant computation
3. **Asynchronous processing**: use a message queue to handle large batch conversion jobs
4. **Health checks**: implement a `/health` endpoint to monitor service status
5. **Versioning**: provide separate endpoints for different model versions
6. **Dependency management**: use pinned dependency versions to avoid version conflicts

## Quick Checklist

Before running inference, make sure all of the following are done:

- [ ] Python 3.10.15 environment created and activated
- [ ] PyTorch 2.8.0 and CUDA 12.8 installed
- [ ] All dependencies installed correctly (including Git dependencies)
- [ ] Pretrained model downloaded to the correct location (at least one .ckpt file)
- [ ] Input image folder prepared
- [ ] Corresponding segmentation-map folder prepared (named input_seg)
- [ ] Output folder created

## Verify Installation

Run the following script to verify the environment setup:

```python
# save as check_installation.py
import sys
import os

def check_installation():
    errors = []
    warnings = []

    # Check the Python version
    if sys.version_info[:2] != (3, 10):
        warnings.append(f"Python version is {sys.version_info.major}.{sys.version_info.minor}, recommended 3.10")

    # Check the required packages
    required_packages = [
        'torch', 'torchvision', 'numpy', 'transformers',
        'omegaconf', 'einops', 'PIL', 'cv2', 'clip'
    ]

    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'cv2':
                import cv2
            else:
                __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            errors.append(f"✗ {package} not installed")

    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            warnings.append("CUDA not available, will run on CPU (very slow)")
    except:
        pass

    # Check the model file
    model_paths = [
        'pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt'
    ]

    model_found = False
    for path in model_paths:
        if os.path.exists(path):
            print(f"✓ Model found: {path}")
            model_found = True

    if not model_found:
        errors.append("✗ No pretrained models found")

    # Check the config file
    if os.path.exists('configs/generate.yaml'):
        print("✓ Config file found")
    else:
        errors.append("✗ configs/generate.yaml not found")

    # Summary
    print("\n" + "="*50)
    if errors:
        print("❌ Installation incomplete. Errors:")
        for error in errors:
            print(f"  {error}")
    else:
        print("✅ Installation successful!")

    if warnings:
        print("\n⚠️  Warnings:")
        for warning in warnings:
            print(f"  {warning}")

    return len(errors) == 0

if __name__ == "__main__":
    success = check_installation()
    sys.exit(0 if success else 1)
```

Run the verification:
```bash
python check_installation.py
```

## Minimal Test Example

Create a minimal test to verify the inference functionality:

```bash
# Create the test script
cat > test_inference.py << 'EOF'
import os
from PIL import Image
import numpy as np

# Create the test directories
os.makedirs('minimal_test/input', exist_ok=True)
os.makedirs('minimal_test/input_seg', exist_ok=True)
os.makedirs('minimal_test/output', exist_ok=True)

# Create a simple test image
test_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
test_img.save('minimal_test/input/test.jpg')

# Create the corresponding segmentation map
seg_img = Image.new('RGB', (512, 512), (255, 255, 255))
seg_img.save('minimal_test/input_seg/test.png')

print("Test data created. Run inference with:")
print("python infer.py \\")
print("    --input minimal_test/input \\")
print("    --output minimal_test/output \\")
print("    --ckpt pretrained/DiffV2IR/IR-500k/finetuned_checkpoints/after_phase_2.ckpt \\")
print("    --steps 20 \\")
print("    --config configs/generate.yaml")
EOF

python test_inference.py
```

## Technical Support

- GitHub Issues: https://github.com/Jian-Zhang-3DV/DiffV2IR/issues
- Original project: https://github.com/your-original-repo/DiffV2IR
- Paper link: https://arxiv.org/abs/2503.19012
- Project homepage: https://diffv2ir.github.io/

## License

This project is developed based on the Stable Diffusion codebase and follows the corresponding open-source licenses.

## Changelog

- 2024-08-24: Updated the integration guide; added complete environment setup and inference instructions
- 2024-08-21: Initial release
