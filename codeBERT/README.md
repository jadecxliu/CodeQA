# CodeQA

## Experiments on CodeBERT

### Dependency

- pip install torch
- pip install transformers

### Data

You can download data from [Google Drive](https://drive.google.com/drive/folders/1i04sJNUHwMuDfMV2UfWeQG-Uv8MRw_qh?usp=sharing). Unzip it and move it to `./data`.  

### Train 

We fine-tune the model on 3*1080Ti GPUs.

Please run the following scripts:

`bash java_script.sh [gpu-id] [model-name]`

`bash python_script.sh [gpu-id] [model-name]`

### Inference

After the training process, several best checkpoints are stored in a folder named after your model name, for example, `./output/[model-name]/checkpoint-best-bleu/pytorch_model.bin`. You can run the following scripts to get the results on test set:

`bash java_script_test.sh [gpu-id] [model-name]`

`bash python_script_test.sh [gpu-id] [model-name]`

### Pretrained Model

Java and Python pre-trained models (20 epochs) are available [here](https://drive.google.com/drive/folders/1A_C6O649cXjjpk3KKHIe6eaEU5tBaMLJ?usp=sharing).























