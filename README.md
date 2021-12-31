# CodeQA
CodeQA is a free-form question answering dataset for the purpose of source code comprehension: given a code snippet and a question, a textual answer is required to be generated. 

To obtain natural and faithful questions and answers, we implement syntactic rules and semantic analysis to transform code comments into question-answer pairs. 

We hope this new dataset can serve as a useful research benchmark for source code comprehension.

You can find more details, analyses, and baseline results in our Findings of EMNLP 2021 paper "[CodeQA: A Question Answering Dataset for Source Code Comprehension](https://arxiv.org/pdf/2109.08365.pdf)".

## Data

The dataset (ver. 1.0) can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1i04sJNUHwMuDfMV2UfWeQG-Uv8MRw_qh?usp=sharing). 

It contains a Java dataset with 119,778 question-answer pairs and a Python dataset with 70,085 question-answer pairs. 

A few examples of CodeQA data format are shown in `data_sample`. Each one contains a question, an answer and a code snippet. For the code snippet, we provide an original version (`.code.original`) as well as a processed version (`.code`). Details of the processing are available in the following code summarization datasets.

## Evaluation 

We follow the same evaluation method of automatic metrics (BLEU, ROUGE-L, METEOR) as in [Ahmad et al. (2020)](https://arxiv.org/abs/2005.00653). 

Source code can be found [here](https://github.com/wasiahmad/NeuralCodeSum).

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

## Acknowledgements

Our CodeQA dataset is based on two code summarization datasets, [code-docstring-corpus](https://github.com/EdinburghNLP/code-docstring-corpus) and [TL-CodeSum](https://github.com/xing-hu/TL-CodeSum).

We are thankful to the authors for making dataset and code available.

## Citation

If you use our dataset, please cite us!

```
@inproceedings{liu2021codeqa,
  title={CodeQA: A Question Answering Dataset for Source Code Comprehension},
  author={Liu, Chenxiao and Wan, Xiaojun},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2021},
  pages={2618--2632},
  year={2021}
}
```
