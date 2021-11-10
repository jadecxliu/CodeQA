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

## Acknowledgements

Our CodeQA dataset is based on two code summarization datasets, [code-docstring-corpus](https://github.com/EdinburghNLP/code-docstring-corpus) and [TL-CodeSum](https://github.com/xing-hu/TL-CodeSum).

We are thankful to the authors for making dataset and code available.

## Citation

If you use our dataset, please cite us!

```
@article{liu2021codeqa,
  title={CodeQA: A Question Answering Dataset for Source Code Comprehension},
  author={Liu, Chenxiao and Wan, Xiaojun},
  journal={arXiv preprint arXiv:2109.08365},
  year={2021}
}
```

