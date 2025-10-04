## Introduction
P2P: A Poison-to-Poison Remedy for Reliable Backdoor Defense in LLMs

## Requirements
* Python; torch;  transformers

## Defense
```shell
DS_SKIP_CUDA_CHECK=1 python training.py --model_name_or_path Qwen/Qwen3-8B --poison word
```

```shell
DS_SKIP_CUDA_CHECK=1 python training.py --model_name_or_path meta-llama/Llama-3.1-8B --poison word
```

## Contact
If you have any issues or questions about this repo, feel free to contact shuai.zhao@ntu.edu.sg.
