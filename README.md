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

## Inference

Wei, Shaokui, Hongyuan Zha, and Baoyuan Wu. "Mitigating backdoor attack by injecting proactive defensive backdoor." Advances in Neural Information Processing Systems 37 (2024): 80674-80705.


## Contact
If you have any issues or questions about this repo, feel free to contact shuai.zhao@ntu.edu.sg.

## Citation
If you find our work valuable and use it in your research, please cite our paper using the following BibTeX entry:
```shell
@article{zhao2025p2p,
  title={P2P: A Poison-to-Poison Remedy for Reliable Backdoor Defense in LLMs},
  author={Zhao, Shuai and Wu, Xinyi and Zhao, Shiqian and Wu, Xiaobao and Guo, Zhongliang and Jia, Yanhao and Luu, Anh Tuan},
  journal={arXiv preprint arXiv:2510.04503},
  year={2025}
}
```
