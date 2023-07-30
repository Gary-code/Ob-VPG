# Ob-VPG: Object-level Visual Paraphrase Generation

Released code for paper [Visual Paraphrase Generation with Key Information Retained](https://dl.acm.org/doi/10.1145/3585010) in TOMM 2023.

![model-614](./pic/model.png)

## Requirements and Setup

1. Install Anaconda or Miniconda distribution based on Python3+ from their [downloads' site](https://conda.io/docs/user-guide/install/download.html).
2. Install python package the code needs.

## Dataset & Pretrain VisualBERT model

All the training, validation and test data in the `data_sentences` folder.

* All this data is preprocessed from the MSCOCO caption dataset.
* More details about preprocessing, you can see in [repository](https://github.com/Gary-code/PQG)

You can download visualBERT model from [link](https://huggingface.co/uclanlp/visualbert-vqa-coco-pre)

## Training & Evaluation

```sh
python train.py
```

## Reference

```
@article{xie2023visual,
  title={Visual paraphrase generation with key information retained},
  author={Xie, Jiayuan and Chen, Jiali and Cai, Yi and Huang, Qingbao and Li, Qing},
  journal={ACM Transactions on Multimedia Computing, Communications and Applications},
  volume={19},
  number={6},
  pages={1--19},
  year={2023},
  publisher={ACM New York, NY}
}
```



