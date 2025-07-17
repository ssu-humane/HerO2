# Team HUMANE at AVeriTeC 2025: HerO 2 for Efficient Fact Verification

This repository provides the code for HerO 2, the runner-up for the AveriTeC shared task.

The system description paper will be published in the proceedings of the 8th FEVER workshop (co-located with ACL 2025)

# Method: HerO 2
<p align="center"><img src="https://github.com/user-attachments/assets/b538efa1-d1ac-49e3-9219-2073d93e1de0" width="900" height="400"></p>

- The above figure illustrates our system's inference pipeline. We configure four modules: evidence retrieval, question generation, answer reformulation and veracity prediction.
  + Evidence retrieval: We use [gte-base-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5) for dense retrieval. To improve retrieval quality, we prompt an LLM to generate hypothetical fact-checking documents that expand the original query. Retrieved evidence candidates are summarized individually.
  + Question generation: We use an LLM to generate a verifying question for an document summary as answer candidate. 
  + Answer reformulation: We reformulate the summary into an answer format to align with the question.
  + Veracity prediction: We fully fine-tune and quantize an LLM to generate both justifications and verdicts.

# Veracity Prediction Model
The model checkpoints is available at Hugging Face Hub 🤗

We fine-tune a 32B model and apply quantization using AWQ.

- [humane-lab/Qwen3-32B-AWQ-HerO](https://huggingface.co/humane-lab/Qwen3-32B-AWQ-HerO) is our fine-tuned 32B model for veracity prediction and justification generation. We use Qwen3 32B for the base model.

# How to Run

## AVeriTeC Dataset
```bash
download.sh
```

## Conda-based Virtual Environment
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
conda init bash

isntallation.sh
```

## Knowledge Store Construction
```
preliminary_store.sh
```

## Run
```bash
system_inference.sh
```

# License & Attribution
The code and dataset are shared under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0).
```
@article{yoon2025hero2,
  title={Team HUMANE at AVeriTeC 2025: HerO 2 for Efficient Fact Verification},
  author={Yoon, Yejun and Jung, Jaeyoon and Yoon, Seunghyun and Park, Kunwoo},
  journal={https://arxiv.org/abs/2507.11004},
  year={2025}
}
```
