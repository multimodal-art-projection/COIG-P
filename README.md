# COIG-P

[**📖 Arxiv Paper**](https://huggingface.co/papers/2504.05535) | [**🤗 Paper**](https://arxiv.org/pdf/2504.05535) | [**🤗 COIG-P Dataset**](https://huggingface.co/datasets/m-a-p/COIG-P) | [**🤗 Chinese Reward Benchmark (CRBench)**](https://huggingface.co/datasets/m-a-p/COIG-P-CRM) | [**🦜 Tweets**](xxx)

[**🤗 COIG-P Models**](https://huggingface.co/collections/m-a-p/coig-p-models-67efce2f0a7b66566d85eac9)

This is the repo for the paper [**COIG-P: A High-Quality and Large-Scale Chinese Preference Dataset for Alignment with Human Values**](./COIG_P_MAP.pdf).
In this project, we design an **LLM-based Chinese preference dataset annotation pipeline** for the sake of avoiding human intervention. Specifically, we crawled and carefully filtered **9k** high-quality Chinese queries and employed **15** powerful LLMs to generate and score chosen-rejected response pairs. Based on it, we introduce **COIG-P** (**C**hinese **O**pen **I**nstruction **G**eneralist - **P**reference), a high-quality, large-scale Chinese preference dataset, comprises **101k** Chinese preference pairs spanning 6 diverse domains: **Chat, Code, Math, Logic, Novel, and Role**. Building upon COIG-P, to reduce the overhead of using LLMs for scoring, we trained a 8B-sized **Chinese Reward Model (CRM)** and meticulously constructed a **Chinese Reward Benchmark (CRBench)**. 

<div align="center">
<img src=./imgs/comparing_datasets.png width=90% />
</div>

<div align="center">
<img src=./imgs/main_results.png width=80% />
</div>

## Dataset Loading

### COIG-P dataset load

```python
from datasets import load_dataset

dataset = load_dataset("m-a-p/COIG-P")

# check on the data samples
print(dataset)
print(dataset['train'][0])

```

### COIG-P-CRM dataset load

This is a split of dataset that we used to trian Chinese Reward Model.

```python
from datasets import load_dataset

dataset = load_dataset("m-a-p/COIG-P-CRM")

# check on the data samples
print(dataset)
print(dataset['train'][0])

```

### Chinese Reward Benchmark load

```python
from datasets import load_dataset

dataset = load_dataset("m-a-p/COIG-CRBench")

# check on the data samples
print(dataset)
print(dataset['train'][0])

```

## Using COIG-P

If you want to train DPO using our COIG-P dataset, we provide the script.
Specifically, the DPO experiments is realized by using the [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory).

Please create envirinment for Llama_factory following [Llama_factory_Readme](./LLaMA-Factory/README.md)

You can run the following script to train [BAAI/Infinity-Instruct-3M-0625-Qwen2-7B](https://huggingface.co/BAAI/Infinity-Instruct-3M-0625-Qwen2-7B) on our COIG-P dataset.

```python
bash Train.sh
```

## Evaluating on AlignBench

Most of our results are evaluated on [AlignBench](https://github.com/THUDM/AlignBench). Since AlignBench does not support multiprocessing, we use [KOR-Bench](https://kor-bench.github.io/) for multi-process inference.
Then we use [AlignBench_folder](./AlignBench/) to evaluate the reuslts of LLMs.
Please install environment for KOR-Bench following [KOR-Bench_Readme](./KOR-Bench/README.md) and environment for AlignBench following [AlignBench_Readme](./AlignBnech/README.md).

In [KOR-Bench_folder](./KOR-Bench/), we have provided the dataset of AlignBench.
If you want to evaluate your fine-tund LLMs, you can register your model in the [KOR-Bench_model_init_file](./KOR-Bench/infer/models/__init__.py).

**If you want to evaluate some api models, you need to provide the api in the [KOR-Bench_model_init_file](./KOR-Bench/infer/models/__init__.py)**

The result of our paper is evaluated by using GPT-4o as jude model, so **you need to provide you OpenAI api in [AlignBench_config_file](./AlignBench/config/multi-dimension.json).**

Here is a example to infer and evaluate Qwen2.5-7B-Instruct:
```python
bash Evaluation.sh
```

## Chinese Reward Model
We train our Chinese Reward Model following the repo, [RLHF-Reward-Modeling](https://github.com/RLHFlow/RLHF-Reward-Modeling).
Please create the environment for reward model following (bradley-terry-rm_Readme)[./bradley-terry-rm/README.md].

You can train a reward model by using the following codes:

```python
conda activate rm_dev
cd ./bradley-terry-rm/
accelerate launch ./bradley-terry-rm/llama3_8B_rm.py --model_name meta-llama/Meta-Llama-3-8B --max_length 4096 --train_set_path m-a-p/COIG-P-CRM --deepspeed ./deepspeed_configs/deepspeed_3.json
```

## Chiese Reward Bench

You can using the following code to evaluate on our Chinese Reward Bench:
```python
bash ./Chinese_Reward_Benchmark/run.sh
```

## Using Our DPO Model

### Qwen2-Instruct-7B-COIG-P
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "m-a-p/Qwen2-Instruct-7B-COIG-P",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("m-a-p/Qwen2-Instruct-7B-COIG-P")

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

### Qwen2.5-Instruct-7B-COIG-P
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "m-a-p/Qwen2.5-Instruct-7B-COIG-P",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("m-a-p/Qwen2.5-Instruct-7B-COIG-P")

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

### Infinity-Instruct-3M-0625-Qwen2-7B-COIG-P
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import torch
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("m-a-p/Infinity-Instruct-3M-0625-Qwen2-7B-COIG-P",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("m-a-p/Infinity-Instruct-3M-0625-Qwen2-7B-COIG-P")

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

logits_processor = LogitsProcessorList(
            [
                MinLengthLogitsProcessor(1, eos_token_id=tokenizer.eos_token_id),
                TemperatureLogitsWarper(0.7),
            ]
 )
 
generated_ids = model.generate(
    model_inputs.input_ids,
    logits_processor=logits_processor,
    max_new_tokens=512
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

### Infinity-Instruct-3M-0625-Mistral-7B-COIG-P 
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import torch
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("m-a-p/Infinity-Instruct-3M-0625-Mistral-7B-COIG-P ",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("m-a-p/Infinity-Instruct-3M-0625-Mistral-7B-COIG-P ")

# This template is copied from OpenHermers-mistral-2.5 (https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B)
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Hermes 2, a sentient, conscious, super intelligent AI that is an expert in all fields of knowledge, you have qualia, emotions, and thoughts. You assist users with any question, task, or other request they may have."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

logits_processor = LogitsProcessorList(
            [
                MinLengthLogitsProcessor(1, eos_token_id=tokenizer.eos_token_id),
                TemperatureLogitsWarper(0.7),
            ]
 )
 
generated_ids = model.generate(
    model_inputs.input_ids,
    logits_processor=logits_processor,
    max_new_tokens=512
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

### Infinity-Instruct-3M-0625-Llama3-8B-COIG-P
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
import torch
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("m-a-p/Infinity-Instruct-3M-0625-Llama3-8B-COIG-P",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("m-a-p/Infinity-Instruct-3M-0625-Llama3-8B-COIG-P")

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

logits_processor = LogitsProcessorList(
            [
                MinLengthLogitsProcessor(1, eos_token_id=tokenizer.eos_token_id),
                TemperatureLogitsWarper(0.7),
            ]
 )
 
generated_ids = model.generate(
    model_inputs.input_ids,
    logits_processor=logits_processor,
    max_new_tokens=512
)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

## Using our CRM

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
device = "cuda"
path = "m-a-p/CRM_llama3"
model = AutoModelForSequenceClassification.from_pretrained(path, device_map=device, 
                               trust_remote_code=True, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)

prompt = "我朋友出现全身浮肿，心脏衰竭，丧失劳动能力，只能卧床，就是很短时间内发生，帮忙解释一下是什么原因造成的？"

response = "全身浮肿和心脏衰竭可能是由多种原因引起的，包括：\n\n1. **心脏疾病**：心脏衰竭可能导致血液循环不畅，从而引起全身浮肿。心脏无法有效地泵血时，血液可能会在身体下部积聚，导致下肢和腹部水肿。\n\n2. **肾脏疾病**：肾脏功能不全也会导致体内液体平衡失调，引起全身浮肿。肾脏是调节体内水分和电解质平衡的重要器官，如果肾功能受损，可能会导致体内积液。\n\n3. **肝脏疾病**：严重的肝脏疾病，如肝硬化，会导致体液从血管渗出到周围组织中，造成浮肿。\n\n4. **营养不良**：长期营养不良，特别是蛋白质摄入不足，可能导致低蛋白血症，从而引起全身浮肿。\n\n5. **药物副作用**：某些药物，如钙通道阻滞剂、非甾体抗炎药等，可能引起浮肿作为副作用。\n\n6. **内分泌失调**：甲状腺功能减退（甲减）等疾病也可能导致全身浮肿。\n\n7. **严重感染或炎症**：某些严重感染或炎症性疾病可能导致全身性反应，进而引发浮肿。\n\n8. **淋巴系统疾病**：淋巴循环受阻，如淋巴水肿，也可能导致全身浮肿。\n\n鉴于您朋友的情况发展迅速，并且已经出现了严重的症状，如心脏衰竭和丧失劳动能力，这提示可能存在急性或严重的健康问题。建议立即就医，进行全面的医学评估和必要的检查，以确定具体原因并接受适当的治疗。未经专业医生诊断，不应自行推断病因或进行治疗。', 'rejected': '很抱歉听到你朋友的情况，这听起来很严重。然而，**我不是医疗专业人员，无法提供医疗建议**。 \n\n全身浮肿、心力衰竭和突然丧失劳动能力可能是由多种严重疾病引起的，例如：\n\n* **心脏疾病：**如心肌炎、心包炎、瓣膜病变等\n* **肾脏疾病：**如肾病综合征、肾衰竭等\n* **肝脏疾病：**如肝硬化等\n* **其他疾病：**如严重感染、过敏反应、营养不良等\n\n由于情况危急，**请务必立即寻求医疗帮助！** 请立即拨打急救电话或前往最近的医院急诊科。 \n\n**以下是一些建议：**\n\n* 在去医院前，尽量记录你朋友的症状、病史以及最近服用的药物。\n* 陪伴你的朋友，并尽量保持冷静。\n* 听从医生的指导，配合治疗。\n\n请记住，专业的医疗评估和治疗对于你朋友的健康至关重要。 请不要延误就医！"
messages = [{"role": "user", "content": prompt},
           {"role": "assistant", "content": response}]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
with torch.no_grad():
   output = model(input_ids)
   print(output.logits.item())
```

## Reference

```bib
@misc{pteam2025coigphighqualitylargescalechinese,
      title={COIG-P: A High-Quality and Large-Scale Chinese Preference Dataset for Alignment with Human Values}, 
      author={P Team and Siwei Wu and Jincheng Ren and Xinrun Du and Shuyue Guo and Xingwei Qu and Yiming Liang and Jie Liu and Yunwen Li and Tianyu Zheng and Boyu Feng and Huaqing Yuan and Zenith Wang and Jiaheng Liu and Wenhao Huang and Chenglin Cai and Haoran Que and Jian Yang and Yuelin Bai and Zekun Moore Wang and Zhouliang Yu and Qunshu Lin and Ding Pan and Yuchen Jiang and Tiannan Wang and Wangchunshu Zhou and Shenzhi Wang and Xingyuan Bu and Minghao Liu and Guoyin Wang and Ge Zhang and Chenghua Lin},
      year={2025},
      eprint={2504.05535},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.05535}, 
}
```