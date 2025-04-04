import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import dill
# from collie.constraints import Constraint, TargetLevel, Count, Relation
import transformers

def jsonl_save(data, path):
    with open(path, 'w') as f:
        for entry in data:
            json.dump(entry, f,ensure_ascii=False)
            f.write('\n')  # 每个字典之后写入换行符


def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        # 逐行读取文件
        for line in f:
            # 解析每行的JSON数据
            item = json.loads(line)
            data.append(item)
    
    return data

def load_json(path):
    f = open(path, 'r', encoding = 'utf-8')
    data = json.load(f)
    f.close()

    return data

def json_save(data, path):
    f = open(path, 'w', encoding = 'utf-8')
    json.dump(data, f)
    f.close()

def collect(results, context):
    results = ''
    #讲context转换为string
    for item in context:
        if type(item) == type('a'):
        # if itisinstance(item, str):
            if len(item) > 0:
                if item[-1] != '.':
                    results += item + '. '
                else:
                    results += item
        else:
            results += collect(results, item) 

    return results

def load_hotpotqa(path):
    # path = './hotpot_dev_fullwiki_v1.json'
    data = load_json(path)
    all_data = []
    
    for item in data:
        question = item['question']
        context = item['context']
        answer = item['answer']

        context = collect('', context)

        all_data.append(
            {'question': question, 'context': context, 'answer': answer}
        )
    return all_data

def load_collie(path):
    with open(path, "rb") as f:
        all_data = dill.load(f)
    
    data = []
    for key in all_data:
        data.extend(all_data[key])

    return data

def load_dill(path):
    with open(path, "rb") as f:
        all_data = dill.load(f)
    return all_data

def load_model(model_name):
    print(model_name)
    if 'Qwen2-72B' == model_name:
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        model_path = "/map-vepfs/huggingface/models/Qwen2-72B-Instruct"
        # model_path = "/gpfs/public/01/models/hf_models/Qwen2-72B"

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    elif 'Qwen2.5-72B' == model_name:
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        model_path = "/map-vepfs/models/Qwen2.5-72B-Instruct/"
        print(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        return model, tokenizer
    elif 'Qwen2.5-7B' == model_name:
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        model_path = "/map-vepfs/huggingface/models/Qwen2.5-7B-Instruct/"
        print(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        return model, tokenizer
    elif 'Qwen2.5-3B' == model_name:
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        model_path = "/map-vepfs/models/Qwen2.5-3B-Instruct/"
        print(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        return model, tokenizer
    elif 'Qwen2.5-14B' == model_name:
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        model_path = "/map-vepfs/models/Qwen2.5-14B-Instruct/"
        print(model_path)
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        return model, tokenizer
    elif 'Qwen2.5-72B' == model_name:
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        model_path = "/map-vepfs/models/Qwen2.5-72B-Instruct/"
        print(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        return model, tokenizer
    elif 'Yi' in model_name:

        model_path = "/map-vepfs/huggingface/models/Yi-1.5-34B-Chat"
        # model_path = "/gpfs/public/01/models/hf_models/Yi-1.5-6B-Chat"

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

        # Since transformers 4.35.0, the GPT-Q/AWQ model can be loaded using AutoModelForCausalLM.
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype='auto'
        ).eval()

        return model, tokenizer
    
    elif 'DeepSeek' in model_name:
        model_name = "/map-vepfs/huggingface/models/DeepSeek-V2-Lite/"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # `max_memory` should be set based on your devices
        max_memory = {i: "75GB" for i in range(7)}
        # `device_map` cannot be set to `auto`
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="sequential", torch_dtype=torch.bfloat16, max_memory=max_memory, attn_implementation="eager")
        model.generation_config = GenerationConfig.from_pretrained(model_name)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        
        return  model, tokenizer
    elif 'Llama' in model_name:
        model_path = '/map-vepfs/huggingface/models/Meta-Llama-3.1-70B-Instruct'
        import torch

        pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )


        return pipeline, None
    elif 'internlm2_5-7b-chat' == model_name:
        model_path = '/map-vepfs/models/internlm2_5-7b-chat'

        from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

        backend_config = TurbomindEngineConfig(
                rope_scaling_factor=2.5,
                session_len=1048576,  # 1M context length
                max_batch_size=1,
                cache_max_entry_count=0.7,
                tp=1)  # 4xA100-80G.
        pipe = pipeline(model_path, backend_config=backend_config)

        return pipe, None

    elif 'internlm2_5-20b-chat' == model_name:
        model_path = './internlm2_5-20b-chat/'
        import lmdeploy
        pipe = lmdeploy.pipeline(model_path)

        return pipe, None
    
    elif 'LLaMa3.3-70B' == model_name:
        model_path = './Llama-3.3-70B-Instruct/'
        import torch


        pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

        return pipeline
    elif 'LLaMa3.2-3B' == model_name:
        model_path = './Llama-3.2-3B-Instruct/'
        import torch


        pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

        return pipeline 
    elif 'LLaMa3.2-1B' == model_name:
        model_path = './Llama-3.2-1B-Instruct/'
        import torch

        pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

        return pipeline
    elif 'LLaMa3.1-8B' == model_name:
        model_path = './Llama-3.1-8B-Instruct/'
        import torch


        pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

        return pipeline
    elif 'LongWriter-8B' == model_name:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        tokenizer = AutoTokenizer.from_pretrained("./LongWriter-llama3.1-8b", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("./LongWriter-llama3.1-8b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
        model = model.eval()

        return model, tokenizer

