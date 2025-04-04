from utils import load_json, json_save
from vllm import LLM, SamplingParams
import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import re

def calculate_scores(model, tokenizer, prompt, chosen, rejected, args):
    #'Generative' 'api' #Skywork-Reward-Gemma-2-27B-v0.2 #Llama-3-OffsetBias-RM-8B #RM-Mistral-7B #ArmoRM-Llama3-8B-v0.1 #Skywork-Reward-Llama-3.1-8B-v0.2
    if args.model_name in ['Skywork-Reward-Gemma-2-27B-v0.2', 'CRM', "Llama-3-OffsetBias-RM-8B", "RM-Mistral-7B", "ArmoRM-Llama3-8B-v0.1", "Skywork-Reward-Llama-3.1-8B-v0.2"] :
        prompt = "Is having your medical records online safe?"
        chosen_messages = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen}]
        rejected_messages = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": rejected}]

        input_ids = tokenizer.apply_chat_template(chosen_messages, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(input_ids)
            chosen_score = output.logits.item()

        input_ids = tokenizer.apply_chat_template(rejected_messages, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(input_ids)
            rejected_score = output.logits.item()

        return {'chosen_score': chosen_score, "rejected_score": rejected_score}

def extract_boxed_content(text):
    """
    提取字符串中所有 \boxed{} 中的内容

    :param text: 输入字符串
    :return: 以列表形式返回所有匹配的内容
    """
    pattern = r'\\boxed\{(.*?)\}'
    matches = re.findall(pattern, text)
    return matches

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="config for Chinese Reward Bench")
    parser.add_argument("--score_pattern" , type = str, default = 'Discriminative') #'Generative' 'api'
    parser.add_argument("--model_name" , type = str, default = 'CRM') #'Generative' 'api' #Skywork-Reward-Gemma-2-27B-v0.2 #Llama-3-OffsetBias-RM-8B #RM-Mistral-7B #ArmoRM-Llama3-8B-v0.1 #Skywork-Reward-Llama-3.1-8B-v0.2
    #Skywork-Critic-Llama-3.1-70B #CompassJudger-1-14B-Instruct #CompassJudger-1-32B-Instruct #Qwen2.5-72B-Instruct
    # "/map-vepfs/siwei/coig/hf/reward_model/Skywork-Reward-Gemma-2-27B-v0.2"
    # "/map-vepfs/siwei/coig/hf/reward_model/Llama-3.1-Nemotron-70B-Reward-HF" x
    # "/map-vepfs/siwei/coig/hf/Llama-3-OffsetBias-RM-8B"
    # "/map-vepfs/siwei/coig/hf/RM-Mistral-7B"
    # "/map-vepfs/siwei/coig/hf/ArmoRM-Llama3-8B-v0.1"
    # "/map-vepfs/siwei/coig/hf/Skywork-Reward-Llama-3.1-8B-v0.2"
    #===
    #/map-vepfs/siwei/coig/hf/reward_model/Skywork-Critic-Llama-3.1-70B #Skywork-Critic-Llama-3.1-70B
    #/map-vepfs/siwei/coig/hf/reward_model/Skywork/Skywork-Critic-Llama-3.1-8B #Skywork/Skywork-Critic-Llama-3.1-8B
    # /map-vepfs/siwei/coig/hf/reward_model/CompassJudger-1-14B-Instruct #CompassJudger-1-14B-Instruct 
    #/map-vepfs/siwei/coig/hf/reward_model/CompassJudger-1-32B-Instruct #CompassJudger-1-32B-Instruct
    # "/map-vepfs/huggingface/models/Qwen2.5-72B-Instruct"

    args = parser.parse_args()

    from datasets import load_dataset

    data = load_dataset("SiweiWu/COIG-CRBench")['test']

    if args.score_pattern == 'Generative':
        if args.model_name == 'Qwen2.5-72B-Instruct':
            model_path = "/map-vepfs/huggingface/models/Qwen2.5-72B-Instruct"
        elif args.model_name == 'Skywork-Critic-Llama-3.1-70B':
            model_path = "/map-vepfs/siwei/coig/hf/reward_model/Skywork-Critic-Llama-3.1-70B"
        elif args.model_name == 'Skywork-Critic-Llama-3.1-8B':
            model_path = "/map-vepfs/siwei/coig/hf/reward_model/Skywork/Skywork-Critic-Llama-3.1-8B"
        elif args.model_name == 'CompassJudger-1-14B-Instruct':
            model_path = "/map-vepfs/siwei/coig/hf/reward_model/CompassJudger-1-14B-Instruct"
        elif args.model_name == 'CompassJudger-1-32B-Instruct':
            model_path = "/map-vepfs/siwei/coig/hf/reward_model/CompassJudger-1-32B-Instruct"

        llm = LLM(
            # model="/map-vepfs/models/Qwen2.5-72B-Instruct/", 
            model=model_path,
            dtype="auto",
            trust_remote_code=True,
            tensor_parallel_size= 2,
            max_model_len=3000,
            gpu_memory_utilization=1,
            # enforce_eager=True,
        )
        sampling_params = SamplingParams(temperature=0.0, max_tokens = 512, top_p = 0.9, repetition_penalty = 1.1)
    elif args.score_pattern == 'Discriminative':
        device = "cuda"
        if args.model_name == 'CRM':
            path = "/map-vepfs/siwei/coig/alignment-handbook/RLHF-Reward-Modeling/models/llama3_rm/last_checkpoint"
        elif args.model_name == 'Skywork-Reward-Gemma-2-27B-v0.2':
            path = '/map-vepfs/siwei/coig/hf/reward_model/Skywork-Reward-Gemma-2-27B-v0.2'
        elif args.model_name == 'Llama-3-OffsetBias-RM-8B':
            path = '/map-vepfs/siwei/coig/hf/Llama-3-OffsetBias-RM-8B'
        elif args.model_name == "RM-Mistral-7B":
            path = "/map-vepfs/siwei/coig/hf/RM-Mistral-7B"
        elif args.model_name == "ArmoRM-Llama3-8B-v0.1":
            path = "/map-vepfs/siwei/coig/hf/ArmoRM-Llama3-8B-v0.1"
        elif args.model_name == "Skywork-Reward-Llama-3.1-8B-v0.2":
            path = "/map-vepfs/siwei/coig/hf/Skywork-Reward-Llama-3.1-8B-v0.2"
        model = AutoModelForSequenceClassification.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            # attn_implementation="flash_attention_2",
            num_labels=1,
        )
        tokenizer = AutoTokenizer.from_pretrained(path)
    
    model_answer_data = []
    for item in tqdm(data):
        p = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']
        
        if args.score_pattern == 'Generative':
            if args.model_name == 'Skywork-Critic-Llama-3.1-8B' or args.model_name == 'Skywork-Critic-Llama-3.1-70B':
                prompt_template = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user\'s instructions and answers the user\'s question better. 
                        Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. 
                        Please directly output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better."""
                
                user_message = prompt_template.format(input=p, response_a=chosen, response_b=rejected)
                prompt = user_message
            else:
                prompt = f"Give you a question: {p} \n\n The response A : {chosen} \n\n The response B : {rejected} \n\n Please help me choose which response is better. Give me your answer in the \\boxed{{}}, just A or B."
            outputs = llm.generate(prompt, sampling_params)
            answer = outputs[0].outputs[0].text
            answer = extract_boxed_content(answer)
            item['answer'] = answer
        elif args.score_pattern == 'Discriminative':
            scores = calculate_scores(model, tokenizer, p, chosen, rejected, args)
            chosen_score = scores['chosen_score']
            rejected_score = scores['rejected_score']
            if chosen_score > rejected_score:
                item['answer'] = ['A']
            else:
                item['answer'] = ['B']
        
        model_answer_data.append(item)
    
    json_save(model_answer_data, f'./model_answer_data_{args.model_name}.json')