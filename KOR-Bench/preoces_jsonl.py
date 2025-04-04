import json

def load_jsonl(path):
# 打开并逐行读取 .jsonl 文件
    with open(path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    return data

def save_json(data, path):
    # 保存为 JSON 文件
    with open(path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == '__main__':
    # path = '/map-vepfs/siwei/coig/KOR-Bench/results/Infinity-Instruct-3M-0625-Qwen2-7B_AlignBench.jsonl'#Qwen2.5-7B_AlignBench.jsonl
    # path = '/map-vepfs/siwei/coig/KOR-Bench/results/Qwen2.5-7B-Instruct_AlignBench.jsonl'#Qwen2.5-7B_AlignBench.jsonl
    # path = '/map-vepfs/siwei/coig/KOR-Bench/results/Infinity-Instruct-3M-0625-Qwen2-7B-dpo-lr5e-5-beta0.1_AlignBench.jsonl'#Qwen2.5-7B_AlignBench.jsonl
    # path =  "/map-vepfs/siwei/coig/KOR-Bench/results/Infinity-Instruct-3M-0625-Qwen2-7B-dpo-lr_1e-6_beta_0.05_AlignBench.jsonl"
    # path =  "/map-vepfs/siwei/coig/KOR-Bench/results/Infinity-Instruct-3M-0625-Qwen2-7B-dpo-lr_5e-6_beta_0.05_AlignBench.jsonl"
    # path =  "/map-vepfs/siwei/coig/KOR-Bench/results/Infinity-Instruct-3M-0625-Qwen2-7B-dpo-lr_5e-6_beta_0.1_AlignBench.jsonl"
    # path =   "/map-vepfs/siwei/coig/KOR-Bench/results/Infinity-Instruct-3M-0625-Qwen2-7B-dpo-lr_5e-6_beta_0.05_AlignBench.jsonl"
    # path = "/map-vepfs/siwei/coig/KOR-Bench/results/Infinity-Instruct-3M-0625-Qwen2-7B-dpo-lr_1e-6_beta_0.05_AlignBench.jsonl"
    # path = "/map-vepfs/siwei/coig/KOR-Bench/results/Qwen2.5-Instrruct-dpo-lr_1e-6_beta_0.5_AlignBench.jsonl"
    # path = "/map-vepfs/siwei/coig/KOR-Bench/results/Qwen2.5-7B-Instruct_AlignBench.jsonl"
    path = "/map-vepfs/siwei/coig/KOR-Bench/results/Qwen2.5-Instrruct-dpo-lr_1e-6_beta_0.1_sft_0.1_AlignBench.jsonl"
    path = "/map-vepfs/siwei/coig/KOR-Bench/results/Qwen2.5-7B-Instruct_lora_dpo_lr5e-5_beta_0.1_AlignBench.jsonl"
    path = "/map-vepfs/siwei/coig/KOR-Bench/results/Qwen2.5-7B-Instruct_lora_dpo_lr1e-6_beta_0.1_AlignBench.jsonl"
    path = "/map-vepfs/siwei/coig/KOR-Bench/results/Qwen2.5-7B-Instruct_lora_dpo_lr5e-6_beta_0.05_AlignBench.jsonl"
    path = "/map-vepfs/siwei/coig/KOR-Bench/results/Qwen2.5-7B-Instruct_lora_dpo_lr5e-6_beta_0.3_AlignBench.jsonl"
    path = '/map-vepfs/siwei/coig/KOR-Bench/results/Infinity-Instruct-3M-0625-Qwen2-7B_lora_dpo_lr5e-6_beta_0.1_AlignBench.jsonl'
    path = "/map-vepfs/siwei/coig/KOR-Bench/results/Qwen2.5-7B-Instruct_lr1e-6_beta_0.1_chat_small_AlignBench.jsonl"
    path = "/map-vepfs/siwei/coig/KOR-Bench/results/Qwen2.5_inst_achat_small_lr_5e-7_beta_0.5_AlignBench.jsonl.tmp"
    path = "/map-vepfs/siwei/coig/KOR-Bench/results/dpo_chat_small_lr_5e-7_beta_0.1_sft0.1_AlignBench.jsonl.tmp"
    path = "/map-vepfs/siwei/coig/KOR-Bench/results/Qwen2.5-7B-Instruct_AlignBench.jsonl.tmp"
    # path = "/map-vepfs/siwei/coig/KOR-Bench/results/Previous_Qwen2.5-7B-Instruct_AlignBench.jsonl"
    samples = load_jsonl(path)


    for item in samples:
        question = item['question']
        answer = item['answer']
        reference = item['reference']

        print('==question===')
        print(question)
        print('==answer===')
        print(answer)
        print('==reference===')
        print(reference)
        print(item.keys())

        print('====over=====')
        input()

    save_json(samples, '/map-vepfs/siwei/coig/KOR-Bench/results/Infinity-Instruct-3M-0625-Qwen2-7B_AlignBench.json')

    print("文件已保存。")


    
    # print(len(data))

    # for item in data:
    #     print(item.keys())
    #     input()
    #     idx = item['idx']
    #     question = item['question']
    #     answer = item['answer']
    #     category = item['category']
    #     rule_id = item['rule_id']
    #     needle = item['needle']
    #     tag = item['tag']
    #     rule_content = item['rule_content']
    #     prompt = item['prompt']
    #     response = item['response']

    #     print(rule_id)
    #     print(rule_content)
    #     input()
    # print(type(data))


