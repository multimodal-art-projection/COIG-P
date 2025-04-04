import json
import os
from utils.common import read_yaml, read_json_or_jsonl, read_json_or_jsonl_with_idx
    
# Load the data
def load_data(split='', mode=''):
    samples =read_json_or_jsonl("/map-vepfs/siwei/coig/KOR-Bench/data/", 'data_v1.1_release', 'idx')

    # print(len(samples))
    # input()

    for item in samples:
        question = item['question']
        yield question, item

    # if split in ['cipher', 'logic', 'operation', 'puzzle', 'counterfactual'] and mode in ['zero-shot', 'three-shot', 'trick', 'self-correction', 'self-correction-with-needle']:
    #     rule = read_json_or_jsonl(f'data/{split}', 'rule', 'idx')
    #     sample = read_json_or_jsonl(f'data/{split}', 'sample')
    #     few_shot = read_json_or_jsonl(f'data/{split}', 'three-shot') if mode == 'three-shot' else []
    #     config = f'{mode}'
    #     if mode == 'self-correction' or mode == 'self-correction-with-needle':
    #         config = 'zero-shot'
    #     template = read_yaml(config)
    #     for s in sample:
    #         rule_id = s['rule_id']
    #         rule_content = rule[rule_id]['rule_content']
    #         question = s['question']

    #         if config in ['zero-shot', 'zero-shot-cot']:
    #             prompt_format = [rule_content, question]
    #             prompt = template[f'{split}_prompt_format'][0].format(*prompt_format)
    #         elif config == 'three-shot':
    #             few_shot_qa = [item for fs in few_shot if fs['rule_id'] == rule_id for item in [fs['question'], fs['answer']]]
    #             prompt_format = [rule_content, *few_shot_qa ,question]
    #             prompt = template[f'{split}_prompt_format'][0].format(*prompt_format)
    #         elif config == 'trick' and split == 'puzzle':
    #             prompt_format = [rule_content, question, s['trick']]
    #             prompt = template[f'prompt_format'][0].format(*prompt_format)

    #         s['title'] = rule[rule_id].get('title', '')
    #         s['tag'] = rule[rule_id].get('tag', '')
    #         s['rule_content'] = rule_content
    #         yield prompt, s
    
    # elif split == 'mixed' and mode in ['Multi-Q', 'Multi-R', 'Multi-RQ']:
    #     mixed_data = read_json_or_jsonl(f'data/{split}/', mode)
    #     template = read_yaml('mixed')
    #     for item in mixed_data:
    #         rule_list = item['rule_list']
    #         question_list = item['question_list']
    #         rule_content_list = []
    #         question_content_list = []
            
    #         for rule in rule_list:
    #             category, rule_idx = rule.rsplit('_', 1)
    #             rule_content = read_json_or_jsonl_with_idx(f'data/{category}', 'rule', idx=rule_idx)
    #             rule_content_list.append(rule_content['rule_content'])
                
    #         for question in question_list:
    #             category, question_idx = question.rsplit('_', 1)
    #             question_content = read_json_or_jsonl_with_idx(f'data/{category}', 'sample', idx=question_idx)
    #             question_content_list.append(question_content['question'])

    #         rules_str = '\n'.join(f'{rule}\n' for rule in rule_content_list)
    #         questions_str = '\n'.join(f'question{idx+1}:\n{question}\n' for idx, question in enumerate(question_content_list))
    #         prompt_format = [rules_str, questions_str]
    #         prompt = template['prompt_format'][0].format(*prompt_format)
            
    #         yield prompt, item
    
    # elif split == 'cipher' and mode == 'subquestions':
    #     rule = read_json_or_jsonl(f'data/{split}', 'rule', 'idx')
    #     sample = read_json_or_jsonl(f'data/{split}', 'subquestions')
    #     config = 'zero-shot'
    #     template = read_yaml(config)
    #     for s in sample:
    #         rule_id = s['rule_id']
    #         rule_content = rule[rule_id]['rule_content']
    #         input = s['input']  
    #         for detail in s['steps_details']:
    #             item = {}
    #             for key, value in s.items():
    #                 if key != 'steps_details':
    #                     item[key] = value
    #             description = detail['description']
    #             subquestion = input + '\n\n' + description
    #             prompt_format = [rule_content, subquestion]
    #             prompt = template[f'{split}_prompt_format'][0].format(*prompt_format)

    #             for key, value in detail.items():
    #                 item[key] = value

    #             yield prompt, item

if __name__ == '__main__':
    last_prompt = None

    for prompt, sample in load_data('cipher', 'subquestions'):
        last_prompt = prompt
        # print(sample)

    if last_prompt is not None:
        print(last_prompt)
        

