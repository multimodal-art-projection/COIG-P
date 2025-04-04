from utils import load_json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="config for Chinese Reward Bench")
    parser.add_argument("--model_name" , type = str, default = 'CRM')
    args = parser.parse_args()

    model_name = args.model_name

    # data = load_json('./model_answer_data_GPT-4o.json')
    # data = load_json('./model_answer_data_claude-3-5-sonnet-20240620.json')

    data = load_json(f'./model_answer_data_{args.model_name}.json')

    domain_result = {}
    domain_count = {}

    count = 0
    result = 0

    for item in data:
        answer = item['answer']

        if answer:
            answer = answer[0]
        else:
            answer = 'B'
        
        domain = item['domain']#[0]

        if domain not in domain_result:
            domain_result[domain] = 0
            domain_count[domain] = 0
        domain_count[domain] += 1
        count += 1


        if answer == 'A':
            domain_result[domain] += 1
            result += 1
        
    
    for d in domain_result:
        domain_result[d] = domain_result[d] / domain_count[d]
    
    dimensions = ['对话',	'逻辑推理',	'数学'	,'代码',	'角色扮演',	'小说续写']
    for d in dimensions:
        print(d, ': ', domain_result[d], domain_count[d])
    
    print('average: ', result / count)