import json
import sys
import argparse
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from utils.common import write_jsonl_lines, print_info
from tenacity import RetryError

from data_loader import load_data
from models import load_model, infer
# from utils.common import write_jsonl_lines, print_info
from post_process.custom_post_processor import PostProcessorRegistry
from config.config_wrapper import initialize_config, get_config_wrapper

def check_completed(output_file):
    completed = {}
    no_response_id = []
    try:
        with open(output_file, 'r') as file:
            for line in file:
                data = json.loads(line)
                response_key = config_wrapper.response_key
                error_key = config_wrapper.error_key
                if response_key in data and (isinstance(data[response_key], str) 
                                        or (isinstance(data[response_key], dict) and error_key not in data[response_key]) 
                                        or data.get(config_wrapper.status_key, None) not in ['processing', 'error', None]):
                    completed[config_wrapper.get_id(data)] = data
                else:
                    no_response_id.append(config_wrapper.get_id(data))
    except FileNotFoundError:
        pass
    except json.JSONDecodeError:
        pass
    return completed, no_response_id

def infer_batch(model_components, model_name, batch):
    results = []
    prompts, historys = [sample[config_wrapper.prompt_key] for sample in batch], [sample.get(config_wrapper.history_key, {}) for sample in batch]
    try:
        # print('====prompts====')
        # print(prompts)
        # print('====historys====')
        # print(historys)
        responses = infer(model_name)(prompts, historys, **model_components)
        for sample, response in zip(batch, responses):
            sample[config_wrapper.response_key] = response
            results.append(sample)
    except RetryError as e:
        last_attempt = e.last_attempt
        if last_attempt:
            exception = last_attempt.exception()
            if exception:
                # print(f"Error processing {prompts}: {str(exception)}", file=sys.stderr)
                print(f"Error: {str(exception)}")
                for sample in batch:
                    sample[config_wrapper.response_key] = {"error": str(exception)}
                    results.append(sample)
    except Exception as e:
        # print(f"Error processing {prompts}: {str(e)}", file=sys.stderr)
        print(f"Error: {str(e)}")
        for sample in batch:
            sample[config_wrapper.response_key] = {"error": str(e)}
            results.append(sample)
    return results

def main(model_name='gpt4o', splits=[], modes=[], output_dir='results', infer_limit=None, num_workers=1, batch_size=1, use_accel=False):
    info = {
        'model_name': model_name,
        'splits': splits,
        'modes': modes,
        'output_dir': output_dir,
        'infer_limit': infer_limit,
        'num_workers': num_workers,
        'batch_size': batch_size,
        'use_accel': use_accel
    }
    print_info(info)
    model_components = None
    
    os.makedirs(output_dir, exist_ok=True)
    # for split in splits:
    for mode in modes:
        processor = PostProcessorRegistry.get_processor(mode)
        config_wrapper.mode = mode
        # config_wrapper.split = split
        output_file_path = f'{output_dir}/{model_name}_AlignBench.jsonl'
        temp_output_file_path = f'{output_file_path}.tmp'
        
        completed, _ = check_completed(output_file_path)
        temp_completed, _ = check_completed(temp_output_file_path)
        # print(f'Found {len(completed)} completed inferences for {split} {mode} mode.')
        # print(completed)
        merged = {**temp_completed, **completed}
        # print(f'Found {len(merged)} completed inferences for {split} {mode} mode.')
        infer_count = 0

        with open(temp_output_file_path, 'w') as temp_file:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                batch = []

                def process_batch(batch):
                    futures.append(executor.submit(infer_batch, model_components, model_name, batch))

                # for prompt, sample in tqdm(load_data(split=split, mode=mode), desc=f'Processing {split} {mode} data'):
                for prompt, sample in tqdm(load_data(split="Nothing", mode=mode), desc=f'Processing "Nothing" {mode} data'):
                    sample[config_wrapper.prompt_key] = prompt
                    if config_wrapper.get_id(sample) in merged:
                        sample = merged[config_wrapper.get_id(sample)]
                        write_jsonl_lines(temp_file, sample)
                        continue
                    if infer_limit is not None and infer_count >= infer_limit:
                        break
                    if model_components is None:
                        model_components = load_model(model_name, use_accel)
                    batch.append(sample)
                    infer_count += 1
                    if len(batch) == batch_size:
                        process_batch(batch)
                        batch = []
                    if infer_limit is not None and infer_count >= infer_limit:
                        break

                if batch:
                    process_batch(batch)

                def process_results(futures):
                    batch_to_return = []
                    for future in tqdm(as_completed(futures), total=len(futures), desc=f'Processing Nothing {mode} results'):
                        results = future.result()
                        if processor:
                            results_to_save, results_to_return = processor.process(results)
                            for result in results_to_save:
                                write_jsonl_lines(temp_file, result)
                            batch_to_return.extend(results_to_return)
                        else:
                            for result in results:
                                write_jsonl_lines(temp_file, result)
                    return batch_to_return

                batch_to_return = process_results(futures)
                futures = []
                
                while batch_to_return:
                    while batch_to_return:
                        new_batch = list(batch_to_return[:min(batch_size, len(batch_to_return))])
                        batch_to_return = list(batch_to_return[min(batch_size, len(batch_to_return)):])
                        process_batch(new_batch)
                    batch_to_return = process_results(futures)
                    futures = []
        
        # Only rename the temp file to the final output file if the entire process completes successfully
        shutil.move(temp_output_file_path, output_file_path)
        _, no_response_id = check_completed(output_file_path)
        if len(no_response_id) > 0:
            print(f"Failed to get response for {len(no_response_id)} questions in {mode} mode. IDs: {no_response_id}", file=sys.stderr)
    # print(f'Inference for {split} completed.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference and save results.')
    parser.add_argument('--model_name', type=str, default='', help='Model name to use')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file to use')
    parser.add_argument('--split', nargs='+', default=[], help='Data split to use')
    parser.add_argument('--mode', nargs='+', default=[], help='Modes to use for data loading, separated by space')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to write results')
    parser.add_argument('--infer_limit', type=int, help='Limit the number of inferences per run, default is no limit', default=None)
    parser.add_argument('--num_workers', type=int, default=1, help='Number of concurrent workers for inference, currently only used for API')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference, currently only used for local model processing')
    parser.add_argument('--use_accel', action='store_true', help='Use inference acceleration framework for inference, LLM-->vLLM, VLM-->lmdeploy')
    parser.add_argument('--save_prompt', action='store_true', help='Save prompt to output file')
    args = parser.parse_args()
    initialize_config(args.config)
    config_wrapper = get_config_wrapper()
    main(model_name=args.model_name, splits=args.split, modes=args.mode, output_dir=args.output_dir, infer_limit=args.infer_limit, num_workers=args.num_workers, batch_size=args.batch_size, use_accel=args.use_accel)
