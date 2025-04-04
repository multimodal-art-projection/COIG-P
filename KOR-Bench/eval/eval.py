from eval_utils import evaluate_responses
from utils.common import read_json_or_jsonl
import os
import json
import csv
import sys
import argparse
from prettytable import PrettyTable

MIXED_MODES = ["Multi-Q", "Multi-R", "Multi-RQ"]

def evaluate_all_files_in_folder(folder_path, output_folder, csv_file):
    os.makedirs(output_folder, exist_ok=True)
    model_scores = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".jsonl"):
            print(f"Processing {filename}...")
            parts = os.path.basename(filename).split('_')
            model_name = parts[0]
            question_type = parts[1]
            mode = parts[-1].replace('.jsonl', '')
            if mode in MIXED_MODES:
                question_type = mode
                mode = 'mixed'
            print(question_type, mode)
            data = read_json_or_jsonl(folder_path, filename)
            evaluation_results = evaluate_responses(data, question_type, mode)

            output_file = os.path.join(output_folder, f"evaluation_{filename}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=4)

            correct_count = sum(result['is_correct'] for result in evaluation_results)
            count = len(evaluation_results)
            accuracy = (correct_count / count) * 100 if count > 0 else 0

            if question_type == "counterfactual":
                real_life_count = sum(result['is_real_life'] for result in evaluation_results)
                real_life_rate = (real_life_count / count) * 100 if count > 0 else 0
            if question_type in MIXED_MODES:
                total_pass_rate = sum(result['pass_rate'] for result in evaluation_results)
                pass_rate = (total_pass_rate / count) * 100 if count > 0 else 0

            # Store results in a nested dictionary for each model and mode
            key = (model_name, mode)
            if key not in model_scores:
                model_scores[key] = {}
            model_scores[key][question_type] = {
                'correct': correct_count,
                'total': count,
                'accuracy': accuracy,
                'real_life_count': real_life_count if question_type == "counterfactual" else None,
                'real_life_rate': real_life_rate if question_type == "counterfactual" else None,
                'pass_rate': pass_rate if question_type in MIXED_MODES else None
            }

            # Print individual file results
            print(f"Processed {filename}: Total Correct - {correct_count} out of {count}, Accuracy - {accuracy:.2f}%" +
                  (f", Pass Rate - {pass_rate:.2f}%" if question_type == "mixed" else "") +
                  (f", Real Life Count - {real_life_count}, Real Life Rate - {real_life_rate:.2f}%" if question_type == "counterfactual" else ""))

    # Aggregate results and write to CSV
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['model_name', 'mode', 'total_correct', 'total_count', 'overall_accuracy']
        question_types = set(qt for scores in model_scores.values() for qt in scores)
        for qt in sorted(question_types):
            if qt in MIXED_MODES and 'overall_pass_rate' not in fieldnames:
                fieldnames.append('overall_pass_rate')
            fieldnames.extend([f'{qt}_correct', f'{qt}_total', f'{qt}_accuracy'])
            if qt == "counterfactual":
                fieldnames.extend([f'{qt}_real_life_count', f'{qt}_real_life_rate'])
            if qt in MIXED_MODES:
                fieldnames.extend([f'{qt}_pass_rate'])
        print(fieldnames)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        table = PrettyTable()
        table.field_names = fieldnames

        for (model_name, mode), scores in model_scores.items():
            total_correct = sum(details['correct'] for details in scores.values())
            total_count = sum(details['total'] for details in scores.values())
            overall_accuracy = (total_correct / total_count) * 100 if total_count > 0 else 0
            if mode == 'mixed':
                total_pass_rate = sum(details['pass_rate'] for details in scores.values()) / len(scores)
                overall_pass_rate = sum(details['pass_rate'] for details in scores.values()) / len(scores) if len(scores) > 0 else 0
            row = {
                'model_name': model_name,
                'mode': mode,
                'total_correct': total_correct,
                'total_count': total_count,
                'overall_accuracy': f"{overall_accuracy:.2f}%"
            }
            if mode == 'mixed':
                row['overall_pass_rate'] = f"{overall_pass_rate:.2f}%"

            for question_type, details in scores.items():
                row[f'{question_type}_correct'] = details['correct']
                row[f'{question_type}_total'] = details['total']
                row[f'{question_type}_accuracy'] = f"{details['accuracy']:.2f}%"
                if question_type == "counterfactual":
                    row[f'{question_type}_real_life_count'] = details['real_life_count']
                    row[f'{question_type}_real_life_rate'] = f"{details['real_life_rate']:.2f}%"
                if question_type in MIXED_MODES:
                    row[f'{question_type}_pass_rate'] = f"{details['pass_rate']:.2f}%"
            print(row)
            writer.writerow(row)
            table.add_row([row[field] for field in fieldnames])
            # Print summarized results
            print(f"Model: {model_name}, Mode: {mode}, Total Correct: {total_correct}, Total: {total_count}, Overall Accuracy: {overall_accuracy:.2f}%" +
                  (f", Overall Pass_rate: {overall_pass_rate:.2f}%" if question_type == "mixed" else "") +
                  (f", Real Life Count - {real_life_count}, Real Life Rate - {real_life_rate:.2f}%" if question_type == "counterfactual" else ""))
    print(table)
if __name__ == "__main__":
    # Create parser
    parser = argparse.ArgumentParser(description='Evaluate JSONL files and generate a summary CSV file.')

    # Adding arguments
    parser.add_argument('source_folder', type=str, help='Path to the folder containing JSONL files for evaluation.')
    parser.add_argument('target_root_folder', type=str, help='Path to the folder where output JSON files and the CSV will be stored.')
    parser.add_argument('csv_file', type=str, help='Path to the output CSV file that will store the aggregated results.')

    # Parse arguments
    args = parser.parse_args()

    # Call the function with these parameters
    evaluate_all_files_in_folder(args.source_folder, args.target_root_folder, args.csv_file)


