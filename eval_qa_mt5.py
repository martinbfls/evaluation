import os
import argparse
import time
import csv
import json
import gc

import torch
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import evaluate
from scipy.stats import ttest_rel
import multiprocessing as mp


torch.manual_seed(42)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_examples', type=int, default=None)
    parser.add_argument('--model_path', type=str, default="CohereLab/aya-101")
    parser.add_argument('--results_csv', type=str, default='qa_results.csv')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--coreset_list', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='results_distributed/qa')
    parser.add_argument(
        '--metrics',
        type=str,
        default="bleu,rouge1,rouge2,rougel,loss",
        help="Comma-separated list of metrics to compute"
    )
    return parser.parse_args()


args = parse_args()

selected_metrics = set(m.strip().lower() for m in args.metrics.split(','))

bleu_metric = evaluate.load("bleu") if 'bleu' in selected_metrics else None
rouge_metric = evaluate.load("rouge") if any(m in selected_metrics for m in ['rouge1', 'rouge2', 'rougel']) else None

data = load_from_disk(args.data_path)

def compute_loss(question, answer, tokenizer, model, device):
    input_text = f"question: {question}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(answer, return_tensors="pt", truncation=True).to(device)

    outputs = model(**inputs, labels=labels["input_ids"])
    loss = outputs.loss.item()
    return loss

def compute_bleu(reference, prediction):
    if bleu_metric is None:
        return None
    return bleu_metric.compute(predictions=[prediction], references=[[reference]])["bleu"]


def compute_rouge(reference, prediction):
    if rouge_metric is None:
        return (None, None, None)
    result = rouge_metric.compute(predictions=[prediction], references=[reference])
    return result.get('rouge1'), result.get('rouge2'), result.get('rougeL')

def evaluate_qa(example, tokenizer, model, device):
    question = example['inputs']
    gold_answer = example['targets']

    input_text = f"{question}"
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512).to(device)

    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=4,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )
        generation_time = time.time() - start_time

    predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    results = {}

    if 'bleu' in selected_metrics:
        results['bleu'] = compute_bleu(gold_answer, predicted_answer)

    if any(m in selected_metrics for m in ['rouge1', 'rouge2', 'rougel']):
        rouge1, rouge2, rougel = compute_rouge(gold_answer, predicted_answer)
        if 'rouge1' in selected_metrics:
            results['rouge1'] = rouge1
        if 'rouge2' in selected_metrics:
            results['rouge2'] = rouge2
        if 'rougel' in selected_metrics:
            results['rougel'] = rougel

    if 'loss' in selected_metrics:
        loss = compute_loss(question, gold_answer, tokenizer, model, device)
        if 'loss' in selected_metrics:
            results['loss'] = loss

    return results


def evaluate_model(model_path, subset_name=None, device=None):
    print(f"\nEvaluating {model_path} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device).eval()

    eval_data = data
    if args.max_examples:
        indices = torch.randperm(len(data))[:args.max_examples].tolist()
        eval_data = data.select(indices)

    metrics = {metric: [] for metric in selected_metrics}

    for example in tqdm(eval_data, desc=f"Evaluating {subset_name or 'model'}"):
        scores = evaluate_qa(example, tokenizer, model, device)
        for key in metrics:
            if key in scores and scores[key] is not None:
                metrics[key].append(scores[key])

    results = {f'avg_{k}': np.mean(v) if len(v) > 0 else None for k, v in metrics.items()}
    results['count'] = len(eval_data)
    results['scores_per_sample'] = metrics

    del model, tokenizer

    return metrics, results

def save_results(results_selected, results_random, ttests, pvals, coreset):
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    output_path = os.path.join(args.output_dir, args.results_csv.replace('.csv', f'_{coreset}.csv'))

    fieldnames = ['model_type']
    fieldnames += [f'avg_{metric}' for metric in selected_metrics]
    fieldnames += [f't_stat_{metric}' for metric in selected_metrics]
    fieldnames += [f'p_val_{metric}' for metric in selected_metrics]
    fieldnames.append('examples_count')

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model_type, res in [('selected', results_selected), ('random', results_random)]:
            row = {'model_type': model_type}
            for metric in selected_metrics:
                row[f'avg_{metric}'] = res.get(f'avg_{metric}', None)
                row[f't_stat_{metric}'] = ttests.get(metric, None)
                row[f'p_val_{metric}'] = pvals.get(metric, None)
            row['examples_count'] = res['count']
            writer.writerow(row)

    print(f"Results saved to {output_path}")

def save_results_single(results, subset_name, coreset):
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"results_{subset_name}_coreset-{coreset}.json")
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)


def run_evaluation(model_path, subset_name, coreset, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"\n=== Evaluating {subset_name} for coreset {coreset} on GPU {gpu_id} ===")
    print(f"Model path: {model_path}")
    metrics, results = evaluate_model(model_path, subset_name=subset_name, device=device)

    print(f"\nResults for {subset_name} - Coreset {coreset}:")
    for metric in selected_metrics:
        avg = results.get(f'avg_{metric}', None)
        if avg is not None:
            print(f"{metric.capitalize()}: {avg:.4f}")

    save_results_single(results, subset_name, coreset)

    return metrics


def main():
    if args.coreset_list:
        coresets = [c.strip() for c in args.coreset_list.split(',')]
        print(f"Running evaluation for coresets: {coresets}")

        processes = []

        for coreset in coresets:
            model_selected = f"{args.model_path}_coreset-{coreset}"
            model_random = f"{args.model_path}-random_coreset-{coreset}"
            if coreset == 1.0 or coreset == '1.0':
                model_selected = f'{args.model_path}_final'
                model_random = f'{args.model_path}-random_final'
            
            print(model_selected, model_random)
            
            # Process for selected on GPU 0
            p_sel = mp.Process(
                target=run_evaluation,
                args=(model_selected, 'selected', coreset, 0)
            )
            p_sel.start()
            processes.append(p_sel)

            # Process for random on GPU 1
            p_rand = mp.Process(
                target=run_evaluation,
                args=(model_random, 'random', coreset, 1)
            )
            p_rand.start()
            processes.append(p_rand)

            gc.collect()
            torch.cuda.empty_cache()

        for p in processes:
            p.join()

        for coreset in coresets:
            path_selected = os.path.join(args.output_dir, f"results_selected_coreset-{coreset}.json")
            path_random = os.path.join(args.output_dir, f"results_random_coreset-{coreset}.json")

            with open(path_selected) as f:
                results_selected = json.load(f)

            with open(path_random) as f:
                results_random = json.load(f)

            ttests = {}
            pvals = {}

            for metric in selected_metrics:
                selected_scores = results_selected['scores_per_sample'][metric]
                random_scores = results_random['scores_per_sample'][metric]

                assert len(selected_scores) == len(random_scores), \
                    f"Scores length mismatch for {metric} in coreset {coreset}"
                
                direction = 'greater' if metric in ['bleu', 'rouge1', 'rouge2', 'rougel'] else 'less'
                t_stat, p_val = ttest_rel(selected_scores, random_scores, alternative=direction)
                ttests[metric] = t_stat
                pvals[metric] = p_val

            save_results(results_selected, results_random, ttests, pvals, coreset)


    else:
        _, results = evaluate_model(args.model_path)

        print("\nEvaluation results:")
        for metric in selected_metrics:
            avg_val = results.get(f'avg_{metric}', None)
            if avg_val is not None:
                if metric == 'generation_time':
                    print(f"Average {metric.replace('_', ' ').capitalize()}: {avg_val:.2f}s")
                else:
                    print(f"Average {metric.replace('_', ' ').capitalize()}: {avg_val:.4f}")


if __name__ == "__main__":
    main()
