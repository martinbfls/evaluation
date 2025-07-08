from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
import re
import time
import csv
import os
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='XCOPA evaluation on a single GPU for all languages')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use (device number)')
    parser.add_argument('--langs', type=str, default=None, help='Languages to evaluate (comma separated). If not specified, all languages will be evaluated.')
    parser.add_argument('--max_samples', type=int, default=None, 
                        help='Maximum number of samples to evaluate per language (if None, all samples are evaluated)')
    parser.add_argument('--model_path', type=str, default="CohereLabs/aya-101", 
                        help='Path or name of the model to evaluate')
    parser.add_argument('--local_model', default=False, action='store_true',
                        help='Use local model files only')
    parser.add_argument('--results_csv', type=str, default='results/xcopa_results_single.csv', 
                        help='Path to the CSV file where results will be saved')
    return parser.parse_args()

def eval_example(example, tokenizer, model, device):
    def normalize(text):
        return re.sub(r'\W+', '', text.lower().strip())

    premise = example['premise']
    question = example['question']
    choice1 = example['choice1']
    choice2 = example['choice2']
    label = example['label']
    
    input_text = (
        f"xcopa: premise: {premise}; question: {question}; choice1: {choice1}; choice2: {choice2}; answer with choice1 or choice2:"
    )

    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=10,
            num_beams=4,
            no_repeat_ngram_size=2,
            early_stopping=True,
            do_sample=True,
            temperature=0.7
        )
    
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    if 'choice1' in prediction:
        pred_choice = 'A'
    elif 'choice2' in prediction:
        pred_choice = 'B'
    elif prediction.lower() in choice1.lower() or choice1.lower() in prediction.lower():
        pred_choice = 'A'
    elif prediction.lower() in choice2.lower() or choice2.lower() in prediction.lower():
        pred_choice = 'B'
    else:
        a_similarity = len(set(normalize(prediction)) & set(normalize(choice1))) / (len(set(normalize(choice1))) + 1e-10)
        b_similarity = len(set(normalize(prediction)) & set(normalize(choice2))) / (len(set(normalize(choice2))) + 1e-10)
        pred_choice = 'A' if a_similarity > b_similarity else 'B'

    correct_choice = "A" if label == 0 else "B"
    is_correct = pred_choice == correct_choice

    return {
        'premise': premise,
        'question': question,
        'choice1': choice1,
        'choice2': choice2,
        'label': label,
        'prediction': prediction,
        'pred_choice': pred_choice,
        'correct_choice': correct_choice,
        'correct': is_correct
    }

def main(
    gpu=0, 
    langs=None, 
    max_samples=None, 
    model_path="CohereLabs/aya-101", 
    local_model=False, 
    results_csv='results/xcopa_results_single.csv'
):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu}')
        torch.cuda.set_device(device)
        print(f"Using GPU {gpu}: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("No GPU available. Running on CPU.")
    
    dataset_path = 'cambridgeltl/xcopa'
    
    all_languages = ['et', 'ht', 'id', 'it', 'qu', 'sw', 'ta', 'th', 'tr', 'vi', 'zh']
    if langs:
        languages = langs.split(',') if isinstance(langs, str) else langs
        for lang in languages:
            if lang not in all_languages:
                print(f"Warning: Unknown language '{lang}'. Available languages are: {', '.join(all_languages)}")
    else:
        languages = all_languages
    
    print(f"Languages to evaluate: {', '.join(languages)}")
    
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    results_file = results_csv
    
    file_exists = os.path.exists(results_file)
    
    print(f"Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=local_model)
    model.to(device)
    model.eval()
    
    if not file_exists:
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['language', 'accuracy', 'time_seconds', 'gpu_id'])
    
    all_results = []
    
    for lang in languages:
        lang_start_time = time.time()
        print(f"\nEvaluating language {lang}")
        
        try:
            dataset = load_dataset(dataset_path, lang)['test']
            
            if max_samples and max_samples < len(dataset):
                print(f"Total number of samples for {lang}: {len(dataset)}")
                print(f"Limiting evaluation to {max_samples} samples")
                dataset = dataset.select(range(max_samples))
            
            results = []
            first_wrong_printed = True
            for i, example in enumerate(tqdm(dataset, desc=f"GPU {gpu} - {lang}")):
                result = eval_example(example, tokenizer, model, device)
                results.append(result)
            
                if not result['correct'] and first_wrong_printed:
                    first_wrong_printed = False
                    print(f"Example ({lang}): {result['premise']}")
                    print(f"Prediction: {result['prediction']}")
                    print(f"Pred Choice: {result['pred_choice']}, Correct: {result['correct_choice']}")
                    print(f'choice1: {result["choice1"]}, choice2: {result["choice2"]}')
            
            accuracy = sum(r['correct'] for r in results) / len(results)
            lang_end_time = time.time()
            elapsed_time = lang_end_time - lang_start_time
            
            with open(results_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([lang, accuracy, elapsed_time, gpu])
            
            print(f"Accuracy ({lang}): {accuracy:.4f} (time: {elapsed_time:.2f}s)")
            
            all_results.append({
                'language': lang,
                'accuracy': accuracy,
                'time': elapsed_time
            })
            
        except Exception as e:
            print(f"Error while evaluating language {lang}: {str(e)}")
    
    print("\nSummary of results:")
    total_accuracy = 0
    total_time = 0
    for result in all_results:
        print(f"Language: {result['language']}, Accuracy: {result['accuracy']:.4f}, Time: {result['time']:.2f}s")
        total_accuracy += result['accuracy']
        total_time += result['time']
    
    if all_results:
        avg_accuracy = total_accuracy / len(all_results)
        print(f"\nAverage accuracy: {avg_accuracy:.4f}")
        print(f"Total execution time: {total_time:.2f}s")
    
    return all_results

if __name__ == "__main__":
    args = parse_args()
    main(
        gpu=args.gpu,
        langs=args.langs,
        max_samples=args.max_samples,
        model_path=args.model_path,
        local_model=args.local_model,
        results_csv=args.results_csv
    )
