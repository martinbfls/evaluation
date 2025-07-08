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
    parser = argparse.ArgumentParser(description='XWinograd evaluation on a single GPU for all languages')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use (device number)')
    parser.add_argument('--langs', type=str, default=None, 
                        help='Languages to evaluate (comma-separated). If not specified, all languages will be evaluated.')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test', 'validation'], 
                        help='Dataset split to evaluate (default: test)')
    parser.add_argument('--max_samples', type=int, default=None, 
                        help='Maximum number of examples to evaluate per language (if None, all examples are evaluated)')
    parser.add_argument('--model_path', type=str, default="CohereLabs/aya-101", 
                        help='Path or model name to evaluate')
    parser.add_argument('--local_model', default=False, action='store_true', 
                        help='Use only local files for the model')
    parser.add_argument('--results_csv', type=str, default='results/xwinograd_results.csv', 
                        help='Path to the CSV file where results will be saved')
    return parser.parse_args()

def eval_example(example, tokenizer, model, device):
    """Evaluate a single example"""
    sentence = example['sentence']
    option1 = example['option1']
    option2 = example['option2']
    answer = int(example['answer'])
    
    input_text = (
        f"xwinograd: In the sentence '{sentence}', does the pronoun refer to '{option1}' or '{option2}'? Answer with '1' for '{option1}' or '2' for '{option2}'."
    )

    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=5,
            num_beams=4,
            no_repeat_ngram_size=2,
            early_stopping=True,
            do_sample=False 
        )
    
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
    
    if "1" in prediction or option1.lower() in prediction.lower():
        pred_answer = 1
    elif "2" in prediction or option2.lower() in prediction.lower():
        pred_answer = 2
    else:
        similarity1 = len(set(prediction.lower().split()) & set(option1.lower().split())) / (len(set(option1.lower().split())) + 1e-10)
        similarity2 = len(set(prediction.lower().split()) & set(option2.lower().split())) / (len(set(option2.lower().split())) + 1e-10)
        pred_answer = 1 if similarity1 > similarity2 else 2
    
    is_correct = pred_answer == answer

    return {
        'sentence': sentence,
        'option1': option1,
        'option2': option2,
        'answer': answer,
        'prediction': prediction,
        'pred_answer': pred_answer,
        'correct': is_correct
    }

def main(gpu=0, langs=None, split='test', max_samples=None, model_path="CohereLabs/aya-101", local_model=False, results_csv='results/xwinograd_results.csv'):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu}')
        torch.cuda.set_device(device)
        print(f"Using GPU {gpu}: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("No GPU available. Running on CPU.")
    
    dataset_path = 'Muennighoff/xwinograd'
    
    supported_languages = ['en', 'fr', 'jp', 'pt', 'ru', 'zh']
    
    if langs:
        languages = langs.split(',') if isinstance(langs, str) else langs
        valid_languages = []
        for lang in languages:
            if lang in supported_languages:
                valid_languages.append(lang)
            else:
                print(f"Warning: Language '{lang}' is not supported. Supported languages are: {', '.join(supported_languages)}")
        languages = valid_languages
    else:
        languages = supported_languages
    
    print(f"Languages to evaluate: {', '.join(languages)}")
    
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    results_file = results_csv
    
    file_exists = os.path.exists(results_file)
    
    print(f"Loading model {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=local_model)
    model.to(device)
    model.eval()
    
    if not file_exists:
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['language', 'split', 'accuracy', 'time_seconds', 'gpu_id', 'num_samples'])
    
    all_results = []
    
    for lang in languages:
        lang_start_time = time.time()
        print(f"\n{'='*50}")
        print(f"Evaluating language {lang}")
        print(f"{'='*50}")
        
        try:
            print(f"Loading XWinograd dataset ({split}) for {lang}...")
            try:
                dataset = load_dataset(dataset_path, lang)[split]
            except Exception as e:
                print(f"Error loading dataset for language {lang}: {str(e)}")
                continue
            
            if max_samples and max_samples < len(dataset):
                print(f"Total examples for {lang}: {len(dataset)}")
                print(f"Limiting evaluation to {max_samples} examples")
                dataset = dataset.select(range(max_samples))
            
            print(f"Number of examples to evaluate for {lang}: {len(dataset)}")
            
            results = []
            shown_example = False
            
            for i, example in enumerate(tqdm(dataset, desc=f"GPU {gpu} - {lang}")):
                result = eval_example(example, tokenizer, model, device)
                results.append(result)
                
                if not shown_example and ((len(results) > 10 and not result['correct']) or i == min(10, len(dataset) - 1)):
                    shown_example = True
                    print(f"\nExample ({lang}):")
                    print(f"Sentence: {result['sentence']}")
                    print(f"Option 1: {result['option1']}")
                    print(f"Option 2: {result['option2']}")
                    print(f"Correct answer: {result['answer']}")
                    print(f"Prediction: {result['prediction']}")
                    print(f"Predicted answer: {result['pred_answer']}")
                    print(f"Correct: {result['correct']}")
            
            accuracy = sum(r['correct'] for r in results) / len(results)
            lang_end_time = time.time()
            elapsed_time = lang_end_time - lang_start_time
            
            with open(results_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([lang, split, accuracy, elapsed_time, gpu, len(dataset)])
            
            print(f"Accuracy ({lang}): {accuracy:.4f} (time: {elapsed_time:.2f}s)")
            
            all_results.append({
                'language': lang,
                'split': split,
                'accuracy': accuracy,
                'time': elapsed_time,
                'num_samples': len(dataset)
            })
            
        except Exception as e:
            print(f"Error evaluating language {lang}: {str(e)}")
    
    print("\nSummary of results:")
    total_accuracy = 0
    total_time = 0
    total_samples = 0
    for result in all_results:
        print(f"Language: {result['language']}, Accuracy: {result['accuracy']:.4f}, Examples: {result['num_samples']}, Time: {result['time']:.2f}s")
        total_accuracy += result['accuracy'] * result['num_samples']
        total_time += result['time']
        total_samples += result['num_samples']
    
    if all_results:
        avg_accuracy = total_accuracy / total_samples
        print(f"\nWeighted average accuracy: {avg_accuracy:.4f}")
        print(f"Total execution time: {total_time:.2f}s")
    
    return all_results

if __name__ == "__main__":
    args = parse_args()
    main(
        gpu=args.gpu,
        langs=args.langs,
        split=args.split,
        max_samples=args.max_samples,
        model_path=args.model_path,
        local_model=args.local_model,
        results_csv=args.results_csv
    )
