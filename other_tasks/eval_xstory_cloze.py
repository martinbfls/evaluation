from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
import time
import csv
import os
import numpy as np
import argparse
import pandas as pd
import re

def parse_args():
    parser = argparse.ArgumentParser(description='xStory Cloze evaluation on a single GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number to use')
    parser.add_argument('--max_samples', type=int, default=None, 
                        help='Maximum number of examples to evaluate (if None, evaluate all)')
    parser.add_argument('--test_file', type=str, 
                       default='/srv/home/users/beaufilsm35cs/evaluation/xstory_cloze/test/cloze_test_test__winter2018-cloze_test_ALL_test - 1.csv',
                       help='Path to test CSV file')
    parser.add_argument('--val_file', type=str, 
                       default='/srv/home/users/beaufilsm35cs/evaluation/xstory_cloze/val/cloze_test_val__winter2018-cloze_test_ALL_val - 1 - 1.csv',
                       help='Path to validation CSV file')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val'], 
                        help='Dataset split to evaluate (default: test)')
    parser.add_argument('--model_path', type=str, default="CohereLabs/aya-101", 
                        help='Path or model name to evaluate')
    parser.add_argument('--local_model', default=False, action='store_true', 
                        help='Use local model files only')
    parser.add_argument('--results_csv', type=str, default='results/xstory_cloze_results.csv', 
                        help='Path to CSV file to save results')
    return parser.parse_args()

def construct_story(row):
    """Build the story from the first 4 sentences"""
    story = []
    for i in range(1, 5):
        sentence = row[f'InputSentence{i}']
        if pd.notna(sentence):
            story.append(sentence)
    return " ".join(story)

def eval_example(row, tokenizer, model, device):
    story = construct_story(row)
    
    ending1 = row['RandomFifthSentenceQuiz1']
    ending2 = row['RandomFifthSentenceQuiz2']
    
    correct_ending_idx = 1
    if 'AnswerRightEnding' in row and not pd.isna(row['AnswerRightEnding']):
        correct_ending_idx = int(row['AnswerRightEnding'])
    
    correct_ending = ending1 if correct_ending_idx == 1 else ending2
    
    input_text = (
        f"Given this story: {story}\n"
        f"Which is the most logical ending?\n"
        f"A) {ending1}\n"
        f"B) {ending2}\n"
        f"Answer with 'A' or 'B'."
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
    
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    if re.search(r'\bA\b', prediction) or prediction.lower().find('option a') >= 0:
        pred_ending_idx = 1
        pred_ending = ending1
    elif re.search(r'\bB\b', prediction) or prediction.lower().find('option b') >= 0:
        pred_ending_idx = 2
        pred_ending = ending2
    else:
        similarity_1 = compute_similarity(prediction, ending1)
        similarity_2 = compute_similarity(prediction, ending2)
        pred_ending_idx = 1 if similarity_1 > similarity_2 else 2
        pred_ending = ending1 if pred_ending_idx == 1 else ending2
    
    is_correct = (pred_ending_idx == correct_ending_idx)
    
    return {
        'story': story,
        'ending1': ending1,
        'ending2': ending2,
        'correct_ending_idx': correct_ending_idx,
        'correct_ending': correct_ending,
        'prediction': prediction,
        'pred_ending_idx': pred_ending_idx,
        'pred_ending': pred_ending,
        'correct': is_correct
    }

def compute_similarity(text1, text2):
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def main(
    gpu=0, 
    max_samples=None, 
    test_file='/srv/home/users/beaufilsm35cs/evaluation/xstory_cloze/test/cloze_test_test__winter2018-cloze_test_ALL_test - 1.csv', 
    val_file='/srv/home/users/beaufilsm35cs/evaluation/xstory_cloze/val/cloze_test_val__winter2018-cloze_test_ALL_val - 1 - 1.csv', 
    split='test', 
    model_path="CohereLabs/aya-101", 
    local_model=False, 
    results_csv='results/xstory_cloze_results.csv'
):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu}')
        torch.cuda.set_device(device)
        print(f"Using GPU {gpu}: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("No GPU available. Running on CPU.")
    
    data_file = test_file if split == 'test' else val_file
    print(f"Loading data from {data_file}...")
    
    try:
        df = pd.read_csv(data_file)
        print(f"Total examples: {len(df)}")
        if max_samples is not None and max_samples < len(df):
            df = df.sample(max_samples, random_state=42)
            print(f"Limiting evaluation to {max_samples} examples")
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return
    
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    results_file = results_csv
    file_exists = os.path.exists(results_file)
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=local_model)
    model.to(device)
    model.eval()
    
    if not file_exists:
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['split', 'accuracy', 'elapsed_seconds', 'examples_count', 'gpu_id'])
    
    start_time = time.time()
    results = []
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"GPU {gpu} - {split}"):
        try:
            result = eval_example(row, tokenizer, model, device)
            results.append(result)
            
            if not result['correct'] and len(results) <= 3:
                print(f"\nIncorrect example #{len(results)}:")
                print(f"Story: {result['story']}")
                print(f"Option A: {result['ending1']}")
                print(f"Option B: {result['ending2']}")
                print(f"Prediction: {result['prediction']}")
                print(f"Predicted option: {result['pred_ending_idx']} (Choice {'A' if result['pred_ending_idx'] == 1 else 'B'})")
                print(f"Correct option: {result['correct_ending_idx']} (Choice {'A' if result['correct_ending_idx'] == 1 else 'B'})")
        except Exception as e:
            print(f"Error evaluating example {i}: {str(e)}")
            continue
    
    accuracy = sum(r['correct'] for r in results) / len(results) if results else 0
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([split, accuracy, elapsed_time, len(df), gpu])
    
    print(f"\nResults for {split}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Elapsed time: {elapsed_time:.2f}s")
    print(f"Number of evaluated examples: {len(results)}")
    
    return accuracy, elapsed_time

if __name__ == "__main__":
    args = parse_args()
    main(
        gpu=args.gpu,
        max_samples=args.max_samples,
        test_file=args.test_file,
        val_file=args.val_file,
        split=args.split,
        model_path=args.model_path,
        local_model=args.local_model,
        results_csv=args.results_csv
    )
