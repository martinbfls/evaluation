import argparse
import os
import time
from datetime import datetime
import torch
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from other_tasks.eval_xwinograd import main as xwinograd_main, parse_args as xwinograd_parse_args
from other_tasks.eval_xcopa import main as xcopa_main, parse_args as xcopa_parse_args
from other_tasks.eval_xnli import main as xnli_main, parse_args as xnli_parse_args
from other_tasks.eval_xstory_cloze import main as xstory_cloze_main, parse_args as xstory_cloze_parse_args
from unused.eval_qa_aya import main as qa_main, parse_args as qa_parse_args

def parser():
    parser = argparse.ArgumentParser(description="Run all evaluations for XWinograd, XCOPA, XNLI, and XStory Cloze.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to evaluate.")
    parser.add_argument("--local_model", default=False, action='store_true', help="Whether to use a local model or download from Hugging Face.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the evaluation on.")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save evaluation results.")
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum number of samples to evaluate per dataset.")
    parser.add_argument("--rate_list", type=str, default=None, help="Liste des taux à évaluer, séparés par des virgules (ex: '0.1,0.2,0.3')")
    parser.add_argument("--run_qa", default=False, action='store_true', help="Exécuter l'évaluation QA")

    return parser.parse_args()

def main():
    args = parser()
    
    if args.rate_list:
        rates = [rate.strip() for rate in args.rate_list.split(',')]
        print(f"Evaluation for rates: {rates}")
        
        for rate in rates:
            print(f"\n=== Evaluation for rate:  {rate} ===")
            current_model_path = f"{args.model_name}_{rate}"
            
            # XWinograd
            current_xwinograd_results = os.path.join(args.output_dir, f'xwinograd_results_{rate}.csv')
            print(f"XWinograd - Model: {current_model_path}, Results: {current_xwinograd_results}")
            xwinograd_main(
                gpu=0, 
                langs=None,
                split='test', 
                max_samples=args.max_samples, 
                model_path=current_model_path, 
                local_model=args.local_model, 
                results_csv=current_xwinograd_results
            )
            
            # XCOPA
            current_xcopa_results = os.path.join(args.output_dir, f'xcopa_results_{rate}.csv')
            print(f"XCOPA - Model: {current_model_path}, Results: {current_xcopa_results}")
            xcopa_main(
                gpu=0, 
                langs=None, 
                max_samples=args.max_samples, 
                model_path=current_model_path, 
                local_model=args.local_model, 
                results_csv=current_xcopa_results
            )
            
            # XNLI
            current_xnli_results = os.path.join(args.output_dir, f'xnli_results_{rate}.csv')
            print(f"XNLI - Model: {current_model_path}, Results: {current_xnli_results}")
            xnli_main(
                gpu=0, 
                langs="ar,bg,de,el", 
                split='test', 
                max_examples=args.max_samples, 
                model_path=current_model_path, 
                local_model=args.local_model, 
                results_csv=current_xnli_results
            )
            
            # XStory Cloze
            current_xstory_results = os.path.join(args.output_dir, f'xstory_cloze_results_{rate}.csv')
            print(f"XStory Cloze - Model: {current_model_path}, Results: {current_xstory_results}")
            xstory_cloze_main(
                gpu=0,
                split='test', 
                max_samples=args.max_samples, 
                model_path=current_model_path, 
                local_model=args.local_model, 
                results_csv=current_xstory_results
            )
    else:
        xwinograd_main(
            gpu=0, 
            langs=None,
            split='test', 
            max_samples=args.max_samples, 
            model_path=args.model_name, 
            local_model=args.local_model, 
            results_csv=os.path.join(args.output_dir, 'xwinograd_results.csv')
        )

        xcopa_main(
            gpu=0, 
            langs=None, 
            max_samples=args.max_samples, 
            model_path=args.model_name, 
            local_model=args.local_model, 
            results_csv=os.path.join(args.output_dir, 'xcopa_results.csv')
        )

        xnli_main(
            gpu=0, 
            langs="ar,bg,de,el", 
            split='test', 
            max_examples=args.max_samples, 
            model_path=args.model_name, 
            local_model=args.local_model, 
            results_csv=os.path.join(args.output_dir, 'xnli_results.csv')
        )

        xstory_cloze_main(
            gpu=0,
            split='test', 
            max_samples=args.max_samples, 
            model_path=args.model_name, 
            local_model=args.local_model, 
            results_csv=os.path.join(args.output_dir, 'xstory_cloze_results.csv')
        )

if __name__ == "__main__":

    main()
