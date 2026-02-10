"""
evaluate.py
===========
Main GSM8K evaluation harness.

Usage:
    python evaluate.py --num_questions 100          # Quick test
    python evaluate.py --num_questions 1000         # Full run
    python evaluate.py --num_questions 100 --verbose

Requires:
    pip install anthropic datasets python-dotenv tqdm
    ANTHROPIC_API_KEY set in environment or .env file
"""

import argparse
import json
import os
import re
import time
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

from src.multi_path import ClaudeDynamicReasoningSystem
from src.utils import compare_numerical_answers, extract_numerical_answer

# ============================================================================
# CONFIGURATION
# ============================================================================

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
HAIKU_MODEL = "claude-3-5-haiku-20241022"
MAX_TOKENS = 800           # Higher token limit for multi-step math
FORCE_NO_VALIDATION = False  # Keep validation ON — important for accuracy
DEFAULT_QUESTION_CAP = 100


# ============================================================================
# SYSTEM SETUP
# ============================================================================

def build_system() -> ClaudeDynamicReasoningSystem:
    """Initialise and return the reasoning system."""
    if not API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. "
            "Export it in your shell or add it to a .env file."
        )
    return ClaudeDynamicReasoningSystem(api_key=API_KEY)


def optimize_system_for_math(system: ClaudeDynamicReasoningSystem) -> ClaudeDynamicReasoningSystem:
    """
    Configure the system specifically for GSM8K mathematical reasoning:
    - Switch model to Claude 3.5 Haiku.
    - Enforce a per-call token cap.
    - Optionally disable validation for speed.
    """
    print("\n" + "="*80)
    print("APPLYING MATH REASONING OPTIMISATIONS")
    print("="*80)

    print(f"Model: {HAIKU_MODEL}")
    system.generator.model = HAIKU_MODEL
    system.synthesizer.model = HAIKU_MODEL

    print(f"Token limit: {MAX_TOKENS}")
    original_create_gen = system.generator.client.messages.create
    original_create_synth = system.synthesizer.client.messages.create

    def make_limited(original):
        def wrapper(*args, **kwargs):
            kwargs['max_tokens'] = min(kwargs.get('max_tokens', 1500), MAX_TOKENS)
            return original(*args, **kwargs)
        return wrapper

    system.generator.client.messages.create = make_limited(original_create_gen)
    system.synthesizer.client.messages.create = make_limited(original_create_synth)

    if FORCE_NO_VALIDATION:
        original_classify = system.generator.classifier.classify
        def no_validation_classify(query):
            cls = original_classify(query)
            cls.requires_validation = False
            return cls
        system.generator.classifier.classify = no_validation_classify

    print(f"Validation: {'DISABLED' if FORCE_NO_VALIDATION else 'ENABLED'}")
    print(f"Math verification: ENABLED")
    print("="*80 + "\n")
    return system


# ============================================================================
# MAIN TEST HARNESS — GSM8K
# ============================================================================

def run_math_test(system: ClaudeDynamicReasoningSystem,
                  question_cap: int = DEFAULT_QUESTION_CAP,
                  verbose: bool = False) -> List[dict]:
    """
    Evaluate the system on the GSM8K test split.

    Args:
        system: Initialised ClaudeDynamicReasoningSystem.
        question_cap: Maximum number of questions to answer.
        verbose: If True, print per-question reasoning detail.

    Returns:
        List of per-question result dicts.
    """
    from datasets import load_dataset

    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")
    questions = dataset['test']
    print(f"Dataset has {len(questions)} test questions.")
    print(f"Will answer up to {question_cap} questions.\n")

    results = []
    start_time = time.time()
    answered_count = 0

    for idx, question_data in enumerate(questions):
        if answered_count >= question_cap:
            break

        question = question_data['question']
        full_answer = question_data['answer']
        ground_truth = full_answer.split('####')[-1].strip()

        print(f"\n{'='*80}")
        print(f"QUESTION #{answered_count+1}/{question_cap}")
        print(f"{'='*80}")
        print(f"Q: {question}")
        print(f"Ground Truth: {ground_truth}")

        try:
            result = system.reason_with_synthesis(question)

            if verbose:
                system.display_result(result)

            answer = result.synthesized_answer.definitive_answer
            confidence = result.synthesized_answer.final_confidence
            cost = result.total_cost
            is_correct = compare_numerical_answers(answer, ground_truth)
            predicted_num = extract_numerical_answer(answer)

            print(f"\nModel Answer:   {answer}")
            print(f"Parsed Number:  {predicted_num}")
            print(f"Truth Number:   {extract_numerical_answer(ground_truth)}")
            print(f"Correct:        {is_correct}")
            print(f"Confidence: {confidence:.2f} | Cost: ${cost:.4f} | "
                  f"Time: {result.total_time:.1f}s")

            results.append({
                'question_num': answered_count + 1,
                'dataset_idx': idx,
                'question': question,
                'ground_truth': ground_truth,
                'full_ground_truth': full_answer,
                'model_answer': answer,
                'predicted_number': predicted_num,
                'ground_truth_number': extract_numerical_answer(ground_truth),
                'is_correct': is_correct,
                'confidence': confidence,
                'cost': cost,
                'time': result.total_time,
                'paths_generated': len(result.original_paths),
                'validations': result.total_validations,
                'regenerations': result.total_regenerations,
                'question_type': (result.classification.question_type.value
                                  if result.classification else None),
                'complexity': (result.classification.complexity_level.value
                               if result.classification else None),
            })

            answered_count += 1

            # Intermediate save every 25 questions
            if answered_count % 25 == 0:
                save_results(results, answered_count, final=False)

        except Exception as e:
            print(f"\nERROR on question #{answered_count+1}: {e}")
            print("Skipping and continuing...")
            continue

    total_time = time.time() - start_time
    save_results(results, answered_count, final=True, total_time=total_time)
    print_summary(results, answered_count)
    return results


# ============================================================================
# RESULTS PERSISTENCE
# ============================================================================

def save_results(results: List[dict], count: int,
                 final: bool = True, total_time: Optional[float] = None) -> None:
    """Save evaluation results to JSON in the Results/ directory."""
    os.makedirs('Results', exist_ok=True)
    suffix = "_FINAL" if final else f"_{count}q"
    output_path = f'Results/gsm8k_results{suffix}.json'

    correct_count = sum(1 for r in results if r['is_correct'])
    accuracy = (correct_count / len(results) * 100) if results else 0
    total_cost = sum(r['cost'] for r in results)
    avg_cost = total_cost / len(results) if results else 0
    avg_time = total_time / len(results) if (total_time and results) else 0
    total_validations = sum(r['validations'] for r in results)
    total_regenerations = sum(r['regenerations'] for r in results)
    avg_paths = sum(r['paths_generated'] for r in results) / len(results) if results else 0

    output = {
        'metadata': {
            'dataset': 'GSM8K',
            'description': 'Grade School Math 8K — Mathematical reasoning benchmark',
            'model': HAIKU_MODEL,
            'max_tokens': MAX_TOKENS,
            'validation_enabled': not FORCE_NO_VALIDATION,
            'questions_answered': count,
            'accuracy': round(accuracy, 2),
            'correct_count': correct_count,
            'total_cost_usd': round(total_cost, 4),
            'avg_cost_per_question_usd': round(avg_cost, 4),
            'total_time_seconds': round(total_time, 1) if total_time else None,
            'avg_time_per_question_seconds': round(avg_time, 1),
            'total_validations': total_validations,
            'total_regenerations': total_regenerations,
            'avg_paths_per_question': round(avg_paths, 2),
            'completed_at': time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        'results': results,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    status = "FINAL" if final else "INTERMEDIATE"
    print(f"\n{status} RESULTS SAVED → {output_path}")
    print(f"  Questions: {count} | Accuracy: {accuracy:.1f}% ({correct_count}/{count})")
    print(f"  Total Cost: ${total_cost:.4f} | Avg Time: {avg_time:.2f}s/q")


def print_summary(results: List[dict], count: int) -> None:
    """Print a concise accuracy and cost summary."""
    correct = sum(1 for r in results if r['is_correct'])
    accuracy = (correct / count * 100) if count else 0
    total_cost = sum(r['cost'] for r in results)

    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE: {count} QUESTIONS")
    print(f"{'='*80}")
    print(f"  Accuracy:   {accuracy:.1f}%  ({correct}/{count} correct)")
    print(f"  Total Cost: ${total_cost:.4f}")
    print(f"{'='*80}\n")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Claude 3.5 Haiku + multi-path reasoning on GSM8K."
    )
    parser.add_argument(
        '--num_questions', type=int, default=DEFAULT_QUESTION_CAP,
        help=f"Number of GSM8K test questions to evaluate (default: {DEFAULT_QUESTION_CAP})"
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help="Print full per-question reasoning chains"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    system = build_system()
    system = optimize_system_for_math(system)
    run_math_test(system, question_cap=args.num_questions, verbose=args.verbose)
