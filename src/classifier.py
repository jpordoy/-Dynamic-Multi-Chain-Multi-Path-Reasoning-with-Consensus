"""
classifier.py
=============
Question type and complexity classification for the GSM8K multi-path
reasoning system.

Classes:
- QuestionClassifier: Determines question type, complexity, number of
  reasoning paths to generate, and suggested reasoning approaches.
"""

import re
from typing import List

from src.utils import (
    QuestionClassification,
    QuestionType,
    ComplexityLevel,
    ReasoningApproach,
)


class QuestionClassifier:
    """
    Classifies incoming questions by type and complexity to guide path
    generation strategy.

    Key behaviours:
    - GSM8K-style math word problems always receive 3 paths (algebraic,
      numerical, analytical) with math verification enabled.
    - CommonsenseQA questions always receive 3 paths to break ties.
    - Complexity is inferred from linguistic cues.
    """

    def classify(self, query: str) -> QuestionClassification:
        """
        Classify question and return a QuestionClassification object.

        Args:
            query: The raw question text (may include multiple-choice options).

        Returns:
            A QuestionClassification with type, complexity, path count, etc.
        """
        # Extract only the actual question — ignore multiple-choice option lines
        query_lines = query.split('\n')
        actual_question = query_lines[0]
        for line in query_lines:
            if any(stop in line.lower() for stop in ['please select', 'options:', 'your answer must']):
                break
            actual_question = line

        query_lower = actual_question.lower()
        print(f"[CLASSIFIER] Extracted question: {query_lower[:80]}...")

        question_type = self._determine_type(query_lower, actual_question)
        complexity_level = self._determine_complexity(query_lower, question_type)

        # ── Path count ──────────────────────────────────────────────────────
        if question_type == QuestionType.COMMONSENSE:
            num_paths = 3
            print("[CLASSIFIER] COMMONSENSE → Forcing 3 paths for consensus")
        else:
            num_paths = (
                1 if complexity_level == ComplexityLevel.SIMPLE
                else 3 if complexity_level == ComplexityLevel.EXPERT
                else 2
            )

        # Math always uses 3 paths + forces verification
        if question_type == QuestionType.MATHEMATICAL:
            num_paths = 3
            print("[CLASSIFIER] MATH → Forcing 3 paths (algebraic + numerical + analytical)")

        requires_validation = (
            complexity_level in {ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT}
            or question_type == QuestionType.MATHEMATICAL
        )
        requires_math_verification = (question_type == QuestionType.MATHEMATICAL)

        suggested_approaches = self._suggest_approaches(question_type, complexity_level)
        confidence_threshold = self._get_confidence_threshold(complexity_level)

        print(f"[CLASSIFIER] → Type: {question_type.value}, "
              f"Complexity: {complexity_level.value}, Paths: {num_paths}")

        return QuestionClassification(
            query,
            question_type,
            complexity_level,
            requires_validation,
            requires_math_verification,
            suggested_approaches,
            confidence_threshold,
            num_paths,
        )

    # ── Type determination ───────────────────────────────────────────────────

    def _determine_type(self, query_lower: str, query: str) -> QuestionType:
        """
        Detect question type, safely distinguishing GSM8K math word problems
        from commonsense/factual questions.
        """

        # 1. GSM8K-style math word problems
        gsm8k_patterns = [
            r'\d+\s+(eggs?|apples?|oranges?|books?|pages?|dollars?|cents?|hours?|minutes?|days?|weeks?)',
            r'\d+\s+(per|each|every|total|altogether|remainder|left|remained|sells?|buys?)',
            r'(how many|how much)\s+.*\?',
            r'\d+\s*[\+\-\*\/]\s*\d+',
            r'(half|twice|double|triple|quarter)\s+(of\s+)?\d+',
        ]
        if any(re.search(p, query_lower) for p in gsm8k_patterns):
            if not any(w in query_lower for w in ['typically', 'usually', 'where would', 'what do people']):
                print("[CLASSIFIER] MATH WORD PROBLEM (GSM8K-style)")
                return QuestionType.MATHEMATICAL

        # 2. Strict symbolic / algebraic math
        strict_keywords = [
            'solve', 'equation', 'calculate', 'integral', 'derivative',
            'quadratic', 'factor', 'simplify', 'prove mathematically', 'find x'
        ]
        if any(kw in query_lower for kw in strict_keywords):
            print("[CLASSIFIER] MATH KEYWORD")
            return QuestionType.MATHEMATICAL

        strict_patterns = [
            r'\d+x\s*[\+\-\*/]', r'x\^', r'\b(sin|cos|tan|log)\s*\(', r'√\d',
            r'\d+\s*[\+\-\*/]\s*\d+\s*='
        ]
        if any(re.search(p, query) for p in strict_patterns):
            print("[CLASSIFIER] MATH REGEX")
            return QuestionType.MATHEMATICAL

        # 3. Commonsense cues
        commonsense_cues = [
            'typically', 'usually', 'commonly', 'often', 'likely',
            'where would you', 'what do people usually',
            'what might', 'who might', 'where might',
            'good place to', 'if you wanted to', 'after he', 'after she'
        ]
        if any(cue in query_lower for cue in commonsense_cues):
            print("[CLASSIFIER] COMMONSENSE CUE")
            return QuestionType.COMMONSENSE

        # 4. Factual
        if any(query_lower.startswith(st) for st in ['what is', 'who is', 'where is', 'when was']):
            print("[CLASSIFIER] FACTUAL")
            return QuestionType.FACTUAL

        # 5. Binary
        if (any(query_lower.startswith(st) for st in ['is ', 'are ', 'does ', 'do ', 'can ', 'will '])
                and query.endswith('?')):
            print("[CLASSIFIER] BINARY")
            return QuestionType.BINARY

        # 6. Analytical
        if any(kw in query_lower for kw in ['explain', 'why', 'how does', 'compare']):
            print("[CLASSIFIER] ANALYTICAL")
            return QuestionType.ANALYTICAL

        # 7. Default: numbers → math, else commonsense
        if re.search(r'\d', query):
            print("[CLASSIFIER] DEFAULT → MATHEMATICAL (has numbers)")
            return QuestionType.MATHEMATICAL

        print("[CLASSIFIER] DEFAULT → COMMONSENSE")
        return QuestionType.COMMONSENSE

    # ── Complexity determination ─────────────────────────────────────────────

    def _determine_complexity(self, query_lower: str,
                               question_type: QuestionType) -> ComplexityLevel:
        if any(ind in query_lower for ind in ['prove', 'theorem', 'paradox']):
            return ComplexityLevel.EXPERT
        if any(ind in query_lower for ind in ['system', 'quadratic', 'monty hall']):
            return ComplexityLevel.COMPLEX
        if any(ind in query_lower for ind in ['capital', 'simple']):
            return ComplexityLevel.SIMPLE
        return ComplexityLevel.MODERATE

    # ── Strategy suggestions ─────────────────────────────────────────────────

    def _suggest_approaches(self, question_type: QuestionType,
                            complexity: ComplexityLevel) -> List[ReasoningApproach]:
        if question_type == QuestionType.MATHEMATICAL:
            return [
                ReasoningApproach.ALGEBRAIC,
                ReasoningApproach.NUMERICAL,
                ReasoningApproach.ANALYTICAL,
            ]

        if question_type == QuestionType.COMMONSENSE:
            return [
                ReasoningApproach.ANALYTICAL,
                ReasoningApproach.EVIDENCE_BASED,
                ReasoningApproach.SKEPTICAL,
            ]

        if question_type == QuestionType.BINARY:
            return [ReasoningApproach.ANALYTICAL, ReasoningApproach.SKEPTICAL]

        return [ReasoningApproach.ANALYTICAL, ReasoningApproach.EVIDENCE_BASED]

    # ── Confidence thresholds ────────────────────────────────────────────────

    def _get_confidence_threshold(self, complexity: ComplexityLevel) -> float:
        return {
            ComplexityLevel.SIMPLE: 0.50,
            ComplexityLevel.MODERATE: 0.65,
            ComplexityLevel.COMPLEX: 0.70,
            ComplexityLevel.EXPERT: 0.75,
        }[complexity]
