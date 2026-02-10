"""
utils.py
========
Shared enumerations, dataclasses, and helper functions used across the
GSM8K multi-path reasoning system.

Includes:
- Enums: LogicalOperation, QuestionType, ComplexityLevel, ValidationStatus, ReasoningApproach
- Dataclasses: LogicalStep, ReasoningPath, SynthesizedAnswer, NegotiationResult
- QuestionClassification container
- Numerical answer extraction and comparison utilities
"""

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# ============================================================================
# ENUMS
# ============================================================================

class LogicalOperation(Enum):
    VERDICT = "verdict"
    PREMISE = "premise"
    INFERENCE = "inference"
    EVIDENCE = "evidence"
    CONCLUSION = "conclusion"
    COUNTERARGUMENT = "counterargument"


class QuestionType(Enum):
    BINARY = "binary"
    FACTUAL = "factual"
    MATHEMATICAL = "mathematical"
    ANALYTICAL = "analytical"
    HYPOTHETICAL = "hypothetical"
    PROCEDURAL = "procedural"
    COMMONSENSE = "commonsense"


class ComplexityLevel(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class ValidationStatus(Enum):
    NOT_VALIDATED = "not_validated"
    VALID = "valid"
    INVALID = "invalid"
    UNCERTAIN = "uncertain"


class ReasoningApproach(Enum):
    ANALYTICAL = "analytical"
    SKEPTICAL = "skeptical"
    EVIDENCE_BASED = "evidence_based"
    ALGEBRAIC = "algebraic"
    NUMERICAL = "numerical"
    GEOMETRIC = "geometric"
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    PROOF_BY_CONTRADICTION = "proof_by_contradiction"


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class LogicalStep:
    """A single step within a reasoning process."""
    id: str
    operation: LogicalOperation
    content: str
    confidence: float = 0.8
    validation_status: ValidationStatus = ValidationStatus.NOT_VALIDATED
    validation_feedback: Optional[str] = None
    is_mathematical: bool = False
    calculation_verified: bool = False


@dataclass
class ReasoningPath:
    """A structured reasoning path from query to conclusion."""
    path_id: str
    query: str
    verdict: Optional[str] = None
    steps: List[LogicalStep] = field(default_factory=list)
    conclusion: str = ""
    confidence: float = 0.0
    generation_strategy: str = "base"
    raw_output: str = ""
    generation_time: float = 0.0
    question_type: QuestionType = QuestionType.BINARY
    complexity_level: ComplexityLevel = ComplexityLevel.MODERATE
    answer: Optional[str] = None
    validation_passes: int = 0
    regeneration_count: int = 0

    def to_readable_chain(self) -> str:
        """Convert the reasoning path to a human-readable string."""
        if not self.steps and not self.verdict and not self.answer:
            return f"Reasoning for: {self.query}\n[Generation failed]\n"

        chain = f"Strategy: {self.generation_strategy.upper()}\n"
        chain += f"Type: {self.question_type.value} | Complexity: {self.complexity_level.value}\n"
        chain += f"Query: {self.query}\n\n"

        if self.question_type == QuestionType.BINARY and self.verdict:
            chain += f"VERDICT: {self.verdict}\n\n"
        elif self.answer:
            chain += f"ANSWER: {self.answer}\n\n"

        if self.steps:
            chain += "Reasoning:\n"
            for i, step in enumerate(self.steps, 1):
                marker = ""
                if step.validation_status == ValidationStatus.VALID:
                    marker = " ✓"
                elif step.validation_status == ValidationStatus.INVALID:
                    marker = " ✗"
                chain += f"{i}. [{step.operation.value.upper()}]{marker} {step.content}\n"
                if step.validation_feedback:
                    chain += f"   Validation: {step.validation_feedback}\n"

        chain += f"\nConclusion: {self.conclusion}\n"
        chain += f"Confidence: {self.confidence:.2f}\n"

        if self.validation_passes > 0:
            chain += f"Validation passes: {self.validation_passes}\n"
        if self.regeneration_count > 0:
            chain += f"Regenerations: {self.regeneration_count}\n"

        chain += f"Generation time: {self.generation_time:.2f}s\n"
        return chain


@dataclass
class SynthesizedAnswer:
    """Combines multiple reasoning paths into a unified final answer."""
    query: str
    definitive_answer: str
    supporting_reasoning: List[str]
    conflicting_points: List[str]
    final_confidence: float
    synthesis_explanation: str
    question_type: QuestionType = QuestionType.BINARY
    answer_format: str = "verdict"


class QuestionClassification:
    """Container for question classification metadata."""
    def __init__(self, question, question_type, complexity_level, requires_validation,
                 requires_math_verification, suggested_approaches, confidence_threshold, num_paths):
        self.question = question
        self.question_type = question_type
        self.complexity_level = complexity_level
        self.requires_validation = requires_validation
        self.requires_math_verification = requires_math_verification
        self.suggested_approaches = suggested_approaches
        self.confidence_threshold = confidence_threshold
        self.num_paths = num_paths


@dataclass
class NegotiationResult:
    """Encapsulates all results from the reasoning–synthesis pipeline."""
    original_paths: List[ReasoningPath]
    synthesized_answer: SynthesizedAnswer
    total_time: float
    parallel_speedup: float
    total_cost: float
    total_validations: int = 0
    total_regenerations: int = 0
    classification: Optional[QuestionClassification] = None


# ============================================================================
# ANSWER EXTRACTION & COMPARISON UTILITIES
# ============================================================================

def extract_numerical_answer(text: str) -> Optional[float]:
    """
    Extract a numerical answer from text.
    Handles: integers, decimals, fractions, percentages, currency.
    """
    if not text:
        return None

    text = text.lower().strip()
    text = re.sub(r'^(the answer is|answer:|final answer:)\s*', '', text, flags=re.IGNORECASE)

    # Currency: $42.50
    currency_match = re.search(r'\$?\s*([\d,]+\.?\d*)', text)
    if currency_match:
        try:
            return float(currency_match.group(1).replace(',', ''))
        except ValueError:
            pass

    # Percentage: 42.5%
    pct_match = re.search(r'([\d.]+)\s*%', text)
    if pct_match:
        try:
            return float(pct_match.group(1))
        except ValueError:
            pass

    # Fraction: 3/4
    frac_match = re.search(r'(\d+)\s*/\s*(\d+)', text)
    if frac_match:
        try:
            return float(frac_match.group(1)) / float(frac_match.group(2))
        except (ValueError, ZeroDivisionError):
            pass

    # Plain number
    num_match = re.search(r'([-+]?\d*\.?\d+)', text)
    if num_match:
        try:
            return float(num_match.group(1))
        except ValueError:
            pass

    return None


def compare_numerical_answers(predicted: str, ground_truth: str, tolerance: float = 0.01) -> bool:
    """
    Compare two answer strings numerically with a relative tolerance.

    Args:
        predicted: Model's answer text.
        ground_truth: Correct answer text.
        tolerance: Relative tolerance (default 1%).

    Returns:
        True if answers match within tolerance.
    """
    pred_num = extract_numerical_answer(predicted)
    truth_num = extract_numerical_answer(ground_truth)

    if pred_num is None or truth_num is None:
        return predicted.strip().lower() == ground_truth.strip().lower()

    if pred_num == truth_num:
        return True

    if truth_num == 0:
        return abs(pred_num) < tolerance

    return abs(pred_num - truth_num) / abs(truth_num) <= tolerance
