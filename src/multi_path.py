"""
multi_path.py
=============
Parallel multi-path reasoning generation and the top-level system orchestrator.

Classes:
- ClaudeReasoningGenerator: Generates 2–3 parallel reasoning paths per
  question using Claude, with dynamic confidence scoring and answer extraction.
- ClaudeDynamicReasoningSystem: High-level pipeline orchestrator that ties
  together classification, generation, validation, and synthesis.
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import anthropic

from src.classifier import QuestionClassifier
from src.utils import (
    ComplexityLevel,
    LogicalOperation,
    LogicalStep,
    NegotiationResult,
    QuestionClassification,
    QuestionType,
    ReasoningApproach,
    ReasoningPath,
    SynthesizedAnswer,
    ValidationStatus,
)
from src.verifier import ConfidenceAssessor, MathematicalVerifier


# ============================================================================
# CLAUDE REASONING GENERATOR
# ============================================================================

class ClaudeReasoningGenerator:
    """
    Reasoning engine that interacts with Anthropic's Claude API to generate
    structured, multi-path reasoning chains for a given query.

    Improvements over a single-path baseline:
    - FIX 1: Dynamic confidence calculation (not hardcoded).
    - FIX 2: Answer extraction from CONCLUSION sections.
    - FIX 5: Robust step extraction with multiple numbering patterns.
    """

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-haiku-20241022"
        self.total_tokens_input = 0
        self.total_tokens_output = 0
        self.classifier = QuestionClassifier()
        self.confidence_assessor = ConfidenceAssessor()
        self.math_verifier = MathematicalVerifier()

    # ── Parallel path generation ─────────────────────────────────────────────

    def generate_multiple_paths_parallel(
        self,
        query: str,
        num_paths: int = 3,
        classification: Optional[QuestionClassification] = None,
    ) -> Tuple[List[ReasoningPath], float]:
        """
        Generate multiple reasoning paths for a query using parallel threads.

        Returns:
            (list_of_paths, parallel_speedup_factor)
        """
        if classification is None:
            classification = self.classifier.classify(query)

        start_time = time.time()
        num_paths = classification.num_paths
        strategies = self._select_strategies_enhanced(classification, num_paths)
        paths: List[ReasoningPath] = []

        with ThreadPoolExecutor(max_workers=num_paths) as executor:
            future_to_strategy = {
                executor.submit(
                    self._generate_path, query, strategy_name, instruction, classification
                ): strategy_name
                for strategy_name, instruction in strategies
            }
            for future in as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                try:
                    path = future.result()
                    if path and (path.steps or path.verdict or path.answer):
                        paths.append(path)
                except Exception as e:
                    print(f"  ✗ {strategy} generation error: {e}")

        total_time = time.time() - start_time
        estimated_sequential = sum(p.generation_time for p in paths) if paths else total_time
        speedup = estimated_sequential / total_time if total_time > 0 else 1.0

        if not paths:
            paths.append(self._create_fallback_path(query, classification))

        return paths, speedup

    # ── Strategy selection ───────────────────────────────────────────────────

    def _select_strategies_enhanced(
        self, classification: QuestionClassification, num_paths: int
    ) -> List[Tuple[str, str]]:
        """Select reasoning strategies based on question type with detailed prompts."""

        if classification.question_type == QuestionType.COMMONSENSE:
            strategies = [
                ("analytical",
                 """Use everyday common sense and practical reasoning.

IMPORTANT: Choose the MOST SPECIFIC, CONCRETE answer.
- Prefer specific actions over generic categories (e.g., "singing" > "making music")
- Prefer specific locations over general places (e.g., "refrigerator" > "place")
- Prefer direct terms over vague descriptions
- Think: What is the MOST DIRECT answer to this question?"""),

                ("evidence_based",
                 """Use practical knowledge and real-world experience.

IMPORTANT: Focus on SPECIFICITY.
- What is the most concrete, tangible answer?
- Avoid abstract or overly broad options
- Choose the answer that directly names the thing/action, not a category
- Real-world context: What would people actually say?"""),

                ("skeptical",
                 """Think critically about each option.

IMPORTANT: Eliminate based on specificity.
- Remove vague, generic, or overly broad choices
- Remove abstract concepts when concrete options exist
- Question: Which answer is MOST SPECIFIC and DIRECT?
- Be decisive — choose the clearest, most concrete option"""),
            ]
            return strategies[:num_paths]

        elif classification.question_type == QuestionType.BINARY:
            return [
                ("analytical",
                 "Analyze this systematically using logic and evidence. Be clear and decisive in your verdict."),
                ("skeptical",
                 "Approach critically, questioning assumptions. Challenge the premise if needed."),
                ("evidence_based",
                 "Focus strictly on factual evidence. What does established knowledge say?"),
            ][:num_paths]

        elif classification.question_type == QuestionType.MATHEMATICAL:
            return [
                ("algebraic",
                 """Solve using algebraic methods.

CRITICAL REQUIREMENTS:
1. Show EVERY step of your calculation explicitly
2. Write out ALL arithmetic operations (don't skip steps)
3. Label your steps clearly (Step 1, Step 2, etc.)
4. MUST end with: ANSWER: [number]

Be systematic and careful with calculations."""),

                ("numerical",
                 """Solve using numerical calculations step-by-step.

CRITICAL REQUIREMENTS:
1. Convert word problem to concrete numbers immediately
2. Show ALL arithmetic: 5 + 3 = 8, then 8 × 2 = 16, etc.
3. Label each calculation clearly
4. MUST end with: ANSWER: [number]

Work through the problem methodically. Double-check arithmetic."""),

                ("analytical",
                 """Solve by analysing the problem structure first.

CRITICAL REQUIREMENTS:
1. Identify what is given and what is asked
2. Break into sub-problems
3. Solve each sub-problem showing all arithmetic
4. MUST end with: ANSWER: [number]"""),
            ][:num_paths]

        elif classification.question_type == QuestionType.FACTUAL:
            return [
                ("direct",
                 "Provide the factual answer with supporting context. Be precise and cite established facts.")
            ]

        else:
            return [
                ("analytical",
                 "Analyse this comprehensively. Break down the question and reason systematically."),
                ("evidence_based",
                 "Base your analysis on established knowledge. What do authoritative sources say?"),
            ][:num_paths]

    # ── Path dispatch ────────────────────────────────────────────────────────

    def _generate_path(
        self,
        query: str,
        strategy: str,
        instruction: str,
        classification: QuestionClassification,
    ) -> Optional[ReasoningPath]:
        """Dispatch to binary or adaptive generator based on question type."""
        if classification.question_type == QuestionType.BINARY:
            return self._generate_binary_path(query, strategy, instruction, classification)
        return self._generate_adaptive_path(query, strategy, instruction, classification)

    # ── Binary path generation ───────────────────────────────────────────────

    def _generate_binary_path(
        self,
        query: str,
        strategy: str,
        instruction: str,
        classification: QuestionClassification,
    ) -> Optional[ReasoningPath]:
        """Generate a reasoning path for a binary (YES/NO) question."""
        start_time = time.time()
        prompt = f"""Question: {query}

{instruction}

Provide your analysis in this EXACT format:

VERDICT: YES or NO

REASONING:
Step 1: [Your first point]
Step 2: [Your second point]
Step 3: [Your third point]

CONCLUSION: [Your final conclusion]

Be direct and committed in your verdict."""

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = message.content[0].text
            self.total_tokens_input += message.usage.input_tokens
            self.total_tokens_output += message.usage.output_tokens

            path = self._parse_binary_response(query, response_text, strategy, classification)
            path.generation_time = time.time() - start_time
            path.question_type = QuestionType.BINARY
            return path
        except Exception as e:
            print(f"Claude API error for {strategy}: {e}")
            return None

    # ── Adaptive path generation ─────────────────────────────────────────────

    def _generate_adaptive_path(
        self,
        query: str,
        strategy: str,
        instruction: str,
        classification: QuestionClassification,
    ) -> Optional[ReasoningPath]:
        """Generate a reasoning path for non-binary question types."""
        start_time = time.time()

        is_commonsense_qa = 'Please select ONLY ONE' in query
        if is_commonsense_qa:
            parts = query.split('\n\nPlease select ONLY ONE')
            question_text = parts[0]
            choices_text = parts[1] if len(parts) > 1 else ""
            prompt = self._build_commonsense_prompt(question_text, choices_text, strategy)
        else:
            output_format = self._get_output_format(classification.question_type)
            prompt = f"""Question: {query}

{instruction}

{output_format}

Be clear and systematic."""

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=800,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = message.content[0].text
            self.total_tokens_input += message.usage.input_tokens
            self.total_tokens_output += message.usage.output_tokens

            path = self._parse_adaptive_response(query, response_text, strategy, classification)
            path.generation_time = time.time() - start_time
            return path
        except Exception as e:
            print(f"Generation error for {strategy}: {e}")
            return None

    def _build_commonsense_prompt(self, question: str, choices: str, strategy: str) -> str:
        """Build a commonsense-specific prompt guiding toward specific, concrete answers."""
        if strategy == "analytical":
            return f"""{question}

{choices}

Use EVERYDAY COMMONSENSE reasoning:

ANALYSIS:
Step 1: What is the question REALLY asking?
Step 2: Evaluate each option — which is MOST SPECIFIC and DIRECT?
Step 3: Select the MOST SPECIFIC answer

ANSWER: [Single letter A-E]: [choice text]

CRITICAL: Choose the MOST SPECIFIC, CONCRETE option that makes real-world sense."""

        elif strategy == "evidence_based":
            return f"""{question}

{choices}

Use PRACTICAL KNOWLEDGE to find the MOST SPECIFIC answer:

ANALYSIS:
Step 1: What SPECIFIC action or thing is being asked about?
Step 2: Apply specificity test — which is MOST CONCRETE and DIRECT?
Step 3: Select the MOST SPECIFIC, PRACTICAL answer

ANSWER: [Single letter A-E]: [choice text]"""

        else:
            return f"""{question}

{choices}

ANALYSIS:
Step 1: Parse what the question means in NORMAL LANGUAGE
Step 2: Evaluate specificity of each option
Step 3: Select the MOST SPECIFIC answer that makes REAL-WORLD SENSE

ANSWER: [Single letter A-E]: [choice text]"""

    def _get_output_format(self, question_type: QuestionType) -> str:
        """Return the expected output format string for the given question type."""
        if question_type == QuestionType.FACTUAL:
            return """Provide your answer:

ANSWER: [factual answer]

REASONING:
Step 1: [supporting evidence]
Step 2: [additional support]

CONCLUSION: [summary]"""

        elif question_type == QuestionType.MATHEMATICAL:
            return """Solve this step-by-step:

SOLUTION:
Step 1: [Understand the problem — identify what's given and what's asked]
Step 2: [Set up equations or identify operations needed]
Step 3: [Perform calculations with clear arithmetic]
Step 4: [Continue solving until you reach the final number]

CRITICAL: You MUST end with this exact format:
ANSWER: [numerical value]

Example: ANSWER: 42

Show ALL calculations explicitly. Double-check your arithmetic."""

        else:
            return """Provide analysis:

ANALYSIS:
Step 1: [key insight]
Step 2: [supporting evidence]
Step 3: [conclusion]

CONCLUSION: [summary]"""

    # ── Response parsing ─────────────────────────────────────────────────────

    def _parse_binary_response(
        self, query: str, response: str, strategy: str,
        classification: QuestionClassification,
    ) -> ReasoningPath:
        verdict = self._extract_verdict(response)
        steps = self._extract_reasoning_steps(response, strategy, QuestionType.BINARY)
        conclusion = self._extract_conclusion(response, verdict)
        confidence = self._calculate_dynamic_confidence(steps, response, verdict)

        return ReasoningPath(
            path_id=f"{strategy}_{int(time.time()*1000)}",
            query=query,
            verdict=verdict,
            steps=steps,
            conclusion=conclusion,
            confidence=confidence,
            generation_strategy=strategy,
            raw_output=response,
            question_type=classification.question_type,
            complexity_level=classification.complexity_level,
        )

    def _parse_adaptive_response(
        self, query: str, response: str, strategy: str,
        classification: QuestionClassification,
    ) -> ReasoningPath:
        answer = self._extract_answer_improved(response, classification.question_type)
        steps = self._extract_reasoning_steps(response, strategy, classification.question_type)
        conclusion = self._extract_conclusion(response, answer)
        confidence = self._calculate_dynamic_confidence(steps, response, answer)

        return ReasoningPath(
            path_id=f"{strategy}_{int(time.time()*1000)}",
            query=query,
            answer=answer,
            steps=steps,
            conclusion=conclusion,
            confidence=confidence,
            generation_strategy=strategy,
            raw_output=response,
            question_type=classification.question_type,
            complexity_level=classification.complexity_level,
        )

    # ── Text extraction helpers ──────────────────────────────────────────────

    def _extract_verdict(self, response: str) -> Optional[str]:
        """Extract a YES/NO/TRUE/FALSE verdict from the response."""
        for pattern in [r'VERDICT:\s*([A-Z]+)', r'Verdict:\s*([A-Z]+)']:
            match = re.search(pattern, response, re.MULTILINE | re.IGNORECASE)
            if match:
                v = match.group(1).upper()
                if v in {'TRUE', 'FALSE', 'YES', 'NO', 'DEPENDS', 'UNCLEAR'}:
                    return v

        response_lower = response.lower()
        if any(p in response_lower for p in ['is false', 'not true', 'myth']):
            return 'FALSE'
        if any(p in response_lower for p in ['is true', 'correct', 'accurate']):
            return 'TRUE'
        return None

    def _extract_answer_improved(
        self, response: str, question_type: QuestionType
    ) -> Optional[str]:
        """
        Extract answer from model response, prioritising explicit ANSWER markers
        then falling back to CONCLUSION sections.
        """
        print(f"\n[DEBUG ANSWER EXTRACT] Question type: {question_type}")

        if question_type == QuestionType.MATHEMATICAL:
            # Explicit markers
            for pattern in [
                r'ANSWER\s*[:\=]\s*\$?(\d+(?:,\d{3})*(?:\.\d+)?)',
                r'\\boxed\{(\d+(?:,\d{3})*(?:\.\d+)?)\}',
                r'(?:final\s+answer|the\s+answer)(?:\s+is|\s*:|\s*=)\s*\$?\*?\*?(\d+(?:,\d{3})*(?:\.\d+)?)',
            ]:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    return match.group(1).replace(',', '').strip()

            # CONCLUSION section
            cm = re.search(r'CONCLUSION:\s*(.+?)(?:\n\n|VERIFICATION|$)',
                           response, re.IGNORECASE | re.DOTALL)
            if cm:
                nm = re.search(r'\$?(\d+(?:,\d{3})*(?:\.\d+)?)', cm.group(1))
                if nm:
                    return nm.group(1).replace(',', '')

            # Last "= NUMBER" in last 5 lines
            for line in reversed(response.split('\n')[-5:]):
                m = re.search(r'=\s*\$?(\d+(?:,\d{3})*(?:\.\d+)?)\s*$', line)
                if m:
                    return m.group(1).replace(',', '')

            # Last number in final third
            numbers = re.findall(r'\$?(\d+(?:,\d{3})*(?:\.\d+)?)',
                                 response[int(len(response) * 0.7):])
            if numbers:
                return numbers[-1].replace(',', '')
            return None

        # Commonsense / other: look for "ANSWER: X: text"
        for pattern in [
            r'ANSWER:\s*([A-E]):\s*(.+?)(?:\n\n|CONCLUSION|VERIFICATION|$)',
            r'ANSWER:\s*\*?\*?([A-E])\s*:\s*(.+?)(?:\n\n|$)',
            r'##?\s*ANSWER:?\s*([A-E]):\s*(.+?)(?:\n|$)',
        ]:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                letter = match.group(1).upper()
                text = re.sub(r'\*\*|__|`', '', match.group(2).strip())
                return f"{letter}: {text}"

        for pattern in [
            r'CONCLUSION:\s*([A-E]):\s*(.+?)(?:\n\n|$)',
            r'CONCLUSION:\s*\*?\*?([A-E])\s+(?:is|are|would be)\s+(.+?)(?:\n|$)',
        ]:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                return re.sub(r'\*\*|__|`', '', f"{match.group(1)}: {match.group(2)}")

        return None

    def _extract_reasoning_steps(
        self, response: str, strategy: str, question_type: QuestionType
    ) -> List[LogicalStep]:
        """
        Extract structured reasoning steps from a model response.
        Handles markdown headers, bold text, numbered lists, and freeform prose.
        """
        steps: List[LogicalStep] = []
        reasoning_text = ""

        for pattern, _ in [
            (r'SOLUTION:(.*?)(?=ANSWER:|CONCLUSION:|VERIFICATION:|$)', 'SOLUTION'),
            (r'ANALYSIS:(.*?)(?=ANSWER:|CONCLUSION:|$)', 'ANALYSIS'),
            (r'REASONING:(.*?)(?=ANSWER:|CONCLUSION:|$)', 'REASONING'),
        ]:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                reasoning_text = match.group(1)
                break

        if not reasoning_text:
            reasoning_text = response

        lines = [l.strip() for l in reasoning_text.split('\n') if l.strip()]

        step_patterns = [
            (r'^##?\s*[Ss]tep\s+(\d+)[:\.\-]?\s*(.+)', 'Markdown Header'),
            (r'^\*\*[Ss]tep\s+(\d+)[:\.\-]?\*\*\s*(.+)', 'Bold Header'),
            (r'^[Ss]tep\s+(\d+)[:\.\-]\s*(.+)', 'Simple Step'),
            (r'^###\s+(.+)', 'Subheader'),
            (r'^(\d+)[\.\)]\s+(.+)', 'Numbered'),
            (r'^(Analyze|Evaluate|Identify|Determine|Select|Compare|Calculate|Solve|Define|Explain)\b\s*[:\-]?\s*(.+)',
             'Action Verb'),
            (r'^\*\*([A-Z][a-zA-Z\s]+)\*\*\s*(.+)', 'Bold Keyword'),
        ]

        step_count = 0
        for line in lines:
            if len(line) < 15:
                continue
            if line.startswith('##') and len(line) < 30:
                continue

            matched = False
            for pattern, pname in step_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if not match:
                    continue

                groups = match.groups()
                if pname in ('Markdown Header', 'Bold Header', 'Simple Step', 'Numbered'):
                    content = groups[-1].strip()
                elif pname == 'Subheader':
                    content = groups[0].strip()
                elif pname in ('Action Verb', 'Bold Keyword'):
                    content = f"{groups[0]}: {groups[1]}" if len(groups) > 1 else groups[0]
                else:
                    content = groups[-1].strip()

                content = re.sub(r'\*\*|__|`', '', content)
                if len(content) < 20:
                    continue

                is_math = '=' in content or any(
                    op in content.lower()
                    for op in ['divide', 'multiply', 'subtract', 'add', 'sum',
                               'calculate', 'solve', 'substitute', 'factor',
                               'simplify', 'expand', 'equation', 'formula']
                )

                confidence = self.confidence_assessor.assess_step_confidence(
                    content, is_math, question_type
                )

                step = LogicalStep(
                    id=f"{strategy}_step_{step_count+1}",
                    operation=self._classify_operation(content),
                    content=content,
                    confidence=confidence,
                    is_mathematical=is_math,
                )

                # Math verification for mathematical question types
                if is_math and question_type == QuestionType.MATHEMATICAL:
                    calculations = self.math_verifier.extract_calculations(content)
                    if calculations:
                        all_valid = True
                        for calc in calculations:
                            is_valid, feedback, _ = self.math_verifier.verify_calculation(calc)
                            if not is_valid:
                                all_valid = False
                                step.validation_status = ValidationStatus.INVALID
                                step.validation_feedback = feedback
                                step.confidence *= 0.5
                                print(f"[MATH VERIFY] ❌ {feedback}")
                                break
                        if all_valid:
                            step.calculation_verified = True
                            step.confidence = min(step.confidence * 1.1, 0.95)
                            print("[MATH VERIFY] ✓ All calculations verified")

                steps.append(step)
                step_count += 1
                matched = True
                break

            if not matched and len(line) > 40:
                if any(w in line.lower() for w in ['is', 'are', 'has', 'have', 'consists', 'includes', 'contains']):
                    content = re.sub(r'\*\*|__|`', '', line)
                    steps.append(LogicalStep(
                        id=f"{strategy}_step_{step_count+1}",
                        operation=LogicalOperation.INFERENCE,
                        content=content,
                        confidence=0.70,
                        is_mathematical=False,
                    ))
                    step_count += 1

        return steps

    def _calculate_dynamic_confidence(
        self,
        steps: List[LogicalStep],
        response: str,
        answer: Optional[str] = None,
    ) -> float:
        """Calculate path confidence from step scores, answer presence, and uncertainty."""
        if steps:
            avg = sum(s.confidence for s in steps) / len(steps)
            if answer and answer not in {"Unable to determine", ""}:
                avg += 0.10
            uncertainty = sum(
                1 for w in ['might', 'maybe', 'possibly', 'unclear', 'uncertain']
                if w in response.lower()
            )
            avg -= uncertainty * 0.05
            return min(max(avg, 0.30), 0.95)

        if answer and answer not in {"Unable to determine", ""}:
            return self._assess_answer_quality(answer, response)

        return 0.55 if len(response) > 200 else 0.30

    def _assess_answer_quality(self, answer: str, response: str) -> float:
        """Score answer quality based on length, definitive language, and uncertainty."""
        base = 0.60

        if len(answer) > 50:
            base += 0.12
        elif len(answer) > 20:
            base += 0.08

        if len(response) > 800:
            base += 0.10
        elif len(response) > 400:
            base += 0.05

        definitive = ['therefore', 'thus', 'must', 'always', 'is', 'are',
                      'the answer', 'conclusion', 'result', 'standard']
        base += min(sum(1 for w in definitive if w in response.lower()) * 0.03, 0.12)

        uncertain = ['might', 'possibly', 'approximately', 'roughly', 'likely',
                     'probably', 'seems', 'appears']
        base -= sum(1 for w in uncertain if w in response.lower()) * 0.04

        return min(max(base, 0.35), 0.90)

    def _classify_operation(self, content: str) -> LogicalOperation:
        """Classify a reasoning step's logical operation type."""
        cl = content.lower()
        if any(w in cl for w in ['evidence', 'research', 'data']):
            return LogicalOperation.EVIDENCE
        if any(w in cl for w in ['therefore', 'thus', 'implies']):
            return LogicalOperation.INFERENCE
        if any(w in cl for w in ['however', 'but', 'although']):
            return LogicalOperation.COUNTERARGUMENT
        return LogicalOperation.PREMISE

    def _extract_conclusion(self, response: str, verdict_or_answer: Optional[str]) -> str:
        """Extract or construct the conclusion from the response."""
        m = re.search(r'CONCLUSION:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        if m:
            conclusion = m.group(1).strip().split('\n')[0]
            return f"{verdict_or_answer}. {conclusion}" if verdict_or_answer else conclusion
        return f"Answer: {verdict_or_answer}" if verdict_or_answer else "See reasoning above"

    def _create_fallback_path(
        self, query: str, classification: Optional[QuestionClassification] = None
    ) -> ReasoningPath:
        """Create a minimal fallback path when all generation attempts fail."""
        return ReasoningPath(
            path_id=f"fallback_{int(time.time())}",
            query=query,
            verdict="UNCLEAR" if classification and classification.question_type == QuestionType.BINARY else None,
            answer="Unable to determine",
            steps=[],
            conclusion="Generation failed",
            confidence=0.2,
            generation_strategy="fallback",
            question_type=classification.question_type if classification else QuestionType.BINARY,
            complexity_level=classification.complexity_level if classification else ComplexityLevel.MODERATE,
        )

    def get_total_cost(self) -> float:
        """Estimate total API cost based on token usage (Claude 3.5 Haiku pricing)."""
        input_cost = (self.total_tokens_input / 1_000_000) * 0.80
        output_cost = (self.total_tokens_output / 1_000_000) * 4.00
        return input_cost + output_cost


# ============================================================================
# MAIN SYSTEM ORCHESTRATOR
# ============================================================================

class ClaudeDynamicReasoningSystem:
    """
    High-level pipeline orchestrator.

    Steps per query:
    1. Classify the question type and complexity.
    2. Generate multiple reasoning paths in parallel.
    3. Validate and synthesize a final answer.
    4. Return a NegotiationResult with all metrics.
    """

    def __init__(self, api_key: str):
        print("Initialising Claude Dynamic Reasoning System...")
        self.generator = ClaudeReasoningGenerator(api_key)
        # Import here to avoid circular dependency at module level
        from src.synthesis import AnswerSynthesizer
        self.synthesizer = AnswerSynthesizer(
            self.generator.client, self.generator.model
        )
        print(f"System ready — model: {self.generator.model}\n")

    def reason_with_synthesis(self, query: str, num_paths: int = 3) -> NegotiationResult:
        """
        Run the full reasoning → synthesis pipeline for a query.

        Args:
            query: The question or problem to solve.
            num_paths: Suggested number of paths (overridden by classifier).

        Returns:
            A NegotiationResult containing paths, synthesized answer, and metrics.
        """
        print(f"\n{'='*70}")
        print(f"QUERY: {query}")
        print('='*70)

        start_time = time.time()

        print("\n[0] Classifying question...")
        classification = self.generator.classifier.classify(query)
        print(f"Type: {classification.question_type.value}, "
              f"Complexity: {classification.complexity_level.value}")
        print(f"Validation: {classification.requires_validation}, "
              f"Paths: {classification.num_paths}")

        print(f"\n[1] Generating reasoning paths...")
        paths, speedup = self.generator.generate_multiple_paths_parallel(
            query, num_paths, classification
        )

        print(f"Generated {len(paths)} path(s) ({speedup:.2f}x speedup):")
        for path in paths:
            if path.verdict:
                ans_str = f"verdict={path.verdict}"
            elif path.answer:
                ans_str = f"answer={str(path.answer)[:40]}..."
            else:
                ans_str = "no answer"
            print(f"  • {path.generation_strategy}: {len(path.steps)} steps, "
                  f"{ans_str}, conf={path.confidence:.2f}")

        print("\n[2] Synthesising final answer...")
        synthesized = self.synthesizer.synthesize_final_answer(query, paths, classification)

        total_time = time.time() - start_time
        total_cost = self.generator.get_total_cost()

        print(f"\n[3] Complete in {total_time:.2f}s")
        print(f"Answer: {synthesized.definitive_answer}")
        print(f"Confidence: {synthesized.final_confidence:.2f}")
        print(f"API Cost: ${total_cost:.4f}")

        total_validations = sum(p.validation_passes for p in paths)
        total_regenerations = sum(p.regeneration_count for p in paths)

        return NegotiationResult(
            original_paths=paths,
            synthesized_answer=synthesized,
            total_time=total_time,
            parallel_speedup=speedup,
            total_cost=total_cost,
            total_validations=total_validations,
            total_regenerations=total_regenerations,
            classification=classification,
        )

    def display_result(self, result: NegotiationResult) -> None:
        """Print the full reasoning and synthesis output in a readable format."""
        print("\n" + "="*70)
        print("FINAL RESULT")
        print("="*70)

        if result.classification:
            print(f"\nQuestion Classification:")
            print(f"  Type: {result.classification.question_type.value}")
            print(f"  Complexity: {result.classification.complexity_level.value}")

        print("\n" + "▶"*35)
        print("SYNTHESISED ANSWER")
        print("▶"*35)
        print(f"\nQuery: {result.synthesized_answer.query}")
        print(f"\nAnswer: {result.synthesized_answer.definitive_answer}")
        print(f"Confidence: {result.synthesized_answer.final_confidence:.2f}")

        if result.synthesized_answer.supporting_reasoning:
            print("\nKey Supporting Points:")
            for i, point in enumerate(result.synthesized_answer.supporting_reasoning, 1):
                print(f"  {i}. {point}")

        if result.synthesized_answer.conflicting_points:
            print("\nConflicting Aspects:")
            for conflict in result.synthesized_answer.conflicting_points:
                print(f"  ⚠ {conflict}")

        print("\nSynthesis Process:")
        print(result.synthesized_answer.synthesis_explanation)

        print("\n" + "-"*70)
        print("INDIVIDUAL REASONING PATHS")
        print("-"*70)
        for path in result.original_paths:
            print(f"\n{path.to_readable_chain()}")

        print("\n" + "-"*70)
        print("PERFORMANCE & COST")
        print("-"*70)
        print(f"Total time: {result.total_time:.2f}s")
        print(f"Parallel speedup: {result.parallel_speedup:.2f}x")
        print(f"Total API cost: ${result.total_cost:.4f}")

        if result.total_validations > 0:
            print(f"\nValidation Statistics:")
            print(f"  Validation passes: {result.total_validations}")
            print(f"  Regenerations: {result.total_regenerations}")

        print("\n" + "="*70)
