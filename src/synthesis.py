"""
synthesis.py
============
Answer synthesis, validation, regeneration, divergence detection, and
confidence-weighted consensus for the GSM8K multi-path reasoning system.

Classes:
- SpecificityScorer: Scores how concrete/specific an answer is.
- AnswerSynthesizer: Combines multiple ReasoningPath objects into a single
  synthesised answer, with optional step validation and regeneration.
"""

import re
import time
from typing import Dict, List, Optional, Tuple

import anthropic

from src.utils import (
    ComplexityLevel,
    LogicalOperation,
    LogicalStep,
    QuestionClassification,
    QuestionType,
    ReasoningPath,
    SynthesizedAnswer,
    ValidationStatus,
)
from src.verifier import MathematicalVerifier


# ============================================================================
# SPECIFICITY SCORER
# ============================================================================

class SpecificityScorer:
    """
    Score answer specificity to prefer concrete answers over abstract ones.

    Philosophy: "singing" (specific) > "making music" (generic).
    Shorter, concrete action words score highest.
    """

    @staticmethod
    def score_specificity(answer_text: str, question: str) -> float:
        """Return a specificity score in [0.0, 1.0] for the given answer."""
        score = 0.5
        answer_lower = answer_text.lower().strip()
        answer_len = len(answer_text)

        # Rule 1: Length heuristic
        if answer_len <= 8:
            score += 0.30
        elif answer_len <= 15:
            score += 0.20
        elif answer_len <= 25:
            score += 0.10
        elif answer_len > 40:
            score -= 0.15

        # Rule 2: Concrete action verbs
        concrete_actions = {
            'singing', 'torn', 'walk', 'disturb', 'attention', 'walking',
            'running', 'reading', 'writing', 'eating', 'drinking', 'talking',
            'listening', 'watching', 'playing', 'building', 'cooking',
            'cleaning', 'studying', 'sleeping', 'dancing', 'swimming',
            'driving', 'crying', 'laughing', 'screaming', 'whispering',
            'jumping', 'sitting', 'standing',
        }
        if any(action in answer_lower for action in concrete_actions):
            score += 0.35

        # Rule 3: Abstract/generic terms (penalty)
        abstract_terms = {
            'making', 'doing', 'having', 'being', 'getting', 'going',
            'coming', 'taking', 'giving', 'putting', 'becoming',
            'thing', 'stuff', 'something', 'anything', 'everything',
            'general', 'various', 'some', 'any', 'all',
            'live in', 'work in', 'be in', 'go to',
        }
        abstract_count = sum(1 for t in abstract_terms if t in answer_lower)
        score -= 0.30 * abstract_count

        # Rule 4: Specific nouns
        specific_nouns = {
            'bank', 'library', 'hospital', 'school', 'office', 'restaurant',
            'store', 'park', 'theater', 'gym', 'refrigerator', 'oven',
            'desk', 'chair', 'door', 'window', 'table', 'bed', 'car', 'phone',
            'hand', 'foot', 'eye', 'ear', 'mouth', 'nose',
        }
        if any(noun in answer_lower for noun in specific_nouns):
            score += 0.25

        # Rule 5: Generic category words (penalty)
        generic_categories = {
            'place', 'location', 'area', 'region', 'spot', 'site',
            'item', 'object', 'article', 'activity', 'action', 'process',
            'method', 'way', 'type', 'kind', 'sort', 'form', 'category',
            'person', 'people', 'individual', 'someone',
        }
        if any(cat in answer_lower for cat in generic_categories):
            score -= 0.25

        # Rule 6: Multi-word penalty
        word_count = len(answer_text.split())
        if word_count >= 3:
            score -= 0.10 * (word_count - 2)

        # Rule 7: Article prefix
        if any(answer_lower.startswith(a) for a in ['a ', 'an ', 'the ']):
            score -= 0.05

        # Rule 8: Single concrete word bonus
        if (word_count == 1 and answer_len <= 10 and
                any(x in answer_lower for x in concrete_actions | specific_nouns)):
            score += 0.20

        # Rule 9: Question context
        if question:
            q_lower = question.lower()
            if 'where' in q_lower and any(n in answer_lower for n in specific_nouns):
                score += 0.10
            elif ('what do' in q_lower or 'what are' in q_lower) and \
                    any(a in answer_lower for a in concrete_actions):
                score += 0.10

        return max(0.0, min(1.0, score))


# ============================================================================
# ANSWER SYNTHESIZER
# ============================================================================

class AnswerSynthesizer:
    """
    Combines multiple reasoning paths into a single synthesised final answer.

    Features:
    - Optional step-level validation with regeneration.
    - Numerical divergence detection (>10% difference triggers a third path).
    - Specificity-based answer selection for commonsense questions.
    - Confidence-weighted consensus voting.
    """

    def __init__(self, client: anthropic.Anthropic, model: str):
        self.client = client
        self.model = model
        self.total_tokens_input = 0
        self.total_tokens_output = 0
        self.specificity_scorer = SpecificityScorer()

    # ── Main entry point ─────────────────────────────────────────────────────

    def synthesize_final_answer(
        self,
        query: str,
        paths: List[ReasoningPath],
        classification: QuestionClassification,
    ) -> SynthesizedAnswer:
        """
        Synthesise a final answer from multiple reasoning paths.

        Pipeline:
        1. (Optional) Validate + regenerate individual steps.
        2. Detect numerical divergence; generate consensus path if needed.
        3. Extract answers from paths.
        4. Vote / select best answer.
        5. Verify mathematical calculations.
        6. Build and return SynthesizedAnswer.
        """
        if classification.requires_validation:
            print(f"\n[1.5] Validating paths (complexity: {classification.complexity_level.value})...")
            paths = self._validate_and_regenerate_paths(paths, classification)

        print("\n[1.6] Checking for numerical divergence...")
        divergence_detected, divergent_pair = self._detect_numerical_divergence(paths)

        if divergence_detected and divergent_pair:
            print(f"\n[DIVERGENCE] Detected: "
                  f"{divergent_pair[0].generation_strategy} vs "
                  f"{divergent_pair[1].generation_strategy}")
            consensus_path = self._regenerate_for_consensus(query, divergent_pair, classification)
            if consensus_path:
                paths.append(consensus_path)
                print(f"[CONSENSUS] Path added ({len(paths)} total paths now)")

        # ── Answer extraction ────────────────────────────────────────────────
        if classification.question_type == QuestionType.MATHEMATICAL:
            print("\n[1.7] Extracting mathematical answers from paths...")
            extracted_answers = []
            for i, path in enumerate(paths):
                answer = None
                if path.answer and path.answer != "Unable to determine":
                    answer = path.answer
                elif path.raw_output:
                    answer = self._extract_answer_from_raw_output(path.raw_output)
                    if answer:
                        path.answer = answer
                elif path.steps:
                    last = path.steps[-1]
                    m = re.search(r'=\s*\$?(\d+(?:,\d{3})*(?:\.\d+)?)', last.content)
                    if m:
                        answer = m.group(1).replace(',', '')
                        path.answer = answer

                if answer:
                    extracted_answers.append(answer)
                    print(f"  Path {i+1}: {answer}")
                else:
                    print(f"  Path {i+1}: ✗ No answer found")

            if extracted_answers:
                definitive_answer = self._select_best_numerical_answer(extracted_answers, paths)
            else:
                definitive_answer = "Unable to determine answer"
            answer_format = "answer"

        elif classification.question_type == QuestionType.BINARY:
            verdicts = [p.verdict for p in paths if p.verdict]
            definitive_answer = self._determine_consensus_verdict(verdicts, paths)
            answer_format = "verdict"

        else:
            answers = [p.answer for p in paths if p.answer]
            definitive_answer = self._synthesize_answers_with_specificity(answers, paths)
            answer_format = "answer"

        # ── Math verification ────────────────────────────────────────────────
        confidence_multiplier = 1.0
        if classification.question_type == QuestionType.MATHEMATICAL:
            print("\n[VERIFICATION] Verifying mathematical answer...")
            math_verifier = MathematicalVerifier()
            all_calculations = []
            for path in paths:
                for step in path.steps:
                    if step.is_mathematical:
                        all_calculations.extend(
                            math_verifier.extract_calculations(step.content)
                        )

            print(f"[VERIFICATION] Found {len(all_calculations)} calculation(s)")
            failed_calcs = []
            for calc in all_calculations:
                is_valid, feedback, _ = math_verifier.verify_calculation(calc)
                if not is_valid:
                    failed_calcs.append(feedback)

            if failed_calcs:
                print(f"[VERIFICATION] {len(failed_calcs)} error(s) found:")
                for err in failed_calcs[:3]:
                    print(f"  • {err}")
                failure_rate = len(failed_calcs) / len(all_calculations) if all_calculations else 0
                confidence_multiplier = (
                    0.5 if failure_rate > 0.5
                    else 0.7 if failure_rate > 0.25
                    else 0.85
                )
            else:
                print(f"[VERIFICATION] All {len(all_calculations)} calculation(s) verified")
                confidence_multiplier = 1.1

        # ── Build result ─────────────────────────────────────────────────────
        supporting_reasoning = self._extract_key_points(paths)
        conflicting_points = self._identify_conflicts(paths)
        synthesis_explanation = self._generate_synthesis_explanation(
            query, paths, definitive_answer, conflicting_points
        )
        base_confidence = self._calculate_synthesis_confidence(paths, conflicting_points)
        final_confidence = min(base_confidence * confidence_multiplier, 0.95)

        return SynthesizedAnswer(
            query=query,
            definitive_answer=definitive_answer,
            supporting_reasoning=supporting_reasoning,
            conflicting_points=conflicting_points,
            final_confidence=final_confidence,
            synthesis_explanation=synthesis_explanation,
            question_type=classification.question_type,
            answer_format=answer_format,
        )

    # ── Validation & regeneration ────────────────────────────────────────────

    def _validate_and_regenerate_paths(
        self,
        paths: List[ReasoningPath],
        classification: QuestionClassification,
    ) -> List[ReasoningPath]:
        """Validate each step in each path; regenerate from the first failure."""
        validated_paths = []
        total_validations = 0
        total_regenerations = 0
        force_validation = (classification.question_type == QuestionType.MATHEMATICAL)

        for idx, path in enumerate(paths):
            print(f"  Validating path {idx+1}/{len(paths)} ({path.generation_strategy})...")
            if not path.steps:
                validated_paths.append(path)
                continue

            needs_regen = False
            failed_step_idx = -1

            for step_idx, step in enumerate(path.steps):
                if not force_validation and step.confidence >= classification.confidence_threshold:
                    step.validation_status = ValidationStatus.VALID
                    continue

                result = self._validate_step(
                    step, path.steps[:step_idx], path.query, classification.question_type
                )
                path.validation_passes += 1
                total_validations += 1

                if result['is_valid']:
                    step.validation_status = ValidationStatus.VALID
                    step.validation_feedback = result['feedback']
                    step.confidence = max(step.confidence, result['confidence'])
                else:
                    step.validation_status = ValidationStatus.INVALID
                    step.validation_feedback = result['feedback']
                    needs_regen = True
                    failed_step_idx = step_idx
                    print(f"    ✗ Step {step_idx+1} failed validation")
                    break

            if needs_regen and failed_step_idx >= 0:
                print(f"    ↻ Regenerating from step {failed_step_idx+1}...")
                regen = self._regenerate_from_failed_step(path, failed_step_idx, classification)
                if regen:
                    path = regen
                    path.regeneration_count += 1
                    total_regenerations += 1
                    print("    ✓ Regeneration successful")
                else:
                    print("    ✗ Regeneration failed, keeping original")

            validated_paths.append(path)

        print(f"  Validation complete: {total_validations} checks, "
              f"{total_regenerations} regenerations")
        return validated_paths

    def _validate_step(
        self,
        step: LogicalStep,
        previous_steps: List[LogicalStep],
        query: str,
        question_type: QuestionType,
    ) -> Dict:
        """Validate a single reasoning step (math first, then LLM fallback)."""
        # Math verification before expensive LLM call
        if step.is_mathematical and question_type == QuestionType.MATHEMATICAL:
            math_verifier = MathematicalVerifier()
            calculations = math_verifier.extract_calculations(step.content)
            if calculations:
                for calc in calculations:
                    is_valid, feedback, _ = math_verifier.verify_calculation(calc)
                    if not is_valid:
                        return {'is_valid': False, 'confidence': 0.3,
                                'feedback': f"Math error: {feedback}"}
                return {'is_valid': True, 'confidence': 0.9,
                        'feedback': "✓ All mathematical calculations verified"}

        # LLM validation fallback
        context = f"Original Question: {query}\n\n"
        if previous_steps:
            context += "Previous reasoning steps:\n"
            for i, ps in enumerate(previous_steps, 1):
                context += f"{i}. {ps.content}\n"
            context += "\n"
        context += f"Step to validate:\n{step.content}"

        if step.is_mathematical:
            prompt = (f"{context}\n\nValidate this mathematical step. Check:\n"
                      "1. Is the arithmetic/algebra correct?\n"
                      "2. Does it follow logically from previous steps?\n"
                      "3. Are there any calculation errors?\n\n"
                      "Respond EXACTLY:\nVALID: YES or NO\nCONFIDENCE: 0.0 to 1.0\n"
                      "FEEDBACK: Brief explanation")
        else:
            prompt = (f"{context}\n\nValidate this reasoning step. Check:\n"
                      "1. Does it logically follow from previous steps?\n"
                      "2. Is it factually accurate?\n"
                      "3. Is it relevant to answering the question?\n\n"
                      "Respond EXACTLY:\nVALID: YES or NO\nCONFIDENCE: 0.0 to 1.0\n"
                      "FEEDBACK: Brief explanation")

        try:
            message = self.client.messages.create(
                model=self.model, max_tokens=300, temperature=0.2,
                messages=[{"role": "user", "content": prompt}],
            )
            response = message.content[0].text
            self.total_tokens_input += message.usage.input_tokens
            self.total_tokens_output += message.usage.output_tokens

            is_valid = 'VALID: YES' in response.upper()
            cm = re.search(r'CONFIDENCE:\s*(0?\.\d+|1\.0)', response)
            confidence = float(cm.group(1)) if cm else 0.7
            fm = re.search(r'FEEDBACK:\s*(.+?)(?:\n|$)', response, re.DOTALL)
            feedback = fm.group(1).strip() if fm else response[:100]
            return {'is_valid': is_valid, 'confidence': confidence, 'feedback': feedback}
        except Exception as e:
            return {'is_valid': True, 'confidence': 0.6,
                    'feedback': f"Validation error: {str(e)[:50]}"}

    def _regenerate_from_failed_step(
        self,
        path: ReasoningPath,
        failed_step_idx: int,
        classification: QuestionClassification,
    ) -> Optional[ReasoningPath]:
        """Regenerate reasoning from a failed step using an alternative approach."""
        valid_steps = path.steps[:failed_step_idx]
        failed_step = path.steps[failed_step_idx]

        context = f"Question: {path.query}\n\n"
        if valid_steps:
            context += "These steps are correct:\n"
            for i, s in enumerate(valid_steps, 1):
                context += f"{i}. {s.content}\n"
            context += "\n"
        context += f"This step FAILED validation:\n{failed_step.content}\n"
        context += f"Reason: {failed_step.validation_feedback}\n\n"

        if classification.question_type == QuestionType.MATHEMATICAL:
            approach = ("Try NUMERICAL approach instead. Calculate with actual numbers."
                        if 'algebraic' in path.generation_strategy.lower()
                        else "Try ALGEBRAIC approach instead. Use equations and variables.")
        else:
            approach = "Try a completely different reasoning approach."

        prompt = (f"{context}{approach}\n\nContinue solving from where valid steps ended. "
                  f"Show clear reasoning.\n\nFormat:\n"
                  f"Step {failed_step_idx+1}: [new step]\n"
                  f"Step {failed_step_idx+2}: [next step]\n...\nANSWER: [final answer]")

        try:
            message = self.client.messages.create(
                model=self.model, max_tokens=1000, temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
            )
            response = message.content[0].text
            self.total_tokens_input += message.usage.input_tokens
            self.total_tokens_output += message.usage.output_tokens

            new_steps = self._extract_reasoning_steps_from_text(
                response, f"{path.generation_strategy}_regen", classification.question_type
            )
            if not new_steps:
                return None

            new_answer = self._extract_answer_from_text(response, classification.question_type)
            combined = valid_steps + new_steps

            return ReasoningPath(
                path_id=path.path_id,
                query=path.query,
                verdict=None if classification.question_type != QuestionType.BINARY else new_answer,
                answer=new_answer if classification.question_type != QuestionType.BINARY else None,
                steps=combined,
                conclusion=f"Regenerated: {new_answer}" if new_answer else "Regenerated",
                confidence=self._confidence_from_steps(new_answer, combined, response),
                generation_strategy=f"{path.generation_strategy}_regenerated",
                raw_output=response,
                question_type=path.question_type,
                complexity_level=path.complexity_level,
                validation_passes=path.validation_passes,
                regeneration_count=path.regeneration_count,
            )
        except Exception as e:
            print(f"    ⚠ Regeneration error: {e}")
            return None

    # ── Numerical divergence detection ──────────────────────────────────────

    def _detect_numerical_divergence(
        self, paths: List[ReasoningPath]
    ) -> Tuple[bool, Optional[Tuple]]:
        """
        Detect when two paths give numerical answers that differ by more than 10%.

        Returns:
            (divergence_detected, (path1, path2, pct_diff)) or (False, None)
        """
        numerical_answers = []
        for path in paths:
            if path.answer:
                nums = self._extract_numerical_values(path.answer.lower())
                if nums:
                    numerical_answers.append((path, max(nums)))

        for i, (p1, n1) in enumerate(numerical_answers):
            for p2, n2 in numerical_answers[i+1:]:
                if n1 > 0 and n2 > 0:
                    pct_diff = abs(n1 - n2) / max(n1, n2)
                    if pct_diff > 0.10:
                        print(f"  ⚠️ DIVERGENCE: {n1:.3f} vs {n2:.3f} "
                              f"({pct_diff*100:.1f}% diff)")
                        return True, (p1, p2, pct_diff)
        return False, None

    def _regenerate_for_consensus(
        self,
        query: str,
        divergent_paths: Tuple,
        classification: QuestionClassification,
    ) -> Optional[ReasoningPath]:
        """Generate a third verification path when two paths numerically disagree."""
        path1, path2, pct_diff = divergent_paths
        print(f"\n  [REGENERATION] Consensus attempt...")
        print(f"  Path 1 ({path1.generation_strategy}): {path1.answer}")
        print(f"  Path 2 ({path2.generation_strategy}): {path2.answer}")

        prompt = (
            f"Question: {query}\n\n"
            f"Two independent approaches gave different answers:\n"
            f"- Approach 1 ({path1.generation_strategy}): {path1.answer}\n"
            f"- Approach 2 ({path2.generation_strategy}): {path2.answer}\n\n"
            f"These differ by {pct_diff*100:.1f}%.\n\n"
            "Solve from scratch using a THIRD distinct approach. "
            "Be extremely careful with calculations. Show every step.\n\n"
            "Format:\nAPPROACH: [method name]\n\nSOLUTION:\n"
            "Step 1: ...\n...\n\nANSWER: [final numerical answer]\n\n"
            "VERIFICATION: [double-check your calculation]"
        )

        try:
            message = self.client.messages.create(
                model=self.model, max_tokens=1500, temperature=0.5,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = message.content[0].text
            self.total_tokens_input += message.usage.input_tokens
            self.total_tokens_output += message.usage.output_tokens

            answer = self._extract_answer_from_regeneration(response_text)
            if answer:
                print(f"  ✓ Consensus path generated: {answer}")
                return ReasoningPath(
                    path_id=f"consensus_{int(time.time()*1000)}",
                    query=query,
                    answer=answer,
                    steps=[],
                    conclusion="Consensus answer after divergence detection",
                    confidence=0.72,
                    generation_strategy="consensus_verification",
                    raw_output=response_text,
                    question_type=classification.question_type,
                    complexity_level=classification.complexity_level,
                )
            print("  ✗ Could not extract answer from consensus path")
            return None
        except Exception as e:
            print(f"  ✗ Regeneration error: {e}")
            return None

    def _extract_answer_from_regeneration(self, response: str) -> Optional[str]:
        """Extract answer from a regeneration response."""
        m = re.search(r'ANSWER:\s*\*?\*?(.+?)(?:\n\n|VERIFICATION|$)',
                      response, re.IGNORECASE | re.DOTALL)
        if m:
            answer = re.sub(r'^\*\*|\*\*$', '', m.group(1)).strip()
            if answer and len(answer) > 3:
                return answer
        return None

    # ── Synthesis with specificity ───────────────────────────────────────────

    def _synthesize_answers_with_specificity(
        self, answers: List[str], paths: List[ReasoningPath]
    ) -> str:
        """Select the best answer using confidence-weighted scoring with specificity bonus."""
        if not answers:
            return "Unable to determine answer"

        groups = self._group_equivalent_answers(answers)
        if len(groups) == 1:
            return f"{answers[0]} (unanimous)"

        best_answer = None
        best_score = -1
        for i, path in enumerate(paths):
            if not path.answer or i >= len(answers):
                continue
            conf_score = path.confidence
            spec_score = self.specificity_scorer.score_specificity(path.answer, path.query)
            combined = 0.70 * conf_score + 0.30 * spec_score
            print(f"  Path {i+1}: conf={conf_score:.2f}, spec={spec_score:.2f}, "
                  f"combined={combined:.2f}, answer='{path.answer[:40]}'")
            if combined > best_score:
                best_score = combined
                best_answer = path.answer

        return f"{best_answer} (best, score={best_score:.2f})" if best_answer \
               else f"{paths[0].answer} (fallback)"

    def _select_best_numerical_answer(
        self, answers: List[str], paths: List[ReasoningPath]
    ) -> str:
        """Select the best numerical answer via majority vote + confidence weighting."""
        if not answers:
            return "Unable to determine answer"

        vote_scores: Dict = {}
        for i, ans in enumerate(answers):
            m = re.search(r'(\d+(?:\.\d+)?)', ans.replace(',', ''))
            if not m:
                continue
            try:
                num = float(m.group(1))
                conf = paths[i].confidence if i < len(paths) else 0.5
                if num not in vote_scores:
                    vote_scores[num] = {'score': 0, 'count': 0, 'original': ans}
                vote_scores[num]['score'] += conf
                vote_scores[num]['count'] += 1
            except ValueError:
                pass

        if not vote_scores:
            return answers[0]

        best = max(vote_scores.items(), key=lambda x: (x[1]['count'], x[1]['score']))
        result = best[1]['original']
        count = best[1]['count']
        total = len(answers)

        if count == total:
            return f"{result} (unanimous)"
        elif count > total / 2:
            return f"{result} (consensus {count}/{total})"
        return f"{result} (best of {count}/{total})"

    # ── Answer equivalence ───────────────────────────────────────────────────

    def _answers_equivalent(self, ans1: str, ans2: str) -> bool:
        """Check if two answers are semantically equivalent."""
        clean1 = re.sub(r'\*\*|__|`|~~', '', ans1).lower().strip()
        clean2 = re.sub(r'\*\*|__|`|~~', '', ans2).lower().strip()

        if clean1 == clean2:
            return True

        nums1 = self._extract_numerical_values(clean1)
        nums2 = self._extract_numerical_values(clean2)

        if nums1 and nums2:
            return self._numbers_equivalent(nums1, nums2)

        sol1 = set(re.findall(r'([a-z])\s*=\s*([-+]?\d+\.?\d*)', clean1))
        sol2 = set(re.findall(r'([a-z])\s*=\s*([-+]?\d+\.?\d*)', clean2))
        if sol1 and sol2 and sol1 == sol2:
            return True

        if (clean1 in clean2 or clean2 in clean1) and (len(clean1) < 20 or len(clean2) < 20):
            return True

        if not nums1 and not nums2:
            w1, w2 = set(clean1.split()), set(clean2.split())
            if w1 and w2:
                return len(w1 & w2) / len(w1 | w2) > 0.7

        return False

    def _group_equivalent_answers(self, answers: List[str]) -> List[List[str]]:
        """Group semantically equivalent answers together."""
        groups: List[List[str]] = []
        used: set = set()

        for i, ai in enumerate(answers):
            if i in used:
                continue
            group = [ai]
            used.add(i)
            for j, aj in enumerate(answers):
                if j <= i or j in used:
                    continue
                if self._answers_equivalent(ai, aj):
                    group.append(aj)
                    used.add(j)
            groups.append(group)

        return groups

    def _extract_numerical_values(self, text: str) -> List[float]:
        """Extract all numerical values from text."""
        numbers = []
        for p in re.findall(r'([\d.]+)\s*%', text):
            try:
                numbers.append(float(p) / 100)
            except ValueError:
                pass
        pcts = re.findall(r'([\d.]+)\s*%', text)
        for n in re.findall(r'(?:^|\s)([\d.]+)(?:\s|$|[^%\d.])', text):
            if n not in pcts:
                try:
                    numbers.append(float(n))
                except ValueError:
                    pass
        for n, d in re.findall(r'(\d+)/(\d+)', text):
            try:
                numbers.append(float(n) / float(d))
            except (ValueError, ZeroDivisionError):
                pass
        return sorted(set(numbers))

    def _numbers_equivalent(self, nums1: List[float], nums2: List[float]) -> bool:
        """Compare two number lists with tolerance."""
        if len(nums1) != len(nums2):
            return False
        for n1, n2 in zip(nums1, nums2):
            if 0 <= n1 <= 1 and 0 <= n2 <= 1:
                if abs(n1 - n2) > 0.05:
                    return False
            else:
                if abs(n1 - n2) > max(0.1 * max(abs(n1), abs(n2)), 0.01):
                    return False
        return True

    # ── Helper methods ───────────────────────────────────────────────────────

    def _determine_consensus_verdict(
        self, verdicts: List[str], paths: List[ReasoningPath]
    ) -> str:
        if not verdicts:
            return "Unable to determine"
        counts: Dict[str, int] = {}
        for v in verdicts:
            counts[v] = counts.get(v, 0) + 1
        verdict, count = max(counts.items(), key=lambda x: x[1])
        if count == len(verdicts):
            return f"{verdict} (unanimous)"
        elif count > len(verdicts) / 2:
            return f"{verdict} (majority {count}/{len(verdicts)})"
        return f"{verdict} (contested {count}/{len(verdicts)})"

    def _extract_key_points(self, paths: List[ReasoningPath]) -> List[str]:
        key_points = []
        for path in paths:
            relevant = [s for s in path.steps
                        if s.operation in (LogicalOperation.EVIDENCE, LogicalOperation.INFERENCE)]
            for step in relevant[:2]:
                key_points.append(f"[{path.generation_strategy}] {step.content}")
        return key_points[:5]

    def _identify_conflicts(self, paths: List[ReasoningPath]) -> List[str]:
        conflicts = []
        verdicts = [p.verdict for p in paths if p.verdict]
        if len(set(verdicts)) > 1:
            conflicts.append(f"Verdict disagreement: {', '.join(set(verdicts))}")
        answers = [p.answer for p in paths if p.answer]
        if len(answers) > 1 and len(self._group_equivalent_answers(answers)) > 1:
            conflicts.append("Answer disagreement")
        return conflicts

    def _generate_synthesis_explanation(
        self, query: str, paths: List[ReasoningPath],
        answer: str, conflicts: List[str],
    ) -> str:
        explanation = f"Analysed {len(paths)} independent reasoning approaches.\n"
        explanation += ("All paths converged with consistent reasoning."
                        if not conflicts
                        else f"Found {len(conflicts)} disagreement(s).")
        return explanation

    def _extract_answer_from_raw_output(self, raw_output: str) -> Optional[str]:
        """Extract numerical answer from raw model output."""
        m = re.search(r'ANSWER\s*[:\=]\s*\$?(\d+(?:,\d{3})*(?:\.\d+)?)',
                      raw_output, re.IGNORECASE)
        if m:
            return m.group(1).replace(',', '')
        m = re.search(r'\\boxed\{(\d+(?:,\d{3})*(?:\.\d+)?)\}', raw_output)
        if m:
            return m.group(1).replace(',', '')
        for line in reversed(raw_output.split('\n')[-5:]):
            m = re.search(r'=\s*\$?(\d+(?:,\d{3})*(?:\.\d+)?)\s*$', line)
            if m:
                return m.group(1).replace(',', '')
        return None

    def _extract_reasoning_steps_from_text(
        self, text: str, strategy: str, question_type: QuestionType
    ) -> List[LogicalStep]:
        steps = []
        for line in text.split('\n'):
            m = re.match(r'^(?:step\s+)?(\d+)[\.\):\-]\s*(.+)', line, re.IGNORECASE)
            if m and len(m.group(2)) > 10:
                content = m.group(2).strip()
                is_math = '=' in content or any(
                    op in content.lower() for op in ['calculate', 'solve', 'divide']
                )
                steps.append(LogicalStep(
                    id=f"{strategy}_step_{len(steps)+1}",
                    operation=LogicalOperation.INFERENCE,
                    content=content,
                    confidence=0.75,
                    is_mathematical=is_math,
                ))
        return steps

    def _extract_answer_from_text(
        self, text: str, question_type: QuestionType
    ) -> Optional[str]:
        m = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if lines and '=' in lines[-1]:
            return lines[-1]
        return None

    def _confidence_from_steps(
        self, answer: Optional[str], steps: List[LogicalStep], text: str
    ) -> float:
        base = 0.7
        if answer:
            base += 0.1
        if len(steps) >= 3:
            base += 0.05
        uncertainty = sum(1 for w in ['might', 'maybe', 'possibly'] if w in text.lower())
        base -= uncertainty * 0.03
        return min(max(base, 0.1), 0.95)

    def _calculate_synthesis_confidence(
        self, paths: List[ReasoningPath], conflicts: List[str]
    ) -> float:
        avg = sum(p.confidence for p in paths) / len(paths) if paths else 0.5
        return min(max(avg - len(conflicts) * 0.1, 0.15), 0.95)
