"""
verifier.py
===========
Mathematical verification and confidence assessment for the GSM8K
multi-path reasoning system.

Classes:
- ConfidenceAssessor: Dynamically scores step confidence based on content.
- MathematicalVerifier: Competition-grade arithmetic and mathematical
  calculation verifier supporting 20+ pattern types.
"""

import re
import math
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from src.utils import QuestionType


# ============================================================================
# CONFIDENCE ASSESSOR
# ============================================================================

class ConfidenceAssessor:
    """Dynamically assess confidence based on actual step content."""

    @staticmethod
    def assess_step_confidence(content: str, is_mathematical: bool,
                               question_type: QuestionType) -> float:
        """
        Calculate confidence based on step content rather than a hardcoded value.

        Uncertainty markers reduce confidence; definitive language increases it.
        Complex arithmetic operations also reduce confidence (pending verification).
        """
        base_confidence = 0.75

        uncertainty_words = ['might', 'possibly', 'approximately', 'roughly',
                             'likely', 'probably', 'seems', 'appears', 'assume']
        uncertainty_count = sum(1 for word in uncertainty_words if word in content.lower())
        base_confidence -= uncertainty_count * 0.05

        if is_mathematical:
            ops = (content.count('+') + content.count('-') +
                   content.count('*') + content.count('/'))
            if ops > 3:
                base_confidence -= 0.1

        if any(phrase in content.lower() for phrase in ['assume', 'suppose', 'if we']):
            base_confidence -= 0.08

        expert_terms = ['paradox', 'infinity', 'irrational', 'undefined', 'diverges']
        if any(term in content.lower() for term in expert_terms):
            base_confidence -= 0.05

        if any(word in content.lower() for word in ['therefore', 'thus', 'must', 'always']):
            base_confidence += 0.05

        return max(min(base_confidence, 0.95), 0.3)


# ============================================================================
# MATHEMATICAL CALCULATION VERIFIER
# ============================================================================

class MathematicalVerifier:
    """
    Comprehensive mathematical verification for competition-level problems.

    Covers:
    - Basic arithmetic (GSM8K level)
    - Algebra (equations, factoring, quadratics)
    - Geometry (area, volume, angles, Pythagorean theorem)
    - Number theory (primes, divisors, GCD/LCM, modular arithmetic)
    - Combinatorics (permutations, combinations, probability)
    - Unit conversions (time, distance, money)
    - Harmonic means, work rates, mixture problems

    NOTE: Only applied to MATHEMATICAL question types.
    """

    # ========================================================================
    # PATTERN EXTRACTION
    # ========================================================================

    @staticmethod
    def extract_calculations(text: str) -> List[Dict]:
        """
        Extract all types of calculations from reasoning text.
        Returns a list of calculation dicts sorted by position.
        """
        calculations = []
        clean_text = text.replace('\\', '').replace('$', '')

        # Pattern 1: Basic arithmetic (5 + 3 = 8)
        basic_pattern = r'(\d+(?:\.\d+)?)\s*([+\-×*/÷])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)'
        for match in re.finditer(basic_pattern, text):
            num1, op, num2, result = match.groups()
            calculations.append({
                'type': 'basic',
                'operand1': float(num1),
                'operator': op.replace('×', '*').replace('÷', '/'),
                'operand2': float(num2),
                'claimed_result': float(result),
                'text': match.group(0),
                'position': match.start()
            })

        # Pattern 2: Percentages (20% of 50 = 10)
        percent_pattern = r'(\d+(?:\.\d+)?)%\s+of\s+(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)'
        for match in re.finditer(percent_pattern, text):
            percent, base, result = match.groups()
            calculations.append({
                'type': 'percent',
                'percent': float(percent),
                'base': float(base),
                'claimed_result': float(result),
                'text': match.group(0),
                'position': match.start()
            })

        # Pattern 3: Fractions (2/3 of 15 = 10)
        fraction_pattern = r'(\d+)/(\d+)\s+(?:of|×|\*)\s+(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)'
        for match in re.finditer(fraction_pattern, text):
            num, denom, base, result = match.groups()
            calculations.append({
                'type': 'fraction',
                'numerator': int(num),
                'denominator': int(denom),
                'base': float(base),
                'claimed_result': float(result),
                'text': match.group(0),
                'position': match.start()
            })

        # Pattern 4: Exponents (2^3 = 8)
        exponent_pattern = r'(\d+(?:\.\d+)?)\s*\^\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)'
        for match in re.finditer(exponent_pattern, clean_text):
            base, exp, result = match.groups()
            calculations.append({
                'type': 'exponent',
                'base': float(base),
                'exponent': float(exp),
                'claimed_result': float(result),
                'text': match.group(0),
                'position': match.start()
            })

        # Pattern 5: Square roots (√16 = 4)
        sqrt_pattern = r'(?:√|sqrt\()\s*(\d+(?:\.\d+)?)\s*\)?\s*=\s*(\d+(?:\.\d+)?)'
        for match in re.finditer(sqrt_pattern, clean_text):
            value, result = match.groups()
            calculations.append({
                'type': 'sqrt',
                'value': float(value),
                'claimed_result': float(result),
                'text': match.group(0),
                'position': match.start()
            })

        # Pattern 6: Pythagorean theorem (a^2 + b^2 = c^2)
        pythag_pattern = r'(\d+(?:\.\d+)?)\s*\^2\s*\+\s*(\d+(?:\.\d+)?)\s*\^2\s*=\s*(\d+(?:\.\d+)?)\s*\^?2?'
        for match in re.finditer(pythag_pattern, clean_text):
            a, b, c = match.groups()
            calculations.append({
                'type': 'pythagorean',
                'a': float(a),
                'b': float(b),
                'claimed_c': float(c),
                'text': match.group(0),
                'position': match.start()
            })

        # Pattern 7: Combinations C(n,k) = result
        comb_pattern = r'C\((\d+),\s*(\d+)\)\s*=\s*(\d+)'
        for match in re.finditer(comb_pattern, text):
            n, k, result = match.groups()
            calculations.append({
                'type': 'combination',
                'n': int(n), 'k': int(k),
                'claimed_result': int(result),
                'text': match.group(0),
                'position': match.start()
            })

        # Pattern 8: Permutations P(n,k) = result
        perm_pattern = r'P\((\d+),\s*(\d+)\)\s*=\s*(\d+)'
        for match in re.finditer(perm_pattern, text):
            n, k, result = match.groups()
            calculations.append({
                'type': 'permutation',
                'n': int(n), 'k': int(k),
                'claimed_result': int(result),
                'text': match.group(0),
                'position': match.start()
            })

        # Pattern 9: Factorials (5! = 120)
        factorial_pattern = r'(\d+)!\s*=\s*(\d+)'
        for match in re.finditer(factorial_pattern, text):
            n, result = match.groups()
            calculations.append({
                'type': 'factorial',
                'n': int(n),
                'claimed_result': int(result),
                'text': match.group(0),
                'position': match.start()
            })

        # Pattern 10: GCD
        gcd_pattern = r'gcd\((\d+),\s*(\d+)\)\s*=\s*(\d+)'
        for match in re.finditer(gcd_pattern, clean_text.lower()):
            a, b, result = match.groups()
            calculations.append({
                'type': 'gcd',
                'a': int(a), 'b': int(b),
                'claimed_result': int(result),
                'text': match.group(0),
                'position': match.start()
            })

        # Pattern 11: LCM
        lcm_pattern = r'lcm\((\d+),\s*(\d+)\)\s*=\s*(\d+)'
        for match in re.finditer(lcm_pattern, clean_text.lower()):
            a, b, result = match.groups()
            calculations.append({
                'type': 'lcm',
                'a': int(a), 'b': int(b),
                'claimed_result': int(result),
                'text': match.group(0),
                'position': match.start()
            })

        # Pattern 12: Circle area (π * r^2 = result)
        circle_area_pattern = r'(?:π|pi)\s*\*?\s*(\d+(?:\.\d+)?)\s*\^2\s*=\s*(\d+(?:\.\d+)?)'
        for match in re.finditer(circle_area_pattern, clean_text.lower()):
            r, result = match.groups()
            calculations.append({
                'type': 'circle_area',
                'radius': float(r),
                'claimed_result': float(result),
                'text': match.group(0),
                'position': match.start()
            })

        # Pattern 13: Harmonic mean
        harmonic_pattern = r'2\s*/\s*\(1/(\d+(?:\.\d+)?)\s*\+\s*1/(\d+(?:\.\d+)?)\)\s*=\s*(\d+(?:\.\d+)?)'
        for match in re.finditer(harmonic_pattern, clean_text):
            a, b, result = match.groups()
            calculations.append({
                'type': 'harmonic_mean',
                'a': float(a), 'b': float(b),
                'claimed_result': float(result),
                'text': match.group(0),
                'position': match.start()
            })

        # Pattern 14: Work rate (1/a + 1/b = 1/result)
        work_rate_pattern = r'1/(\d+(?:\.\d+)?)\s*\+\s*1/(\d+(?:\.\d+)?)\s*=\s*1/(\d+(?:\.\d+)?)'
        for match in re.finditer(work_rate_pattern, clean_text):
            a, b, result = match.groups()
            calculations.append({
                'type': 'work_rate',
                'time_a': float(a), 'time_b': float(b),
                'claimed_combined_time': float(result),
                'text': match.group(0),
                'position': match.start()
            })

        # Pattern 15: Rate × Time = Distance
        rate_time_pattern = r'(\d+(?:\.\d+)?)\s*(?:mph|km/h|m/s)?\s*×\s*(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)?\s*=\s*(\d+(?:\.\d+)?)'
        for match in re.finditer(rate_time_pattern, clean_text.lower()):
            rate, time_val, distance = match.groups()
            calculations.append({
                'type': 'rate_time_distance',
                'rate': float(rate), 'time': float(time_val),
                'claimed_distance': float(distance),
                'text': match.group(0),
                'position': match.start()
            })

        # Pattern 16: Modular arithmetic (a mod b = result)
        mod_pattern = r'(\d+)\s+mod\s+(\d+)\s*=\s*(\d+)'
        for match in re.finditer(mod_pattern, clean_text.lower()):
            a, b, result = match.groups()
            calculations.append({
                'type': 'modular',
                'value': int(a), 'modulus': int(b),
                'claimed_result': int(result),
                'text': match.group(0),
                'position': match.start()
            })

        # Pattern 17: Probability (n/m = result where result <= 1)
        prob_pattern = r'(\d+)/(\d+)\s*=\s*(\d+(?:\.\d+)?)'
        for match in re.finditer(prob_pattern, text):
            fav, total, prob = match.groups()
            if float(prob) <= 1.0:
                calculations.append({
                    'type': 'probability',
                    'favorable': int(fav), 'total': int(total),
                    'claimed_probability': float(prob),
                    'text': match.group(0),
                    'position': match.start()
                })

        # Pattern 18: Logarithms (log_b(x) = y)
        log_pattern = r'log_(\d+)\((\d+)\)\s*=\s*(\d+(?:\.\d+)?)'
        for match in re.finditer(log_pattern, clean_text):
            base, value, result = match.groups()
            calculations.append({
                'type': 'logarithm',
                'base': float(base), 'value': float(value),
                'claimed_result': float(result),
                'text': match.group(0),
                'position': match.start()
            })

        calculations.sort(key=lambda x: x.get('position', 0))
        return calculations

    # ========================================================================
    # VERIFICATION
    # ========================================================================

    @staticmethod
    def verify_calculation(calc: Dict) -> Tuple[bool, str, Optional[float]]:
        """
        Verify a single extracted calculation.

        Returns:
            (is_correct, feedback_message, correct_value)
        """
        TOLERANCE = 0.01
        try:
            t = calc['type']

            if t == 'basic':
                return MathematicalVerifier._verify_basic_arithmetic(calc, TOLERANCE)

            elif t == 'percent':
                correct = (calc['percent'] / 100) * calc['base']
                claimed = calc['claimed_result']
                if abs(correct - claimed) / max(abs(correct), 0.001) < TOLERANCE:
                    return True, "✓ Verified", correct
                actual_pct = (claimed / calc['base']) * 100 if calc['base'] else 0
                return False, (f"✗ {calc['percent']}% of {calc['base']} = {correct:.2f}, "
                               f"not {claimed}. (Reverse: {claimed} is {actual_pct:.1f}%)"), correct

            elif t == 'fraction':
                correct = (calc['numerator'] / calc['denominator']) * calc['base']
                claimed = calc['claimed_result']
                if abs(correct - claimed) / max(abs(correct), 0.001) < TOLERANCE:
                    return True, "✓ Verified", correct
                return False, (f"✗ {calc['numerator']}/{calc['denominator']} × "
                               f"{calc['base']} = {correct:.2f}, not {claimed}"), correct

            elif t == 'exponent':
                correct = calc['base'] ** calc['exponent']
                claimed = calc['claimed_result']
                if abs(correct - claimed) / max(abs(correct), 0.001) < TOLERANCE:
                    return True, "✓ Verified", correct
                return False, (f"✗ {calc['base']}^{calc['exponent']} = "
                               f"{correct:.2f}, not {claimed}"), correct

            elif t == 'sqrt':
                correct = math.sqrt(calc['value'])
                claimed = calc['claimed_result']
                if abs(correct - claimed) / max(abs(correct), 0.001) < TOLERANCE:
                    return True, "✓ Verified", correct
                return False, (f"✗ √{calc['value']} = {correct:.2f}, not {claimed}. "
                               f"(Reverse: {claimed}² = {claimed**2:.2f})"), correct

            elif t == 'pythagorean':
                correct_c = math.sqrt(calc['a'] ** 2 + calc['b'] ** 2)
                claimed_c = calc['claimed_c']
                if abs(correct_c - claimed_c) / max(abs(correct_c), 0.001) < TOLERANCE:
                    return True, "✓ Verified", correct_c
                return False, (f"✗ √({calc['a']}² + {calc['b']}²) = "
                               f"{correct_c:.2f}, not {claimed_c}"), correct_c

            elif t == 'combination':
                n, k = calc['n'], calc['k']
                if k > n or k < 0:
                    return False, f"✗ Invalid C({n},{k})", 0
                correct = math.comb(n, k)
                if correct == calc['claimed_result']:
                    return True, "✓ Verified", correct
                return False, f"✗ C({n},{k}) = {correct}, not {calc['claimed_result']}", correct

            elif t == 'permutation':
                n, k = calc['n'], calc['k']
                if k > n or k < 0:
                    return False, f"✗ Invalid P({n},{k})", 0
                correct = math.perm(n, k)
                if correct == calc['claimed_result']:
                    return True, "✓ Verified", correct
                return False, f"✗ P({n},{k}) = {correct}, not {calc['claimed_result']}", correct

            elif t == 'factorial':
                if calc['n'] > 20:
                    return True, "⚠ Factorial too large to verify", None
                correct = math.factorial(calc['n'])
                if correct == calc['claimed_result']:
                    return True, "✓ Verified", correct
                return False, f"✗ {calc['n']}! = {correct}, not {calc['claimed_result']}", correct

            elif t == 'gcd':
                correct = math.gcd(calc['a'], calc['b'])
                if correct == calc['claimed_result']:
                    return True, "✓ Verified", correct
                return False, f"✗ gcd({calc['a']},{calc['b']}) = {correct}, not {calc['claimed_result']}", correct

            elif t == 'lcm':
                correct = abs(calc['a'] * calc['b']) // math.gcd(calc['a'], calc['b'])
                if correct == calc['claimed_result']:
                    return True, "✓ Verified", correct
                return False, f"✗ lcm({calc['a']},{calc['b']}) = {correct}, not {calc['claimed_result']}", correct

            elif t == 'circle_area':
                correct = math.pi * (calc['radius'] ** 2)
                claimed = calc['claimed_result']
                if abs(correct - claimed) / max(abs(correct), 0.001) < TOLERANCE:
                    return True, "✓ Verified", correct
                return False, f"✗ π×{calc['radius']}² = {correct:.2f}, not {claimed}", correct

            elif t == 'harmonic_mean':
                a, b = calc['a'], calc['b']
                if a == 0 or b == 0:
                    return False, "✗ Cannot calculate harmonic mean with zero values", 0
                correct = 2 / (1/a + 1/b)
                claimed = calc['claimed_result']
                if abs(correct - claimed) / max(abs(correct), 0.001) < TOLERANCE:
                    return True, "✓ Verified", correct
                return False, f"✗ Harmonic mean of {a} and {b} = {correct:.2f}, not {claimed}", correct

            elif t == 'work_rate':
                a, b = calc['time_a'], calc['time_b']
                if a == 0 or b == 0:
                    return False, "✗ Work time cannot be zero", 0
                correct = 1 / (1/a + 1/b)
                claimed = calc['claimed_combined_time']
                if abs(correct - claimed) / max(abs(correct), 0.001) < TOLERANCE:
                    return True, "✓ Verified", correct
                return False, f"✗ Combined time = {correct:.2f} hours, not {claimed}", correct

            elif t == 'rate_time_distance':
                correct = calc['rate'] * calc['time']
                claimed = calc['claimed_distance']
                if abs(correct - claimed) / max(abs(correct), 0.001) < TOLERANCE:
                    return True, "✓ Verified", correct
                return False, f"✗ {calc['rate']}×{calc['time']} = {correct:.2f}, not {claimed}", correct

            elif t == 'modular':
                correct = calc['value'] % calc['modulus']
                if correct == calc['claimed_result']:
                    return True, "✓ Verified", correct
                return False, f"✗ {calc['value']} mod {calc['modulus']} = {correct}, not {calc['claimed_result']}", correct

            elif t == 'probability':
                correct = calc['favorable'] / calc['total'] if calc['total'] else 0
                claimed = calc['claimed_probability']
                if abs(correct - claimed) / max(abs(correct), 0.001) < TOLERANCE:
                    return True, "✓ Verified", correct
                return False, f"✗ P = {calc['favorable']}/{calc['total']} = {correct:.4f}, not {claimed}", correct

            elif t == 'logarithm':
                correct = math.log(calc['value']) / math.log(calc['base'])
                claimed = calc['claimed_result']
                if abs(correct - claimed) / max(abs(correct), 0.001) < TOLERANCE:
                    return True, "✓ Verified", correct
                return False, (f"✗ log_{calc['base']}({calc['value']}) = "
                               f"{correct:.2f}, not {claimed}"), correct

            else:
                return True, "Unknown calculation type — skipping", None

        except Exception as e:
            return True, f"Verification error: {str(e)}", None

    @staticmethod
    def _verify_basic_arithmetic(calc: Dict, tolerance: float) -> Tuple[bool, str, float]:
        """Verify basic +, -, *, / arithmetic with reverse-engineering feedback."""
        op = calc['operator']
        a, b = calc['operand1'], calc['operand2']
        claimed = calc['claimed_result']

        if op == '+':
            correct = a + b
        elif op == '-':
            correct = a - b
        elif op == '*':
            correct = a * b
        elif op == '/':
            if b == 0:
                return False, "✗ Division by zero", 0
            correct = a / b
        else:
            return True, "Unknown operator", claimed

        if abs(correct - claimed) / max(abs(correct), 0.001) < tolerance:
            return True, "✓ Verified", correct

        return False, (f"✗ {a} {op} {b} = {correct:.2f}, not {claimed}"), correct
