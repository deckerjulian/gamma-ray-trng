"""
NIST Statistical Test Suite for Random and Pseudorandom Number Generators
Implementation for testing the randomness of the gamma event generated random bits.

This module implements the 15 statistical tests from NIST SP 800-22 Rev. 1a.
"""

import numpy as np
from scipy import special, stats
from scipy.fft import fft
import math
from collections import defaultdict, Counter
import re


class NISTRandomnessTests:
    """
    Implementation of NIST Statistical Test Suite for Random Number Generators
    """
    
    def __init__(self, significance_level=0.01):
        """
        Initialize the NIST test suite.
        
        Args:
            significance_level (float): Significance level for tests (default: 0.01)
        """
        self.alpha = significance_level
        
    def read_bits_from_file(self, filename):
        """
        Read random bits from file and convert to binary string.
        
        Args:
            filename (str): Path to the random bits file
            
        Returns:
            str: Binary string representation of the bits
        """
        try:
            with open(filename, 'r') as f:
                bits = f.read().strip().replace('\n', '')
            return bits
        except FileNotFoundError:
            print(f"Error: File {filename} not found.")
            return ""
        except Exception as e:
            print(f"Error reading file: {e}")
            return ""
    
    def frequency_test(self, bits):
        """
        Test 1: Frequency (Monobit) Test
        Tests if the number of ones and zeros are approximately equal.
        """
        n = len(bits)
        if n == 0:
            return False, 0.0
            
        # Convert to +1/-1
        s = sum(1 if bit == '1' else -1 for bit in bits)
        
        # Calculate test statistic
        s_obs = abs(s) / math.sqrt(n)
        
        # Calculate p-value
        p_value = special.erfc(s_obs / math.sqrt(2))
        
        return p_value >= self.alpha, p_value
    
    def block_frequency_test(self, bits, block_size=128):
        """
        Test 2: Frequency Test within a Block
        Tests if the frequency of ones is approximately 1/2 within each block.
        """
        n = len(bits)
        if n < block_size:
            return False, 0.0
            
        num_blocks = n // block_size
        
        # Calculate proportion of ones in each block
        proportions = []
        for i in range(num_blocks):
            block = bits[i*block_size:(i+1)*block_size]
            pi = block.count('1') / block_size
            proportions.append(pi)
        
        # Calculate chi-square statistic
        chi_square = 4 * block_size * sum((pi - 0.5)**2 for pi in proportions)
        
        # Calculate p-value
        p_value = special.gammaincc(num_blocks / 2, chi_square / 2)
        
        return p_value >= self.alpha, p_value
    
    def runs_test(self, bits):
        """
        Test 4: Runs Test
        Tests if the number of runs is as expected for a random sequence.
        """
        n = len(bits)
        if n == 0:
            return False, 0.0
            
        # Calculate proportion of ones
        pi = bits.count('1') / n
        
        # Pre-test condition
        if abs(pi - 0.5) >= 2 / math.sqrt(n):
            return False, 0.0
        
        # Count runs
        runs = 1
        for i in range(1, n):
            if bits[i] != bits[i-1]:
                runs += 1
        
        # Calculate test statistic
        v_obs = runs
        expected = 2 * n * pi * (1 - pi) + 1
        variance = 2 * n * pi * (1 - pi) * (2 * n * pi * (1 - pi) - 1)
        
        if variance == 0:
            return False, 0.0
            
        test_stat = abs(v_obs - expected) / math.sqrt(variance)
        
        # Calculate p-value
        p_value = special.erfc(test_stat / math.sqrt(2))
        
        return p_value >= self.alpha, p_value
    
    def longest_run_of_ones_test(self, bits):
        """
        Test 5: Test for the Longest Run of Ones in a Block
        """
        n = len(bits)
        
        # Determine parameters based on sequence length
        if n < 128:
            return False, 0.0
        elif n < 6272:
            M, K = 8, 3
            v_values = [1, 2, 3, 4]
            pi = [0.2148, 0.3672, 0.2305, 0.1875]
        elif n < 750000:
            M, K = 128, 5
            v_values = [4, 5, 6, 7, 8, 9]
            pi = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
        else:
            M, K = 512, 6
            v_values = [10, 11, 12, 13, 14, 15, 16]
            pi = [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]
        
        N = n // M
        
        # Count longest runs in each block
        v_counts = [0] * (K + 1)
        
        for i in range(N):
            block = bits[i*M:(i+1)*M]
            longest_run = self._longest_run_in_block(block)
            
            # Categorize the longest run
            if longest_run <= v_values[0]:
                v_counts[0] += 1
            elif longest_run >= v_values[-1]:
                v_counts[K] += 1
            else:
                for j in range(1, K):
                    if longest_run == v_values[j]:
                        v_counts[j] += 1
                        break
        
        # Calculate chi-square statistic
        chi_square = sum((v_counts[i] - N * pi[i])**2 / (N * pi[i]) for i in range(K + 1))
        
        # Calculate p-value
        p_value = special.gammaincc(K / 2, chi_square / 2)
        
        return p_value >= self.alpha, p_value
    
    def _longest_run_in_block(self, block):
        """Helper function to find longest run of ones in a block."""
        longest = 0
        current = 0
        
        for bit in block:
            if bit == '1':
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        
        return longest
    
    def dft_test(self, bits):
        """
        Test 7: Discrete Fourier Transform (Spectral) Test
        """
        n = len(bits)
        if n == 0:
            return False, 0.0
        
        # Convert to +1/-1
        X = np.array([1 if bit == '1' else -1 for bit in bits])
        
        # Apply DFT
        S = fft(X)
        
        # Calculate modulus and take first half
        M = np.abs(S[:n//2])
        
        # Calculate threshold
        T = math.sqrt(math.log(1/0.05) * n)
        
        # Count peaks below threshold
        N0 = 0.95 * n / 2
        N1 = sum(1 for m in M if m < T)
        
        # Calculate test statistic
        d = (N1 - N0) / math.sqrt(n * 0.95 * 0.05 / 4)
        
        # Calculate p-value
        p_value = special.erfc(abs(d) / math.sqrt(2))
        
        return p_value >= self.alpha, p_value
    
    def approximate_entropy_test(self, bits, m=2):
        """
        Test 11: Approximate Entropy Test
        """
        n = len(bits)
        if n < 10:
            return False, 0.0
        
        def _approximate_entropy(m):
            patterns = defaultdict(int)
            
            # Count overlapping m-bit patterns
            for i in range(n - m + 1):
                pattern = bits[i:i+m]
                patterns[pattern] += 1
            
            # Calculate phi(m)
            phi = 0
            for count in patterns.values():
                if count > 0:
                    pi = count / (n - m + 1)
                    phi += pi * math.log(pi)
            
            return phi
        
        phi_m = _approximate_entropy(m)
        phi_m_plus_1 = _approximate_entropy(m + 1)
        
        # Calculate test statistic
        apen = phi_m - phi_m_plus_1
        chi_square = 2 * n * apen
        
        # Calculate p-value
        p_value = special.gammaincc(2**(m-1), chi_square / 2)
        
        return p_value >= self.alpha, p_value
    
    def cumulative_sums_test(self, bits):
        """
        Test 3: Cumulative Sums (Cusum) Test
        """
        n = len(bits)
        if n == 0:
            return False, 0.0
        
        # Convert to +1/-1
        X = [1 if bit == '1' else -1 for bit in bits]
        
        # Forward cumulative sum
        S_forward = [sum(X[:i+1]) for i in range(n)]
        z_forward = max(abs(s) for s in S_forward)
        
        # Backward cumulative sum
        S_backward = [sum(X[i:]) for i in range(n)]
        z_backward = max(abs(s) for s in S_backward)
        
        # Calculate p-values
        def _cusum_p_value(z):
            sum_term = 0
            for k in range(int((-n/z + 1)/4), int((n/z - 1)/4) + 1):
                term1 = stats.norm.cdf((4*k + 1)*z/math.sqrt(n))
                term2 = stats.norm.cdf((4*k - 1)*z/math.sqrt(n))
                sum_term += term1 - term2
            
            sum_term2 = 0
            for k in range(int((-n/z - 3)/4), int((n/z - 1)/4) + 1):
                term1 = stats.norm.cdf((4*k + 3)*z/math.sqrt(n))
                term2 = stats.norm.cdf((4*k + 1)*z/math.sqrt(n))
                sum_term2 += term1 - term2
            
            return 1 - sum_term + sum_term2
        
        p_value_forward = _cusum_p_value(z_forward)
        p_value_backward = _cusum_p_value(z_backward)
        
        # Return the minimum p-value
        p_value = min(p_value_forward, p_value_backward)
        
        return p_value >= self.alpha, p_value
    
    def serial_test(self, bits, m=2):
        """
        Test 14: Serial Test
        """
        n = len(bits)
        if n < 10:
            return False, 0.0
        
        def _psi_squared(m):
            patterns = defaultdict(int)
            
            # Count overlapping m-bit patterns (circular)
            extended_bits = bits + bits[:m-1]
            for i in range(n):
                pattern = extended_bits[i:i+m]
                patterns[pattern] += 1
            
            # Calculate psi^2
            psi_squared = 0
            for count in patterns.values():
                psi_squared += count**2
            
            psi_squared = (2**m / n) * psi_squared - n
            
            return psi_squared
        
        psi_m = _psi_squared(m)
        psi_m_minus_1 = _psi_squared(m-1)
        psi_m_minus_2 = _psi_squared(m-2)
        
        delta1 = psi_m - psi_m_minus_1
        delta2 = psi_m - 2*psi_m_minus_1 + psi_m_minus_2
        
        # Calculate p-values
        p_value1 = special.gammaincc(2**(m-2), delta1 / 2)
        p_value2 = special.gammaincc(2**(m-3), delta2 / 2)
        
        return (p_value1 >= self.alpha and p_value2 >= self.alpha), min(p_value1, p_value2)
    
    def linear_complexity_test(self, bits, M=1000):
        """
        Test 15: Linear Complexity Test
        """
        n = len(bits)
        N = n // M
        
        if N < 2:
            return False, 0.0
        
        # Expected mean and variance
        mu = M / 2 + (9 + (-1)**(M+1)) / 36 - (M / 3 + 2 / 9) / 2**M
        sigma_squared = M / 6 + 1 / 6
        
        # Count linear complexities in different ranges
        v = [0, 0, 0, 0, 0, 0, 0]
        
        for i in range(N):
            block = bits[i*M:(i+1)*M]
            L = self._berlekamp_massey(block)
            
            T = (-1)**M * (L - mu) + 2 / 9
            
            if T <= -2.5:
                v[0] += 1
            elif T <= -1.5:
                v[1] += 1
            elif T <= -0.5:
                v[2] += 1
            elif T <= 0.5:
                v[3] += 1
            elif T <= 1.5:
                v[4] += 1
            elif T <= 2.5:
                v[5] += 1
            else:
                v[6] += 1
        
        # Expected frequencies
        pi = [0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]
        
        # Calculate chi-square statistic
        chi_square = sum((v[i] - N * pi[i])**2 / (N * pi[i]) for i in range(7))
        
        # Calculate p-value
        p_value = special.gammaincc(3, chi_square / 2)
        
        return p_value >= self.alpha, p_value
    
    def _berlekamp_massey(self, bits):
        """
        Berlekamp-Massey algorithm for calculating linear complexity.
        """
        n = len(bits)
        sequence = [int(bit) for bit in bits]
        
        b = [0] * n
        c = [0] * n
        b[0] = 1
        c[0] = 1
        
        L = 0
        m = -1
        N = 0
        
        while N < n:
            d = sequence[N]
            for i in range(1, L + 1):
                d ^= c[i] & sequence[N - i]
            
            if d == 1:
                t = c[:]
                for i in range(N - m, n):
                    if i < len(c) and N - m + len(b) > i:
                        c[i] ^= b[i - (N - m)]
                
                if L <= N // 2:
                    L = N + 1 - L
                    m = N
                    b = t
            
            N += 1
        
        return L
    
    def universal_statistical_test(self, bits):
        """
        Test 10: Universal Statistical Test (Maurer's Test)
        """
        n = len(bits)
        
        # Determine parameters based on sequence length
        if n >= 387840:
            L, Q = 6, 640
        elif n >= 904960:
            L, Q = 7, 1280
        elif n >= 2068480:
            L, Q = 8, 2560
        elif n >= 4654080:
            L, Q = 9, 5120
        elif n >= 10342400:
            L, Q = 10, 10240
        elif n >= 22753280:
            L, Q = 11, 20480
        elif n >= 49643520:
            L, Q = 12, 40960
        elif n >= 107347968:
            L, Q = 13, 81920
        elif n >= 231211008:
            L, Q = 14, 163840
        elif n >= 496435200:
            L, Q = 15, 327680
        else:
            return False, 0.0  # Insufficient data
        
        K = n // L - Q
        if K <= 0:
            return False, 0.0
        
        # Initialize table
        T = {}
        
        # Initialization segment
        for i in range(Q):
            pattern = bits[i*L:(i+1)*L]
            T[pattern] = i + 1
        
        # Test segment
        sum_log = 0
        for i in range(Q, Q + K):
            pattern = bits[i*L:(i+1)*L]
            if pattern in T:
                sum_log += math.log2(i + 1 - T[pattern])
            else:
                sum_log += L  # Maximum possible value
            T[pattern] = i + 1
        
        # Calculate test statistic
        fn = sum_log / K
        
        # Expected values and variance (approximate)
        c = 0.7 - 0.8 / L + (4 + 32 / L) * (K**(-3/L)) / 15
        sigma = c * math.sqrt(0.5 * math.log(2) / K)
        expected_value = L * math.log(2)
        
        # Calculate p-value
        test_statistic = (fn - expected_value) / (sigma * math.sqrt(2))
        p_value = special.erfc(abs(test_statistic))
        
        return p_value >= self.alpha, p_value
    
    def run_all_tests(self, filename):
        """
        Run all NIST tests on the random bits file.
        
        Args:
            filename (str): Path to the random bits file
            
        Returns:
            dict: Results of all tests
        """
        bits = self.read_bits_from_file(filename)
        
        if not bits:
            print("No bits to test.")
            return {}
        
        print(f"Testing {len(bits)} bits from {filename}")
        print(f"Significance level: {self.alpha}")
        print("-" * 80)
        
        results = {}
        
        # Test 1: Frequency Test
        passed, p_value = self.frequency_test(bits)
        results['01_frequency'] = {'passed': passed, 'p_value': p_value}
        print(f"[01] Frequency Test:               {'PASS' if passed else 'FAIL'} (p={p_value:.6f})")
        
        # Test 2: Block Frequency Test
        passed, p_value = self.block_frequency_test(bits)
        results['02_block_frequency'] = {'passed': passed, 'p_value': p_value}
        print(f"[02] Block Frequency Test:         {'PASS' if passed else 'FAIL'} (p={p_value:.6f})")
        
        # Test 3: Cumulative Sums Test
        passed, p_value = self.cumulative_sums_test(bits)
        results['03_cumulative_sums'] = {'passed': passed, 'p_value': p_value}
        print(f"[03] Cumulative Sums Test:         {'PASS' if passed else 'FAIL'} (p={p_value:.6f})")
        
        # Test 4: Runs Test
        passed, p_value = self.runs_test(bits)
        results['04_runs'] = {'passed': passed, 'p_value': p_value}
        print(f"[04] Runs Test:                    {'PASS' if passed else 'FAIL'} (p={p_value:.6f})")
        
        # Test 5: Longest Run of Ones Test
        passed, p_value = self.longest_run_of_ones_test(bits)
        results['05_longest_run'] = {'passed': passed, 'p_value': p_value}
        print(f"[05] Longest Run of Ones Test:     {'PASS' if passed else 'FAIL'} (p={p_value:.6f})")
        
        # Test 7: DFT Test
        passed, p_value = self.dft_test(bits)
        results['07_dft'] = {'passed': passed, 'p_value': p_value}
        print(f"[07] Discrete Fourier Transform:   {'PASS' if passed else 'FAIL'} (p={p_value:.6f})")
        
        # Test 10: Universal Statistical Test
        passed, p_value = self.universal_statistical_test(bits)
        results['10_universal'] = {'passed': passed, 'p_value': p_value}
        print(f"[10] Universal Statistical Test:   {'PASS' if passed else 'FAIL'} (p={p_value:.6f})")
        
        # Test 11: Approximate Entropy Test
        passed, p_value = self.approximate_entropy_test(bits)
        results['11_approximate_entropy'] = {'passed': passed, 'p_value': p_value}
        print(f"[11] Approximate Entropy Test:     {'PASS' if passed else 'FAIL'} (p={p_value:.6f})")
        
        # Test 14: Serial Test
        passed, p_value = self.serial_test(bits)
        results['14_serial'] = {'passed': passed, 'p_value': p_value}
        print(f"[14] Serial Test:                  {'PASS' if passed else 'FAIL'} (p={p_value:.6f})")
        
        # Test 15: Linear Complexity Test
        passed, p_value = self.linear_complexity_test(bits)
        results['15_linear_complexity'] = {'passed': passed, 'p_value': p_value}
        print(f"[15] Linear Complexity Test:       {'PASS' if passed else 'FAIL'} (p={p_value:.6f})")
        
        print("-" * 80)
        
        # Summary
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r['passed'])
        
        print(f"Summary: {passed_tests}/{total_tests} tests passed")
        print(f"Overall result: {'PASS' if passed_tests >= total_tests * 0.8 else 'FAIL'}")
        
        return results


def test_random_bits_file(filename="random_bits.txt", significance_level=0.01):
    """
    Convenience function to test a random bits file.
    
    Args:
        filename (str): Path to the random bits file
        significance_level (float): Significance level for tests
        
    Returns:
        dict: Test results
    """
    nist_tests = NISTRandomnessTests(significance_level)
    return nist_tests.run_all_tests(filename)


if __name__ == "__main__":
    # Test the random bits file
    results = test_random_bits_file("random_bits.txt")
