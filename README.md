# numMethods

## CH2120 Numerical Methods Assignments

This repository contains coursework assignments completed for the course **CH2120**. The assignments focus on numerical methods, computational mathematics, number theory, and scientific programming using Python.

---

## Author

**Talla Shreyas**  
**Roll Number:** ES24BTECH11037  
**Course:** CH2120

---

## Repository Structure

```text
numMethods/
│
├── assignment2(1).py
├── assignment3.py
├── Prime numbers Assignment (1).pdf
└── README.md
```

---

## Assignment 1: Prime Numbers and Number Theory

Assignment 1 focuses on prime numbers, their importance in cryptography, and several advanced number theory problems.

The assignment discusses the role of prime numbers in cryptography, especially in systems like RSA, where multiplying large prime numbers is computationally easy but factoring their product is difficult. It also covers prime gaps, prime-counting approximations, Fermat numbers, Willans' formula, and efficient primality testing methods such as the Sieve of Eratosthenes and Miller-Rabin primality testing.

### Topics Covered

- Prime numbers in cryptography
- RSA-based motivation for prime usage
- Prime-counting approximation
- Prime gaps and conjectures
- Fermat numbers
- Willans' formula
- Sieve of Eratosthenes
- Miller-Rabin primality test
- Mersenne primes
- Brocard's conjecture
- Palindromic primes
- Perfect numbers
- Open problems in number theory

### Problems Included

The assignment contains problems related to:

1. Finding the next prime following a specific digit pattern.
2. Finding repunit primes of the form:

   ```text
   (10^N - 1) / 9
   ```

3. Finding Mersenne primes for prime exponents between 2201 and 2299.
4. Verifying Brocard's conjecture for selected primes.
5. Finding a palindromic prime with at least 50 digits.
6. Using Mersenne primes to construct perfect numbers.
7. Exploring an open problem in prime numbers such as:
   - Wieferich primes
   - Goldbach's conjecture
   - Weak Goldbach problem
   - Legendre's conjecture
   - Oppermann's conjecture

### Note

The code for Assignment 1 is currently not included in this repository. However, the assignment problem statement and theoretical background are included in the PDF file.

---

## Assignment 2: Harshad Numbers and Shifted Legendre Polynomial Toolbox

Assignment 2 is implemented as a PyQt5-based desktop application. It provides a graphical interface for working with Harshad numbers and shifted Legendre polynomials.

The application contains three main modules:

1. Factorial Harshad number checking
2. Consecutive Harshad number sequences
3. Shifted Legendre polynomial computations on the interval `[0, 1]`

---

### 1. Factorial Harshad Numbers

This module checks whether factorials are Harshad numbers.

A Harshad number is an integer that is divisible by the sum of its digits.

For a given range of values, the program checks whether:

```text
n!
```

is a Harshad number or not.

The user can input:

- Starting value of `n`
- Ending value of `n`
- Number of non-Harshad factorials to find

The program displays whether each factorial is Harshad or not and summarizes the non-Harshad factorials found.

---

### 2. Consecutive Harshad Numbers

This module searches for sequences of consecutive Harshad numbers.

The user enters a range `[a, b]`, and the application computes sequences of consecutive Harshad numbers for each value of `n` in that range.

The program includes:

- Trivial sequences
- Non-trivial known sequences
- Simulated search for larger sequences
- Analytical explanation for why very long consecutive Harshad runs are computationally difficult

It also explains why finding very long runs, such as 20 consecutive Harshad numbers, is highly unlikely due to the irregular behavior of digit sums and divisibility.

---

### 3. Shifted Legendre Polynomial Tools

This module works with shifted Legendre polynomials on the interval `[0, 1]`.

The shifted Legendre polynomial is defined as:

```text
Q_n(x) = P_n(2x - 1)
```

where `P_n` is the ordinary Legendre polynomial.

The program computes:

- Coefficients of the nth shifted Legendre polynomial
- Companion matrix of the polynomial
- LU decomposition of the companion matrix
- Eigenvalues of the companion matrix
- Roots of the polynomial
- Solution of a linear system `Ax = b`
- Smallest and largest roots refined using Newton-Raphson method

The implementation uses exact rational arithmetic through Python's `Fraction` class to avoid numerical overflow and instability for large values of `n`.

---

## Assignment 3: Gauss-Legendre Quadrature and Heat Equation Solver

Assignment 3 is implemented as a PyQt5 GUI application. It focuses on Gauss-Legendre quadrature, differentiation matrices, and solving a boundary value problem using spectral collocation.

The application contains two major computational parts:

1. Gauss-Legendre quadrature and differentiation matrices
2. Heat equation boundary value problem solver

---

### 1. Gauss-Legendre Quadrature

This part computes Gauss-Legendre nodes and weights using the Golub-Welsch algorithm.

The method constructs a Jacobi tridiagonal matrix whose eigenvalues give the Gauss-Legendre nodes. The weights are computed using the first components of the normalized eigenvectors.

The application also compares weights computed using:

- Eigenvector method
- Lagrange moment method

### Features

- Compute Gauss-Legendre nodes
- Compute quadrature weights
- Compare eigenvector-based weights with Lagrange-based weights
- Plot weights versus roots
- Display eigenvector norms
- Export nodes, weights, and eigenvector data to CSV files

---

### 2. Differentiation Matrices

The program constructs numerical differentiation matrices using barycentric interpolation.

It computes:

- First derivative matrix `D1`
- Second derivative matrix `D2`

These matrices are used for spectral collocation methods.

The user can export the computed matrices as CSV files.

---

### 3. Heat Equation Solver

The second major part solves a boundary value problem related to the heat equation using spectral collocation.

The implemented differential equation is:

```text
f''(η) + 2ηf'(η) = 0
```

with boundary conditions:

```text
f(0) = 0
f(η_max) = 1
```

The numerical solution is compared with the analytical solution:

```text
erf(η)
```

where `erf` is the error function.

### Features

- Uses Gauss-Legendre collocation nodes
- Builds first and second differentiation matrices
- Solves the resulting linear system
- Compares numerical and analytical solutions
- Computes absolute error
- Plots numerical solution vs analytical solution
- Plots absolute error distribution
- Displays error table
- Exports matrices to CSV

---

## Technologies Used

```text
Python
PyQt5
NumPy
SciPy
Matplotlib
gmpy2
fractions
math
sys
```

---

## Installation

Before running the Python files, install the required libraries.

```bash
pip install numpy scipy matplotlib pyqt5 gmpy2
```

If `gmpy2` causes installation issues, try installing it using Conda:

```bash
conda install gmpy2
```

---

## How to Run

### Run Assignment 2

```bash
python "assignment2(1).py"
```

### Run Assignment 3

```bash
python assignment3.py
```

---

## Output

The programs provide graphical interfaces where the user can enter inputs and view results.

The outputs include:

- Harshad number checks
- Consecutive Harshad sequences
- Shifted Legendre polynomial coefficients
- Companion matrices
- LU decompositions
- Eigenvalues and roots
- Gauss-Legendre nodes and weights
- Differentiation matrices
- Numerical solution of the heat equation
- Error plots and tables
- CSV exports

---

## Learning Outcomes

Through these assignments, the following concepts were implemented and studied:

- Prime number theory
- Cryptographic relevance of primes
- Primality testing
- Mersenne primes
- Perfect numbers
- Harshad numbers
- Consecutive number sequences
- Shifted Legendre polynomials
- Companion matrices
- LU decomposition
- Eigenvalue computation
- Newton-Raphson refinement
- Gauss-Legendre quadrature
- Golub-Welsch algorithm
- Barycentric interpolation
- Differentiation matrices
- Spectral collocation
- Numerical solution of boundary value problems
- Error analysis and visualization

---

## Notes

- Assignment 1 code is not available, but the assignment PDF is included.
- Assignment 2 and Assignment 3 are GUI-based Python applications.
- The programs are designed for academic and learning purposes.
- Some computations involving very large numbers or high-degree polynomials may take time depending on the input size and system performance.

---

## License

This repository is created for academic coursework under CH2120.
