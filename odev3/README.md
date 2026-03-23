# Eigenvalues & Eigenvectors – Machine Learning Lab 3

## 1. Matrix Manipulation, Eigenvalues and Eigenvectors in Machine Learning

In machine learning, data is usually represented as matrices. Each row represents a data sample and each column represents a feature.

Matrix manipulation is used in many operations such as transforming data, scaling features, and training models.

Eigenvalues and eigenvectors are special properties of square matrices. They satisfy the equation:

A · v = λ · v

Here:

* v is the eigenvector (a direction)
* λ is the eigenvalue (a scaling factor)

This means when a matrix transforms a vector, the eigenvector does not change direction, only its length changes.

### Where they are used:

* **PCA (Principal Component Analysis):** Used for reducing dimensions. The largest eigenvalues show the most important directions in data.
* **SVD (Singular Value Decomposition):** Used in data compression and noise reduction.
* **Spectral Clustering:** Uses eigenvectors of matrices to group similar data.

---

## 2. Numpy `linalg.eig` Function

The function `numpy.linalg.eig` is used to compute eigenvalues and eigenvectors of a square matrix.

Example usage:

```python
import numpy as np

A = np.array([[4, 2],
              [1, 3]])

values, vectors = np.linalg.eig(A)
```

### How it works:

* It takes a square matrix as input.
* It returns:

  * Eigenvalues
  * Eigenvectors

Internally, it uses optimized numerical methods from the LAPACK library, mainly based on the QR algorithm.

Steps (simplified):

1. Matrix is transformed into a simpler form
2. QR decomposition is applied
3. Iterative process finds eigenvalues

---

## 3. Manual Eigenvalue Calculation and Comparison

Instead of using NumPy, eigenvalues can be calculated manually using the characteristic equation:

det(A − λI) = 0

For the matrix:

A = [[4, 2],
[1, 3]]

The equation becomes:

(4 − λ)(3 − λ) − 2 = 0

This simplifies to:

λ² − 7λ + 10 = 0

Solving this equation gives the eigenvalues.

### Comparison

* Manual calculation gives the same eigenvalues as NumPy
* NumPy is faster and works for large matrices
* Small differences may occur due to numerical precision

---

## Conclusion

Eigenvalues and eigenvectors are important tools in machine learning. They help in understanding data structure, reducing dimensions, and improving model performance.

Manual calculation helps understanding the concept, while libraries like NumPy make computation efficient.

---

## References

## References

* Introduction to Matrices for Machine Learning:
  https://machinelearningmastery.com/introduction-matrices-machine-learning/

* Introduction to Eigenvalues and Eigenvectors:
  https://machinelearningmastery.com/introduction-to-eigendecomposition-eigenvalues-and-eigenvectors/

* NumPy `eig` Documentation:
  https://numpy.org/doc/2.1/reference/generated/numpy.linalg.eig.html

* NumPy Source Code:
  https://github.com/numpy/numpy/tree/main/numpy/linalg

* Eigenvalues Implementation (Reference Repository):
  https://github.com/LucasBN/Eigenvalues-and-Eigenvectors

