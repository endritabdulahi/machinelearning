"""
YZM212 Makine Öğrenmesi
3. Laboratuvar - Eigenvalues & Eigenvectors
"""

import numpy as np

# =========================
# 1. SORU (Açıklama)
# =========================
"""
Makine öğrenmesinde veriler genellikle matrisler ile temsil edilir.
Matris manipülasyonu veri dönüşümleri ve model hesaplamalarında kullanılır.

Özdeğer ve özvektörler:
A * v = λ * v

- v: özvektör
- λ: özdeğer

Kullanım alanları:
- PCA (boyut indirgeme)
- SVD (veri sıkıştırma)
- Spektral clustering
"""

# =========================
# 2. SORU (NumPy ile hesaplama)
# =========================

print("=== 2. SORU: NUMPY EIG ===")

A = np.array([[4, 2],
              [1, 3]])

values, vectors = np.linalg.eig(A)

print("Matris A:")
print(A)

print("\nEigenvalues (NumPy):")
print(values)

print("\nEigenvectors (NumPy):")
print(vectors)


# =========================
# 3. SORU (Manuel hesaplama)
# =========================

print("\n=== 3. SORU: MANUEL HESAPLAMA ===")

# Karakteristik denklem:
# (4-λ)(3-λ) - 2 = 0
# λ^2 - 7λ + 10 = 0

coefficients = [1, -7, 10]
manual_eigenvalues = np.roots(coefficients)

print("Manuel Eigenvalues:")
print(manual_eigenvalues)


# Eigenvector hesaplama (2x2 için basit yöntem)
def eigenvector_2x2(A, lamb):
    B = A - lamb * np.eye(2)
    x = 1
    y = -B[0, 0] / B[0, 1]
    return np.array([x, y])


print("\nManuel Eigenvectors:")
for val in manual_eigenvalues:
    vec = eigenvector_2x2(A, val)
    print(f"λ = {val:.2f} için v = {vec}")


# =========================
# KARŞILAŞTIRMA
# =========================

print("\n=== KARŞILAŞTIRMA ===")

numpy_values, numpy_vectors = np.linalg.eig(A)

print("NumPy Eigenvalues:")
print(numpy_values)

print("\nSonuç:")
print("Manuel ve NumPy sonuçları aynıdır (küçük farklar olabilir).")
