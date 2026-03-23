Makine Öğrenmesi, Matrisler, Özdeğer ve Özvektör İlişkisi

Makine öğrenmesi büyük ölçüde lineer cebire dayanır. Veriler genellikle matrisler şeklinde temsil edilir.

Her satır → bir veri örneği
Her sütun → bir özellik (feature)
Matris Manipülasyonu

Makine öğrenmesinde:

Veri dönüşümleri
Ölçekleme (normalization)
Lineer modeller
gibi işlemler matris çarpımı ve işlemleri ile yapılır.
Özdeğerler ve Özvektörler

Bir kare matris için:

𝐴
𝑣
=
𝜆
𝑣
Av=λv
v → özvektör
λ (lambda) → özdeğer

Bu ifade, dönüşüm sonrası vektörün yönünün değişmediğini gösterir.

Makine Öğrenmesinde Kullanım Alanları

1. PCA (Principal Component Analysis)
   
Boyut indirgeme yöntemi
Kovaryans matrisinin özdeğerleri/özvektörleri kullanılır
En büyük özdeğer → en önemli bilgi yönü

2. SVD (Singular Value Decomposition)
   
Veri sıkıştırma
Gürültü azaltma

3. Spektral Clustering
   
Graf tabanlı kümeleme
Laplacian matrisinin özvektörleri kullanılır

4. Lineer Dönüşümler
   
Veri uzayının yeniden yönlendirilmesi
