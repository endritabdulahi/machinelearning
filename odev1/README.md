# HMM ile İzole Kelime Tanıma Sistemi

## Problem
Bu projede Hidden Markov Model (HMM) kullanarak basit bir konuşma tanıma sistemi simüle edilmiştir. 
Sistem iki kelimeyi ayırt eder: EV ve OKUL.

## Veri
Gerçek ses verisi yerine temsilî gözlem dizileri kullanılmıştır.

High = 0  
Low = 1

## Yöntem
Her kelime için ayrı bir Hidden Markov Model tanımlanmıştır.

Model parametreleri:
- Geçiş olasılıkları
- Emisyon olasılıkları
- Başlangıç olasılıkları

Yeni gelen gözlem dizisi her model için log-likelihood hesaplanarak sınıflandırılır.

## Sonuç
Test gözlem dizisinin hangi modele daha yüksek olasılık verdiğine göre kelime tahmini yapılır.

## Kullanım

Kurulum:

pip install -r requirements.txt

Çalıştırma:

python src/classifier.py

