import numpy as np
import tensorflow as tf
from tensorflow import keras

# Datasets
X = tf.range(10)
dataset = tf.data.Dataset.from_tensor_slices(X)
print(dataset)

for item in dataset:
    print(item)

# 연쇄 변환
dataset = dataset.repeat(3).batch(7)
for item in dataset:
    print(item)

dataset = dataset.map(lambda x: x * 2)
for item in dataset:
    print(item)

dataset = dataset.unbatch()
for item in dataset:
    print(item)

dataset = dataset.filter(lambda x: x < 10)  # keep only items < 10
for item in dataset:
    print(item)

for item in dataset.take(3):
    print(item)


# 데이터 셔플링
tf.random.set_seed(42)

dataset = tf.data.Dataset.range(10).repeat(3)
dataset = dataset.shuffle(buffer_size=5, seed=42).batch(7)
for item in dataset:
    print(item)