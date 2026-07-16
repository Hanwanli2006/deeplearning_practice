import matplotlib.pyplot as plt
logs="""
Epoch 1: Train Acc 9.95% | Val Acc 9.40%
Epoch 2: Train Acc 11.42% | Val Acc 11.11%
Epoch 3: Train Acc 14.35% | Val Acc 21.08%
Epoch 4: Train Acc 19.90% | Val Acc 26.50%
Epoch 5: Train Acc 29.46% | Val Acc 34.76%
Epoch 6: Train Acc 35.29% | Val Acc 41.88%
Epoch 7: Train Acc 42.63% | Val Acc 46.15%
Epoch 8: Train Acc 50.21% | Val Acc 49.29%
Epoch 9: Train Acc 56.85% | Val Acc 64.39%
Epoch 10: Train Acc 64.33% | Val Acc 66.38%
Epoch 11: Train Acc 70.25% | Val Acc 72.93%
Epoch 12: Train Acc 75.60% | Val Acc 73.22%
Epoch 13: Train Acc 76.17% | Val Acc 78.63%
Epoch 14: Train Acc 82.99% | Val Acc 76.07%
Epoch 15: Train Acc 85.22% | Val Acc 81.20%
Epoch 16: Train Acc 86.74% | Val Acc 81.48%
Epoch 17: Train Acc 88.39% | Val Acc 81.77%
Epoch 18: Train Acc 89.44% | Val Acc 83.48%
Epoch 19: Train Acc 91.00% | Val Acc 86.89%
Epoch 20: Train Acc 93.18% | Val Acc 77.49%
Epoch 21: Train Acc 89.82% | Val Acc 82.91%
Epoch 22: Train Acc 92.85% | Val Acc 84.90%
Epoch 23: Train Acc 92.23% | Val Acc 79.77%
Epoch 24: Train Acc 90.57% | Val Acc 85.19%
Epoch 25: Train Acc 94.03% | Val Acc 81.77%
Epoch 26: Train Acc 91.47% | Val Acc 79.20%
Epoch 27: Train Acc 93.04% | Val Acc 84.05%
Epoch 28: Train Acc 94.13% | Val Acc 88.89%
Epoch 29: Train Acc 92.23% | Val Acc 78.92%
Epoch 30: Train Acc 94.27% | Val Acc 82.91%
Epoch 31: Train Acc 93.37% | Val Acc 88.03%
Epoch 32: Train Acc 95.22% | Val Acc 86.61%
Epoch 33: Train Acc 94.74% | Val Acc 84.62%
Epoch 34: Train Acc 95.59% | Val Acc 85.75%
Epoch 35: Train Acc 94.22% | Val Acc 77.21%
Epoch 36: Train Acc 89.58% | Val Acc 78.35%
Epoch 37: Train Acc 93.70% | Val Acc 85.75%
Epoch 38: Train Acc 95.93% | Val Acc 86.89%
Epoch 39: Train Acc 93.04% | Val Acc 82.34%
Epoch 40: Train Acc 95.22% | Val Acc 80.63%
Epoch 41: Train Acc 94.46% | Val Acc 87.46%
Epoch 42: Train Acc 92.23% | Val Acc 76.64%
Epoch 43: Train Acc 93.23% | Val Acc 84.33%
Epoch 44: Train Acc 95.12% | Val Acc 83.76%
Epoch 45: Train Acc 95.74% | Val Acc 84.05%
Epoch 46: Train Acc 94.08% | Val Acc 84.33%
Epoch 47: Train Acc 95.03% | Val Acc 84.33%
Epoch 48: Train Acc 94.84% | Val Acc 86.04%
Epoch 49: Train Acc 94.17% | Val Acc 81.77%
Epoch 50: Train Acc 95.41% | Val Acc 85.19%
"""
train_accs = []
val_accs = []
for line in logs.strip().split("\n"):
    parts = line.split("|")
    train_accs.append(float(parts[0].split()[-1].replace("%","")))
    val_accs.append(float(parts[1].split()[-1].replace("%","")))

plt.plot(train_accs, label="Train")
plt.plot(val_accs, label="Val")
plt.legend()
plt.grid(True)
plt.show()
