import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
x = [0.5, 1.0, 0.2]
W1 = [
    [0.4, -0.2, 0.1], 
    [-0.3, 0.8, 0.5], 
    [0.7, -0.5, 0.9]
]
b1 = [0.1, -0.1, 0.2]
W2 = [0.6, -0.4, 0.3]
b2 = 0.2
z1 = []
a1 = []
for i in range(3):
    z = (W1[i][0] * x[0]) + (W1[i][1] * x[1]) + (W1[i][2] * x[2]) + b1[i]
    z1.append(z)
    a1.append(sigmoid(z))
z2 = (W2[0] * a1[0]) + (W2[1] * a1[1]) + (W2[2] * a1[2]) + b2
a2 = sigmoid(z2)
print("\nINPUTS (3 nodes):")
print(x)

print("\nHIDDEN LAYER")
print("Weights (W1):")
for row in W1:
    print(row)
print("Biases (b1):", b1)

print("\nHidden layer z-values (pre-activation):")
print(z1)

print("\nHidden layer activations (a1):")
print(a1)

print("\nOUTPUT LAYER")
print("Weights (W2):", W2)
print("Bias (b2):", b2)

print("\nz2 (output pre-activation):", z2)
print("a2 (output activation):", a2)
if a2>0.50:
    print("Spam Mail")
else:
    print("Not Spam Mail")
