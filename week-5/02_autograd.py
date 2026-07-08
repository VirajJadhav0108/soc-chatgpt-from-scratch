import torch

torch.manual_seed(42)

x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

z = x**2 + y**2 + 3 * x * y

print("z =", z.item())

z.backward()

print("dz/dx =", x.grad.item())
print("dz/dy =", y.grad.item())

x.grad.zero_()
y.grad.zero_()

a = torch.randn(3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

c = a * b
d = c.sum()

print("\na =", a)
print("b =", b)
print("c =", c)
print("d =", d.item())

d.backward()

print("\nGradient of a:")
print(a.grad)

print("\nGradient of b:")
print(b.grad)

a.grad.zero_()
b.grad.zero_()

w = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

loss = (w ** 2).mean()

print("\nLoss =", loss.item())

loss.backward()

print("\nGradient of w:")
print(w.grad)

w.grad.zero_()

x = torch.linspace(-2, 2, 100).reshape(-1, 1)
y = 4 * x + 2

weight = torch.randn(1, requires_grad=True)
bias = torch.randn(1, requires_grad=True)

learning_rate = 0.05

for epoch in range(100):

    prediction = weight * x + bias

    loss = ((prediction - y) ** 2).mean()

    loss.backward()

    with torch.no_grad():
        weight -= learning_rate * weight.grad
        bias -= learning_rate * bias.grad

    weight.grad.zero_()
    bias.grad.zero_()

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch+1:03d} | Loss: {loss.item():.6f}"
        )

print("\nLearned Weight:", weight.item())
print("Learned Bias:", bias.item())

x = torch.tensor([5.0], requires_grad=True)

y = x ** 3 + 2 * x ** 2 - 5 * x + 1

gradient = torch.autograd.grad(y, x)[0]

print("\nFunction Value:", y.item())
print("Gradient:", gradient.item())

u = torch.randn(2, 2, requires_grad=True)
v = torch.randn(2, 2, requires_grad=True)

result = torch.mm(u, v)

loss = result.sum()

grads = torch.autograd.grad(loss, [u, v])

print("\nGradient of u:")
print(grads[0])

print("\nGradient of v:")
print(grads[1])
