from orgKAARMA import KAARMA

model = KAARMA(10, 5, 1, 1, 1, 0.1)

y = [0]
x = [[[0.1], [0.1, 0.2], [0.1, 0.2, 0.3]]]

model.test(x, y)

