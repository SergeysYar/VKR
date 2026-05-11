# Новый центр
X0 = 113.2
Y0 = 70.2

points = [
    (115.22, 69.11),
    (93.44, 40.60),
    (106.00, 38.31),
    (99.32, 46.78),
    (92.63, 64.37),
    (90.87, 75.62),
    (76.78, 68.58),
    (79.02, 60.88),
    (66.11, 57.33),
    (64.77, 80.82),
    (58.19, 74.68)
]

def convert_points(points, x0, y0):
    result = []
    for i, (x, y) in enumerate(points, start=1):
        x_new = x - x0
        y_new = y - y0
        result.append((i, x_new, y_new))
    return result

# Пересчёт
new_points = convert_points(points, X0, Y0)

# Вывод
for i, x, y in new_points:
    print(f"{i}: ({x:.2f}, {y:.2f})")