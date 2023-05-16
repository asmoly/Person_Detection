def rectangle_overlap(ax, ay, ax1, ay1, bx, by, bx1, by1):
    dx = min(ax1, bx1) - max(ax, bx)
    dy = min(ay1, by1) - max(ay, by)

    print(dx, dy)
    return dx*dy

print(rectangle_overlap(1, 1, 2, 2, 15, 15, 40, 40))