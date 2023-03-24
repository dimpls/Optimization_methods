from typing import Callable
from numpy.linalg import norm
import numpy as np

PHI = (1 + 5 ** 0.5) / 2


def function_lab1(x):
    return x ** 3 - 2 * x - 5


def function_lab2(x):
    return (x[0]) ** 2 + (x[1]) ** 2


def func_nd(x: np.ndarray):
    return sum((xi - i) ** 2 for i, xi in enumerate(x.flat))


def bi_sect(func, a, b, eps, max_iters) -> float:
    """
    Метод Дихотомии
    """
    c: float = .0
    if a > b:
        a, b = b, a

    cnt: int = 0

    for cnt in range(max_iters):

        c = (a + b) * 0.5

        if func(c + eps) > func(c - eps):
            b = c
        else:
            a = c

        if abs(b - a) < eps:
            break

    print('Кол-во итераций:', cnt)

    return (a + b) * .5


def golden_ratio(func, a, b, eps, max_iters):
    """
    Метод Золотого сечения
    """
    c: float = .0
    if a > b:
        a, b = b, a

    cnt: int = 0

    for cnt in range(max_iters):

        x1 = b - (b - a) / PHI
        x2 = a + (b - a) / PHI

        if func(x1) >= func(x2):
            a = x1
        else:
            b = x2

        if abs(a - b) < eps:
            break

    print('Кол-во итераций:', cnt)

    return (a + b) * .5


def fibonacci(n):
    if n < 1:
        return 0, 0
    if n < 2:
        return 0, 1
    prev_num = 0
    num = 1
    while n > num:
        prev_num, num = num, num + prev_num
    return prev_num, num


def fib(func, a, b, eps, max_iters):
    """
       Метод Фибоначчи
    """


    if a > b:
        a, b = b, a

    f_n, f_n_1 = fibonacci((b - a) / eps)

    while f_n_1 != f_n:
        d = b - a
        if abs(d) < eps:
            break
        f_tmp = f_n_1 - f_n # Число Фибоначчи f(n - 1)

        x1 = a + d * f_tmp / f_n_1
        x2 = a + d * f_n / f_n_1

        f_n_1 = f_n # сдвигаем влево
        f_n = f_tmp
        if func(x1) > func(x2):
            a = x1
        else:
            b = x2

    return (a + b) * .5


def bisect_multidimensional(func, r1: np.ndarray, r2: np.ndarray, eps: float, max_iters: int) -> np.ndarray:

    """
    Метод Бисекции
    :param func: целевая функция.
    :param r1: начальное значение вектора координат.
    :param r2: конечное значение вектора координат.
    :param eps: допустимая погрешность.
    :param max_iters: максимальное число итераций.
    :return:  возвращаем найденное оптимальное значение вектора координат.
    """

    d = r2 - r1
    d_eps = eps / np.sqrt(d * d).sum()

    tl = 0.0
    tr = 1.0

    for i in range(max_iters):
        if tr - tl < d_eps:
            break

        tc = (tr + tl) * 0.5

        if func(r1 + d * (tc - d_eps)) > func(r1 + d * (tc + d_eps)):
            tl = tc
        else:
            tr = tc
    return r1 + d * ((tl + tr) * 0.5)


def per_cor_descend(f, x0, eps: float = 1e-3, max_iters: int = 1000):

    """
    Метод покоординатного спуска
    :param f: целевая функция.
    :param x0: начальное значение.
    :param eps: точность оптимизации.
    :param max_iters: максимальное число итераций
    :return: возвращаем найденное оптимальное значение вектора координат.
    """

    opt_cords_n, lam, x1 = 0, 0, 0

    for i in range(max_iters):

        cord_id = i % x0.size

        x0[cord_id] -= eps
        fl = f(x0)
        x0[cord_id] += 2 * eps
        fr = f(x0)
        x0[cord_id] -= eps

        x_curr = x0[cord_id]

        lam = 1.0 if fl > fr else -1.0

        x1 = np.copy(x0)
        x1[cord_id] += lam
        x0 = bisect_multidimensional(f, x0, x1, eps, max_iters)

        if abs(x0[cord_id] - x_curr) < eps:
            opt_cords_n += 1

            if opt_cords_n == x0.size:
                break

        else:
            opt_cords_n = 0

    return x0


r1 = np.array([i + 10 for i in range(32)])
r2 = np.array([1 for i in range(32)])

#print(bisect_multidimensional(func_nd, r1, r2, 1e-6, 1000))

#x0 = np.array([-10, 8])
x0 = np.array([i + 10 for i in range(32)])
print(per_cor_descend(func_nd, x0, 1e-6, 1000))
