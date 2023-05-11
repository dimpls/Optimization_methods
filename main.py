import math
from typing import Callable
from numpy.linalg import norm
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

PHI = (1 + 5 ** 0.5) / 2
eps = 1e-6
max_it = 100
alpha = 1.0

def magnitude_n(array: np.ndarray):
    return math.sqrt(sum((float(x) ** 2 for x in array.flat)))


def function_lab1(x):
    return x[0] ** 2 + 2 * x[1] ** 2


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
    #print('!!!', d)
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


def partial(func, x, index, eps) -> float:
    x[index] += eps
    f_r = func(x)
    x[index] -= 2.0 * eps
    f_l = func(x)
    x[index] += eps
    return (f_r - f_l) * 0.5 / eps


def gradient(func, x, eps):
    g = np.zeros_like(x)
    for i in range(x.size):
        g[i] = partial(func, x, i, eps)
    return g


def gradient_desect(func, x0, eps, max_iters):
    x_0, x_1 = x0, x0
    for i in range(max_iters):
        x_1 = x_0 - 1.0 * gradient(func, x_0, eps)
        x_1 = bisect_multidimensional(func, x_0, x_1, eps, max_iters)
        if np.linalg.norm(x_1 - x_0) < eps:
            break
        x_0 = x_1

    return (x_1 + x_0) * .5


def conj_gradient_desc(function, x0, eps, max_iters):
    prev, temp = x0, x0
    anti = (-1) * gradient(function, x0, eps)
    for i in range(max_iters):
        temp = prev + anti
        temp = bisect_multidimensional(function, prev, temp, eps, 1000)

        if np.linalg.norm(anti) < eps or np.linalg.norm(temp-prev) < eps:
            break

        temp_grad = gradient(function, temp, eps)
        w = math.pow(np.linalg.norm(temp_grad), 2) / math.pow(np.linalg.norm(anti), 2)
        anti = anti * w - temp_grad
        prev = temp

    return temp


def partial2(func, x, index_1, index_2, eps):
    x[index_2] -= eps
    f_l = partial(func, x, index_1, eps)
    x[index_2] += 2.0 * eps
    f_r = partial(func, x, index_1, eps)
    x[index_2] -= eps
    return (f_r - f_l) / eps * 0.5


def hessian(f, x, eps):
    ddf = np.zeros((x.size, x.size,), dtype=np.float32)
    for row in range(x.size):
        for col in range(row + 1):
            ddf[row, col] = partial2(f, x, row, col, eps)
            ddf[col, row] = ddf[row, col]
    return ddf




def newtonRaphson(func, xStart, eps, maxIters):
    xi = xStart.copy()
    xi1 = xStart.copy()
    counter = 0
    while counter != maxIters:
        hessian2 = hessian(func, xi, eps)
        inv_hessian = np.linalg.inv(hessian2)
        grad = gradient(func, xi, eps)
        xi1 = xi - inv_hessian @ grad

        counter += 1
        if np.linalg.norm(xi1 - xi) < eps:
            break
        xi = xi1

    return (xi + xi1) * 0.5


def internal_penalty(x, n, b) -> float:
    return sum(xi for xi in np.power((n @ x - b) * alpha, -2.0).flat)


def external_penalty(x, n, b) -> float:
    dist = n @ x - b
    # return sum(xi for xi in (np.heaviside(dist, 1.0)).flat)
    return sum(xi for xi in (np.heaviside(dist, 1.0) * np.power(32 * dist, 2.0)).flat)


def _eval_penalty(penalty: Callable[[np.ndarray], float], xs: np.ndarray = None, ys: np.ndarray = None):
    if xs is None:
        xs = np.linspace(0.0, 10.0, 256, dtype=np.float32)
    if ys is None:
        ys = np.linspace(0.0, 10.0, 256, dtype=np.float32)
    return np.array([[penalty(np.array([xi, yi]))for yi in ys.flat]for xi in xs.flat])


def eval_ext_penalty(x: np.ndarray = None, y: np.ndarray = None):
    return _eval_penalty(lambda x: external_penalty(x, N, B), x, y)


def eval_int_penalty(x: np.ndarray = None, y: np.ndarray = None):
    return _eval_penalty(lambda x: min(internal_penalty(x, N, B), 100.0), x, y)


def draw_line(n, b, bounds: Tuple[float, float, float, float] = (0.0, 0.0, 10.0, 10.0)) -> None:
    b =     b / n[1]
    k = -n[0] / n[1]
    x_0, x_1 = bounds[0], bounds[2]
    y_0, y_1 = bounds[1], bounds[3]

    y_1 = (y_1 - b) / k
    y_0 = (y_0 - b) / k

    if y_0 < y_1:
        x_1 = min(x_1, y_1)
        x_0 = max(x_0, y_0)
    else:
        x_1 = min(x_1, y_0)
        x_0 = max(x_0, y_1)

    x = [x_0, x_1]
    y = [b + x_0 * k, b + x_1 * k]
    plt.plot(x, y, ':k')

    x_c = sum(x) * 0.5
    y_c = sum(y) * 0.5

    xn = [x_c, x_c + n[0] * 2.5]
    yn = [y_c, y_c + n[1] * 2.5]
    plt.plot(xn, yn, 'r')


def draw_lines(bounds: Tuple[float, float, float, float] = (0.0, 0.0, 10.0, 10.0)):
    for n, b in zip(N, B):
        draw_line(n, b, bounds)




#x0 = np.array([i + 10 for i in range(64)])
#print(gradient_desect(func_nd, x0, 1e-6, 100))
#print(conj_gradient_desc(func_nd, x0, 1e-6, 100))

#x0 = np.array([i for i in range(64)], dtype=float)
#x = np.array([1, 1])
#print(hessian(func_nd, x0, 1e-2))
#print(newtonRaphson(func_nd, x0, 1e-6, 1000))

def func(x):
    return (x[0] - 4) ** 2 + (x[1] - 4) ** 2

d = np.array([-10, 5, 8, 32], dtype = np.float64)
x_start = np.array([-3, -4], dtype = np.float64)
n = np.array([[3,1],[-3,4],[4,-4], [-4, -1]], dtype = np.float64)

print("Метод Ньютона-Рафсона", newtonRaphson(func, x_start, eps, max_it))
print("Метод Ньютона-Рафсона с внешними штрафами", newtonRaphson(lambda x: func(x) + external_penalty(x_start, n , d), x_start, eps, max_it))
print("Метод Ньютона-Рафсона с внутренними штрафами", newtonRaphson(lambda x: func(x) + internal_penalty(x_start, n, d), x_start, eps, max_it))
# a = np.array([i for i in range(0, 128)], dtype = np.float64)
# print("Метод Ньютона-Рафсона", newtone_raphson(func1, a, eps, max_it))

N = np.array([[3,1],[-3,4],[4,-4], [-4, -1]], dtype=np.float32)
n = np.linalg.norm(N, axis=1)
alpha = 1.0
for i in range(N.shape[0]):
    N[i, :] /= n[i]

B = np.array([-10, 5, 8, 32], dtype=np.float32) / n

x = np.linspace(-10.0, 5.0, 256, dtype=np.float32)
y = np.linspace(-10.0, 5.0, 256, dtype=np.float32)
bounds = (np.amin(x), np.amin(y), np.amax(x), np.amax(y))
z = eval_ext_penalty(x, y)
plt.imshow(np.flipud(z), extent=[np.amin(x), np.amax(x), np.amin(y), np.amax(y)])
plt.title("external penalty")
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.grid(True)
draw_lines(bounds)
plt.show()

z = eval_int_penalty(x, y)
plt.imshow(np.flipud(z), extent=[np.amin(x), np.amax(x), np.amin(y), np.amax(y)])
plt.title("internal penalty")
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.grid(True)
draw_lines(bounds)
plt.show()


