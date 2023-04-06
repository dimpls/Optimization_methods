import math
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


def partial(func, x, index, eps) -> float:
    """
    Вычисляет приближенное значение частной производной функции `func` по переменной с индексом `index`
    в точке `x`, используя метод конечных разностей.
    :param func: функция, для которой вычисляется частная производная
    :param x: список значений переменных функции `func` в точке, где необходимо вычислить частную производную
    :param index: индекс переменной в списке `x`, по которой необходимо вычислить частную производную
    :param eps: шаг для вычисления конечных разностей
    :return: приближенное значение частной производной функции `func` по переменной с индексом `index` в точке `x`
    """
    x[index] += eps
    f_r = func(x)
    x[index] -= 2 * eps
    f_l = func(x)
    x[index] += eps
    return (f_r - f_l) / eps * .5


def gradient(func, x, eps):
    """
    Данный код реализует функцию, которая вычисляет градиент
    :param func: функция, чей градиент необходимо вычислить.
    :param x: точка, в которой необходимо вычислить градиент.
    :param eps: шаг для метода конечных разностей.
    :return: вектор частных производных функции func в точке x.
    """
    # Создаем массив нулей для хранения компонент градиента.
    g = np.zeros_like(x)
    # Проходимся по всем компонентам градиента и вычисляем их с помощью функции partial.
    for i in range(x.size):
        g[i] = partial(func, x, i, eps)
    # Возвращаем вектор градиента.
    return g


def gradient_desect(func, x0, eps, max_iters):
    """
    Данная функция реализует метод градиентного спуска с методом бисекции для многомерной функции.
    :param func: целевая функция, для которой необходимо найти минимум.
    :param x0: начальное значение аргументов функции.
    :param eps: заданная точность, при достижении которой алгоритм остановится.
    :param max_iters: максимальное число итераций алгоритма.
    :return: Возвращает найденное значение аргумента функции приближенно, которое минимизирует функцию.
    """
    x_1 = None
    for i in range(max_iters):
        """
        Эта строка вычисляет новое приближение для значения аргумента функции x путем вычитания из текущего значения x0 
        произведения градиента функции gradient(func, x0, eps) на некоторое число 1.0.
        Коэффициент 1.0 здесь служит для управления скоростью сходимости алгоритма, но в данном случае он не играет 
        особой роли, так как не изменяет направление движения.
        """
        x_1 = x0 - 1.0 * gradient(func, x0, eps)
        x_1 = bisect_multidimensional(func, x0, x_1, eps, max_iters)
        if np.linalg.norm(x_1 - x0) < eps:
            break
        x_1, x0 = x0, x_1

    return (x_1 + x0) * .5


def magnitude_n(array: np.ndarray):
    return math.sqrt(sum((x * x for x in array.flat)))


def conj_gradient_desc(function, x0, eps, max_iters):
    """
    Метод сопряженных градиентов для многомерной оптимизации функции.
    :param function: Функция, которую нужно оптимизировать.
    :param x0: Начальное приближение для оптимальной точки.
    :param eps: Заданная точность оптимизации.
    :param max_iters: Максимальное количество итераций.
    :return:
    """

    # Инициализация начальных значений для prev и temp
    prev, temp = x0, x0
    # Расчет антиградиента в начальной точке
    anti = (-1)*gradient(function, x0, eps)
    # Цикл до достижения заданной точности или максимального количества итераций
    for i in range(max_iters):
        # Расчет временной точки temp
        temp = prev + anti
        # Оптимизация функции в интервале между prev и temp методом бисекции
        temp = bisect_multidimensional(function, prev, temp, eps, 1000)

        # Если достигнута заданная точность, выход из цикла
        if magnitude_n(anti) < eps or magnitude_n(temp-prev) < eps:
            break

        # Расчет градиента в точке temp
        temp_grad = gradient(function, temp, eps)
        # Расчет коэффициента w
        w = math.pow(magnitude_n(temp_grad), 2) / math.pow(magnitude_n(anti), 2)
        # Обновление значения антиградиента
        anti = anti*w - temp_grad
        # Обновление значения prev
        prev = temp

    # Возвращение оптимальной точки
    return temp

#r1 = np.array([i + 10 for i in range(32)])
#r2 = np.array([1 for i in range(32)])

#print(bisect_multidimensional(func_nd, r1, r2, 1e-6, 1000))

#x0 = np.array([-10, 8])
x0 = np.array([i + 10 for i in range(32)])
print(gradient_desect(func_nd, x0, 1e-6, 100))
print(conj_gradient_desc(func_nd, x0, 1e-6, 100))
#print(per_cor_descend())

