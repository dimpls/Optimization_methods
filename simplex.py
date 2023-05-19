from collections import namedtuple
from typing import List, Tuple
from fractions import Fraction
import json

import numpy as np
from numpy import ndarray

SOLVE_MAX = 0
SOLVE_MIN = 1
LESS_EQUAL = -1
EQUAL = 0
MORE_EQUAL = 1
CELL_WIDTH = 11
INEQUALITIES_STR = ['<= ', ' = ', '>= ']


def _as_frac(value: float):
    f = Fraction(value).limit_denominator(50)
    i, d, n = f.numerator // f.denominator, f.denominator, f.numerator
    if n == 0:
        return f"{n}"
    if i == 0:
        return f"{n}/{d}"
    if i == n * d:
        return f"{i}"
    return f"{i} {n - d * i}/{d}"


class SimplexStep(namedtuple('SimplexStep', 'main_row, main_col, basis_args, simplex_table')):

    def __new__(cls, main_row: int, main_col: int, basis_args: np.ndarray, simplex_table: np.ndarray):
        return super().__new__(cls, main_row, main_col, basis_args, simplex_table)

    @property
    def main_element(self):
        if self.main_row == -1 or self.main_col == -1:
            return None
        return self.simplex_table[self.main_row, self.main_col]

    def __str__(self):
        """
        Создаёт строковое представление симплекс таблицы.
        :return: строковое представление симплекс таблицы, лол.
        """
        def _xi(i):
            return f"x{i}"

        rows, cols = self.simplex_table.shape
        main_element = self.main_element
        table = [f"a[{self.main_row + 1:^3},{self.main_col + 1:^3}] = ",
                 f"{_as_frac(main_element) if main_element is not None else 'None':<7}\n",
                 f" {'_' * (CELL_WIDTH * (cols + 1) + cols)}\n", f"|{'_' * CELL_WIDTH}|",
                 '|'.join(f'{_xi(i + 1):_^{CELL_WIDTH}}' for i in range(cols - 1)) + f"|{'b':_^{CELL_WIDTH}}|\n"]

        for arg_id, row in zip(self.basis_args.flat, self.simplex_table):
            table.append(f"|{_xi(arg_id + 1):^{CELL_WIDTH}}|" + '|'.join(f'{_as_frac(row_i):^{CELL_WIDTH}}' for row_i in row))
            table.append(f"|\n")

        sim_d = self.simplex_table[-1]
        table.append(f"|{'dC':_^{CELL_WIDTH}}|" + '|'.join(f'{_as_frac(v):_^{CELL_WIDTH}}' for v in sim_d) + f"|\n")
        return ''.join(v for v in table)


class Simplex:

    def __init__(self, weights_v: np.ndarray, bounds_m: np.ndarray, bounds_v: np.ndarray, ineq=None, ref_or_copy=False):
        flag, message = Simplex._validate(weights_v, bounds_m, bounds_v)
        if not flag:
            raise RuntimeError(message)
        self._ineq = np.array([LESS_EQUAL for _ in range(bounds_m.shape[0])])if ineq is None else ineq
        self._bounds_m  =  bounds_m if ref_or_copy else np.copy(bounds_m)
        self._bounds_v  =  bounds_v if ref_or_copy else np.copy(bounds_v)
        self._weights_v = weights_v if ref_or_copy else np.copy(weights_v)
        self._simplex_t = np.copy(self._bounds_m)
        # [(переменные в базисе, с-м таблица)]
        self._simplex_t_history: List[SimplexStep] = []
        self._basis_args = None  # список индексов базисных аргументов
        self._f_mod_args = None  # список индексов аргументов, которые модифицируют целевую функцию
        self._build_sm_table()

    def __str__(self):
        def _convert(index, value) -> str:
            if index == 0:
                return f"-{_as_frac(abs(float(value))):^5} * x{index + 1}" if value < 0 else\
                    f" {_as_frac(abs(float(value))):^5} * x{index + 1}"
            return f"-{_as_frac(abs(float(value))):^5} * x{index + 1}" if value < 0 else \
                f"+{_as_frac(abs(float(value))):^5} * x{index + 1}"

        problem = ["F(x, c) =", " ".join(f"{_convert(i, v)}" for i, v in enumerate(self._weights_v.flat)), "\n"]

        for (row, b, ineq) in zip(self._bounds_m, self._bounds_v, self._ineq):
            s_row = "".join(v for v in ["         ",
                                        " ".join(f"{_convert(i, v)}" for (i, v) in enumerate(row.flat)),
                                        " ",
                                        INEQUALITIES_STR[ineq + 1],
                                        f"{_as_frac(float(b))}"])
            problem.append(s_row)
            problem.append("\n")

        return "".join(v for v in problem)

    @staticmethod
    def _validate(weights: np.ndarray, bounds_mat: np.ndarray, bounds: np.ndarray) -> Tuple[bool, str]:
        _n_weights = weights.size
        _n_bounds  = bounds.size
        _m_rows, _m_cols = bounds_mat.shape
        if _n_weights != _m_cols:
            return False, "Количество столбцов матрицы ограничений не совпадает с количеством весов..."
        if _n_bounds != _m_rows:
            return False, "Количество строк матрицы ограничений не совпадает с количеством ограничений..."
        return True, ""

    @property
    def n_bounds(self) -> int:
        return self._bounds_v.size

    @property
    def n_weights(self) -> int:
        return self._weights_v.size

    def _create_col(self, ineq_row: int, ineq: int):
        """
        Создаёт новую колонку в см таблице в соответствии с типом ограничения.
        :param ineq_row: строка с ограничением.
        :param ineq: тип неравенство.
        :return:
        """
        # last_table = self._s_tables[-1]
        if ineq == LESS_EQUAL:
            col = [[1.0] if i == ineq_row else [0.0] for i in range(self.n_bounds)]
            self._simplex_t = np.hstack((self._simplex_t, np.array(col)))
            return self._simplex_t.shape[1] - 1, - 1

        if ineq == EQUAL:
            col = [[1.0] if i == ineq_row else [0.0] for i in range(self.n_bounds)]
            self._simplex_t = np.hstack((self._simplex_t, np.array(col)))
            return self._simplex_t.shape[1] - 1, self._simplex_t.shape[1] - 1

        if ineq == MORE_EQUAL:
            col_1 = [[-1.0] if i == ineq_row else [0.0] for i in range(self.n_bounds)]
            col_2 = [[1.0] if i == ineq_row else [0.0] for i in range(self.n_bounds)]
            self._simplex_t = np.hstack((self._simplex_t, np.array(col_1)))
            self._simplex_t = np.hstack((self._simplex_t, np.array(col_2)))
            return self._simplex_t.shape[1] - 2, self._simplex_t.shape[1] - 1

        raise RuntimeError("incorrect inequality parameter!")

    def _build_sm_table(self):
        """
        Составляет симплекс таблицу на основе прочитанных данных.
        :return:
        """
        for row_id, b_val in enumerate(self._bounds_v):
            if self._bounds_v[row_id] > 0:
                continue
            self._bounds_m[row_id, :] *= -1.0
            self._ineq [row_id] *= -1

        # TODO Добавить проверку на отрицательность элементов вектора B
        self._basis_args = np.linspace(0, self.n_bounds - 1, self.n_bounds, dtype=int)
        for ineq_id, ineq in enumerate(self._ineq.flat):
            basis_arg_id, basis_arg_id_add = self._create_col(ineq_id, ineq)
            if basis_arg_id != -1:
                self._basis_args[ineq_id] = basis_arg_id

        # текущий размер симплекс таблицы
        rows, cols = self._simplex_t.shape

        # симплекс разности (вектор  -С)
        self._simplex_t = np.vstack((self._simplex_t, np.zeros((1, cols), dtype=float)))

        for index, value in enumerate(self._weights_v.flat):
            self._simplex_t[rows, index] = -value

        # симплекс ограничения (вектор B)
        self._simplex_t = np.hstack((self._simplex_t, np.zeros((rows + 1, 1), dtype=float)))

        for index, value in enumerate(self._bounds_v.flat):
            self._simplex_t[index, cols] = value

    def _get_main_col(self) -> int:
        col_id = np.argmin(self._simplex_t[-1, :-1])
        if self._simplex_t[-1, col_id] >= 0:
            return -1
        return int(col_id)

    def _get_main_row(self, col_id: int) -> int:
        delta = 1e32
        main_row_id = -1
        for row_id, (ai, bi) in enumerate(zip(self._simplex_t[:-1, col_id].flat, self._simplex_t[:-1, -1].flat)):
            if ai <= 0.0:
                continue

            b_div_a = bi / ai
            if delta < b_div_a:
                continue

            delta = b_div_a
            main_row_id = row_id

        return main_row_id

    def _validate_solution(self) -> bool:
        args_fun = self._simplex_t[-1][0:len(self._bounds_m[0])]

        return any(item < 0 for item in args_fun)

    def _current_solution(self) -> np.ndarray:
        print(self._simplex_t)
        delta_row = self._simplex_t[-1, :-1]
        bounds_col = self._simplex_t[:-1, -1]

        #args_fun = self._simplex_t[3][0:2]
        while self._validate_solution():
            main_col = np.argmin(delta_row)
            if main_col == -1:
                break
            main_colon = self._simplex_t[:-1, main_col]

            delta = 1e32

            main_row = -1
            #print(main_colon, bounds_col)
            for row_id, (ai, bi) in enumerate(zip(main_colon.flat, bounds_col.flat)):
                if ai <= 0.0:
                    continue
                b_div_a = bi / ai
                print(b_div_a)
                if delta < b_div_a:
                    continue
                delta = b_div_a
                main_row = row_id
            #print(delta, main_row, "!!")
            #delta  вед эл

            if main_row == -1:
                break

            a_main = self._simplex_t[main_row, main_col]

            self._simplex_t[main_row, :] /= a_main

            print("main-row", main_row, "main-col", main_col, 'delta', delta)
            for row in range(self._simplex_t.shape[0]):
                if row == main_row:
                    continue

                k = self._simplex_t[row, main_col]

                self._simplex_t[row, :] -= k * self._simplex_t[main_row, :]
            print(self._simplex_t)

            # if not self._validate_solution():
            #     lst = []
            #     for i in range(2):
            #         index = np.where(self._simplex_t[i][0:2] == 1.0)
            #         lst.append(f"x{index[0] + 1} = {self._simplex_t[i][-1]}")
            #     return lst
        return self._simplex_t


            #print(self._simplex_t)
            #if self._validate_solution:
            #    current_col = args_fun.min() #минимальный в строке
            #    index = (np.where(args_fun == current_col)) #индекс минимального элемента
            #    #print('col = ', index[0])

            #bi_allow = []
            #for i in range(len(self._simplex_t) - 1):
            #    bi_allow.append(self._simplex_t[i][-1])  # bi разрешающий

            #for i in range(len(self._simplex_t) - 1):
            #    bi_allow[i] = bi_allow[i] / self._simplex_t[i][int(index[0])]

            #positive_numbers = [num for num in bi_allow if num > 0]

            #index_min_bi_allow = bi_allow.index(min(positive_numbers))
            #print('row = ', index_min_bi_allow)


        #    for i in range(len(self._simplex_t)):
        #         lst = []
        #         for j in range(len(self._simplex_t[i])):
        #             if i != index_min_bi_allow:
        #                 koef = self._simplex_t[i][int(index[0])] / self._simplex_t[index_min_bi_allow][int(index[0])]
        #                 #print(self._simplex_t[index_min_bi_allow][j])
        #                 lst.append(self._simplex_t[i][j] - koef * self._simplex_t[index_min_bi_allow][j])
        #             else:
        #                 lst.append(self._simplex_t[i][j] / self._simplex_t[index_min_bi_allow][int(index[0])])
        #         # if i != index_min_bi_allow:
        #         self._simplex_t[i] = np.array(lst)
        # print(self._simplex_t)
        # args_fun = self._simplex_t[3][0:2]
        # if not self._validate_solution():
        #     lst = []
        #     for i in range(2):
        #         index = np.where(self._simplex_t[i][0:2] == 1.0)
        #         lst.append(f"x{index[0] + 1} = {self._simplex_t[i][-1]}")
        #     return lst



    def solve(self):
        return self._current_solution()


    def simplex_steps(self) -> str:
        """
        Создаёт строковое представление симплекс таблицы.
        :return: строковое представление симплекс таблицы, лол.
        """
        return "\n".join(str(v) for v in self._simplex_t_history)


def read_simplex(path_to_file: str) -> List[Simplex]:
    def _read_array(node):
        if 'shape' not in node:
            raise RuntimeError('shape is not defined')
        shape = tuple(map(int, node['shape']))
        if 'data' not in node:
            raise RuntimeError('data is not defined')
        array = np.array(list(map(float, node['data'])), dtype=np.float32)
        return array.reshape(shape)

    def _read_simplex(json_node) -> Simplex:
        solve_type = int(json_node['solve_type']) if 'solve_type' in json_node else SOLVE_MAX

        if 'weights' not in json_node:
            raise RuntimeError('weights is not defined')
        weights = _read_array(json_node['weights'])

        if 'bounds' not in json_node:
            raise RuntimeError('bounds is not defined')
        bounds = _read_array(json_node['bounds'])

        if 'bounds_matrix' not in json_node:
            raise RuntimeError('bounds_matrix is not defined')
        bounds_matrix = _read_array(json_node['bounds_matrix'])

        inequalities = np.array(list(map(int, json_node['inequalities'])), dtype=int) if 'inequalities' in json_node \
            else np.array([LESS_EQUAL for _ in range(bounds_matrix.shape[0])], dtype=int)
        return Simplex(weights, bounds_matrix, bounds, inequalities)

    with open(path_to_file, 'rt') as input_file:
        raw_data = json.loads(input_file.read())
        if "problems" in raw_data:
            return [_read_simplex(node) for node in raw_data['problems']]
        return [_read_simplex(raw_data)]


simplexes = read_simplex('sm_task.json')
print(simplexes[0])
nd = simplexes[0].solve()

for i in range(2):
    index = np.where(nd[i][0:2] == 1.0)
    print(f"x{index[0] + 1} = {nd[i][-1]}")
