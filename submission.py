import os
import sys
from math import sqrt, hypot

import numpy as np


STEINER_SOURCE = r"""
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>

using namespace std;

template <typename T> struct Vec2 {
    /*
    y 軸正は下方向
    x 軸正は右方向
    回転は時計回りが正（y 軸正を上と考えると反時計回りになる）
    */
    using value_type = T;
    T y, x;
    constexpr inline Vec2() = default;
    constexpr Vec2(const T& arg_y, const T& arg_x) : y(arg_y), x(arg_x) {}
    inline Vec2(const Vec2&) = default;            // コピー
    inline Vec2(Vec2&&) = default;                 // ムーブ
    inline Vec2& operator=(const Vec2&) = default; // 代入
    inline Vec2& operator=(Vec2&&) = default;      // ムーブ代入
    template <typename S>
    constexpr inline Vec2(const Vec2<S>& v) : y((T)v.y), x((T)v.x) {}
    inline Vec2 operator+(const Vec2& rhs) const {
        return Vec2(y + rhs.y, x + rhs.x);
    }
    inline Vec2 operator+(const T& rhs) const { return Vec2(y + rhs, x + rhs); }
    inline Vec2 operator-(const Vec2& rhs) const {
        return Vec2(y - rhs.y, x - rhs.x);
    }
    template <typename S> inline Vec2 operator*(const S& rhs) const {
        return Vec2(y * rhs, x * rhs);
    }
    inline Vec2 operator*(const Vec2& rhs) const { // x + yj とみなす
        return Vec2(x * rhs.y + y * rhs.x, x * rhs.x - y * rhs.y);
    }
    template <typename S> inline Vec2 operator/(const S& rhs) const {
        assert(rhs != 0.0);
        return Vec2(y / rhs, x / rhs);
    }
    inline Vec2 operator/(const Vec2& rhs) const { // x + yj とみなす
        return (*this) * rhs.inv();
    }
    inline Vec2& operator+=(const Vec2& rhs) {
        y += rhs.y;
        x += rhs.x;
        return *this;
    }
    inline Vec2& operator-=(const Vec2& rhs) {
        y -= rhs.y;
        x -= rhs.x;
        return *this;
    }
    template <typename S> inline Vec2& operator*=(const S& rhs) const {
        y *= rhs;
        x *= rhs;
        return *this;
    }
    inline Vec2& operator*=(const Vec2& rhs) { return *this = (*this) * rhs; }
    inline Vec2& operator/=(const Vec2& rhs) { return *this = (*this) / rhs; }
    inline bool operator!=(const Vec2& rhs) const {
        return x != rhs.x || y != rhs.y;
    }
    inline bool operator==(const Vec2& rhs) const {
        return x == rhs.x && y == rhs.y;
    }
    inline void rotate(const double& rad) { *this = rotated(rad); }
    inline Vec2<double> rotated(const double& rad) const {
        return (*this) * rotation(rad);
    }
    static inline Vec2<double> rotation(const double& rad) {
        return Vec2(sin(rad), cos(rad));
    }
    inline Vec2<double> rounded() const {
        return Vec2<double>(round(y), round(x));
    }
    inline Vec2<double> inv() const { // x + yj とみなす
        const double norm_sq = l2_norm_square();
        assert(norm_sq != 0.0);
        return Vec2(-y / norm_sq, x / norm_sq);
    }
    inline double l2_norm() const { return sqrt(x * x + y * y); }
    inline double l2_norm_square() const { return x * x + y * y; }
    inline T l1_norm() const { return std::abs(x) + std::abs(y); }
    inline double abs() const { return l2_norm(); }
    inline double phase() const { // [-PI, PI) のはず
        return atan2(y, x);
    }
};
template <typename T, typename S>
inline Vec2<T> operator*(const S& lhs, const Vec2<T>& rhs) {
    return rhs * lhs;
}
template <typename T> ostream& operator<<(ostream& os, const Vec2<T>& vec) {
    os << vec.y << ' ' << vec.x;
    return os;
}

struct SteinerTree {
    using T = float;
    static constexpr auto kMaxNTerminals = 11;
    static constexpr auto kMaxNNodes = 2000;
    struct Edge {
        int to;
        T cost;
    };
    struct Index {
        short terminal_set; // 無ければ -1
        unsigned short node;
    };

    // 入力
    vector<vector<Edge>> G; // グラフ
    vector<int> R;          // ターミナル集合

    array<array<array<Index, 2>, kMaxNNodes>, 1 << (kMaxNTerminals - 1)> b;
    vector<pair<int, int>> result_edges; // ここに結果が格納される

    void Solve() {
        static array<array<T, kMaxNNodes>, 1 << (kMaxNTerminals - 1)> l; // 距離
        const auto comp = [](const pair<T, Index>& a, const pair<T, Index>& b) {
            return a.first > b.first;
        };
        auto N = priority_queue<pair<T, Index>, vector<pair<T, Index>>,
                                decltype(comp)>(comp); // キュー
        static array<array<bool, kMaxNNodes>,
                     1 << (kMaxNTerminals - 1)>
            P; // 確定したかどうか
        for (auto&& li : l)
            fill(li.begin(), li.end(), numeric_limits<T>::infinity());
        for (auto i = 0; i < (int)R.size() - 1; i++) {
            auto r = R[i];
            l[1 << i][r] = 0;
            N.emplace((T)0, Index{(short)(1 << i), (unsigned short)r});
        }
        for (auto&& Pi : P)
            fill(Pi.begin(), Pi.end(), false);
        for (auto v = 0; v < (int)G.size(); v++) {
            l[0][v] = 0;
            P[0][v] = true;
        }
        for (auto&& bi : b)
            for (auto&& bij : bi)
                bij = {Index{-1, 0}, {-1, 0}};
        const auto full = (short)((1 << ((int)R.size() - 1)) - 1);
        while (N.size()) {
            const auto [distance_Iv, Iv] = N.top();
            const auto& [I, v] = Iv;
            N.pop();
            if (l[I][v] != distance_Iv)
                continue;
            if (I == full && v == R.back())
                break;
            P[I][v] = true;
            for (const auto& [w, c] : G[v]) {
                const auto new_distance_Iw = l[I][v] + c;
                if (new_distance_Iw < l[I][w] && !P[I][w]) { // 後半の条件いる？
                    l[I][w] = new_distance_Iw;
                    b[I][w] = {Index{I, v}, {-1, 0}};
                    N.emplace(new_distance_Iw, Index{I, (unsigned short)w});
                }
            }
            // ~I の空でない部分集合
            for (short J = full & ~I; J > 0; J = (J - 1) & ~I) {
                if (!P[J][v])
                    continue;
                const auto IJ = I | J;
                const auto new_distance_IJv = l[I][v] + l[J][v];
                if (new_distance_IJv < l[IJ][v] && !P[IJ][v]) {
                    l[IJ][v] = new_distance_IJv;
                    b[IJ][v] = {Index{I, v}, Index{J, v}};
                    N.emplace(new_distance_IJv, Index{(short)IJ, v});
                }
            }
        }
        // 復元
        result_edges.clear();
        BackTrack({full, (unsigned short)R.back()});
    }

    void BackTrack(const Index& Iv) {
        const auto& [I, v] = Iv;
        if (b[I][v][0].terminal_set == -1)
            return;
        if (b[I][v][1].terminal_set == -1) {
            assert(b[I][v][0].terminal_set == I);
            const auto w = b[I][v][0].node;
            result_edges.emplace_back((int)v, (int)w);
            BackTrack({I, w});
            return;
        }
        BackTrack(b[I][v][0]);
        BackTrack(b[I][v][1]);
    }
};

#define PARSE_ARGS(types, ...)                                                 \
    if (!PyArg_ParseTuple(args, types, __VA_ARGS__))                           \
    return NULL

#define ERROR()                                                                \
    {                                                                          \
        ofstream os("/dev/null");                                              \
        os << "Error at line " << __LINE__ << endl;                            \
        assert(false);                                                         \
    }

#define ERROR_IF_NULL(obj)                                                     \
    if ((obj) == NULL)                                                         \
    ERROR()

template <class T, class ItemParseFunc>
static vector<T> PyIterableToVector(PyObject* const iterable,
                                    const ItemParseFunc& parse) {
    PyObject* iter = PyObject_GetIter(iterable);
    ERROR_IF_NULL(iter);
    PyObject* item;
    auto res = vector<T>();
    while ((item = PyIter_Next(iter))) {
        res.push_back(parse(item));
        Py_DECREF(item);
    }
    Py_DECREF(iter);
    return res;
}

using Point = Vec2<int>;

static PyObject* SolveDijkstraAndSteiner(PyObject* /* self */, PyObject* args) {
    // 入力: 盤面の値、掘った衛星の場所、家と水源が何番目の衛星か

    PyObject *board_py, *satellites_py, *houses_py, *water_sources_py;

    PARSE_ARGS("OOOO", &board_py, &satellites_py, &houses_py,
               &water_sources_py);

    const auto parse_i = [](PyObject* const obj) {
        return (int)PyLong_AsLong(obj);
    };
    const auto parse_p = [](PyObject* const obj) {
        PyObject* const iter = PyObject_GetIter(obj);
        ERROR_IF_NULL(iter);
        auto item = PyIter_Next(iter);
        ERROR_IF_NULL(item);
        const auto first = (int)PyLong_AsLong(item);
        Py_DECREF(item);
        item = PyIter_Next(iter);
        ERROR_IF_NULL(item);
        const auto second = (int)PyLong_AsLong(item);
        Py_DECREF(item);
        Py_DECREF(iter);
        return Point(first, second);
    };
    const auto parse_f = [](PyObject* const obj) {
        return (float)PyFloat_AsDouble(obj);
    };
    const auto parse_vf = [&parse_f](PyObject* const obj) {
        return PyIterableToVector<float>(obj, parse_f);
    };
    auto board = PyIterableToVector<vector<float>>(board_py, parse_vf);
    const auto satellites = PyIterableToVector<Point>(satellites_py, parse_p);
    const auto houses = PyIterableToVector<int>(houses_py, parse_i);
    const auto water_sources =
        PyIterableToVector<int>(water_sources_py, parse_i);
    for (const auto& s : satellites)
        board[s.y][s.x] = 0;

    // 各衛星に対して
    auto target_ids = array<array<int, 200>, 200>();
    for (auto&& t : target_ids)
        fill(t.begin(), t.end(), -1);
    const auto n_satellites = (int)satellites.size();
    const auto n_targets = min(32, n_satellites);
    auto satellite_numbers = vector<int>(n_satellites);
    iota(satellite_numbers.begin(), satellite_numbers.end(), 0);

    auto distances = array<array<float, 200>, 200>();
    static auto from = array<array<array<signed char, 200>, 200>, 2000>();
    for (auto&& d : distances)
        fill(d.begin(), d.end(), numeric_limits<float>::infinity());
    static constexpr auto kDirections =
        array<Point, 4>{Point{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    auto G = vector<vector<SteinerTree::Edge>>(n_satellites);
    for (auto idx_satellites = 0; idx_satellites < n_satellites;
         idx_satellites++) {
        const auto& start = satellites[idx_satellites];

        // 距離が近い衛星を n_targets 個取り出す
        auto sq_distances = vector<int>(n_satellites);
        for (auto u = 0; u < n_satellites; u++)
            sq_distances[u] = (start - satellites[u]).l2_norm_square();
        partial_sort(
            satellite_numbers.begin(), satellite_numbers.begin() + n_targets,
            satellite_numbers.end(), [&sq_distances](const int l, const int r) {
                return sq_distances[l] < sq_distances[r];
            });
        for (auto i = 0; i < n_targets; i++) {
            const auto u = satellite_numbers[i];
            const auto [uy, ux] = satellites[u];
            target_ids[uy][ux] = u;
        }

        const auto max_l_distance =
            (int)sqrt(sq_distances[satellite_numbers[n_targets - 1]]) + 1;

        // ダイクストラ
        {
            distances[start.y][start.x] = 0.0;
            auto n_found_satellites = 1;
            if (target_ids[start.y][start.x] != idx_satellites) {
                ERROR();
            }
            target_ids[start.y][start.x] = -1;
            auto comp = [](const pair<float, Point>& l,
                           const pair<float, Point>& r) {
                return l.first > r.first;
            };
            auto q =
                priority_queue<pair<float, Point>, vector<pair<float, Point>>,
                               decltype(comp)>(comp);
            q.emplace(0.0, start);
            while (!q.empty()) {
                const auto [v_distance, v] = q.top();
                q.pop();
                if (distances[v.y][v.x] != v_distance)
                    continue;
                const auto v_idx = target_ids[v.y][v.x];
                if (v_idx != -1) {
                    G[idx_satellites].push_back({v_idx, v_distance});
                    target_ids[v.y][v.x] = -1;
                    n_found_satellites++;
                }

                for (auto i = 0; i < 4; i++) {
                    const auto d = kDirections[i];
                    const auto u = v + d;
                    if ((u - start).l2_norm_square() >
                            max_l_distance * max_l_distance ||
                        u.y < 0 || u.y >= 200 || u.x < 0 || u.x >= 200)
                        continue;
                    if (board[u.y][u.x] < 0.0) {
                        ERROR();
                    }
                    const auto u_distance = v_distance + board[u.y][u.x];
                    if (u_distance < distances[u.y][u.x]) {
                        distances[u.y][u.x] = u_distance;
                        from[idx_satellites][u.y][u.x] = (signed char)(i + 1);
                        q.emplace(u_distance, u);
                    }
                }
            }
            if (n_found_satellites != n_targets) {
                ERROR();
            }

            // 戻す
            for (auto y = max(0, start.y - max_l_distance);
                 y <= min(199, start.y + max_l_distance); y++)
                for (auto x = max(0, start.x - max_l_distance);
                     x <= min(199, start.x + max_l_distance); x++)
                    distances[y][x] = numeric_limits<float>::infinity();
        }

        // 戻す
        for (auto i = 0; i < n_targets; i++) {
            const auto u = satellite_numbers[i];
            const auto [uy, ux] = satellites[u];
            // target_ids[uy][ux] = -1;
            if (target_ids[uy][ux] != -1) {
                ERROR();
            }
        }
    }

    auto steiner = SteinerTree();
    auto& G2 = steiner.G;
    auto& R = steiner.R;

    // 縮約
    const auto G2_water_source_idx = n_satellites - (int)water_sources.size();
    G2.resize(G2_water_source_idx + 1);
    auto mapping_G_to_G2 = vector<int>(n_satellites);
    fill(mapping_G_to_G2.begin(), mapping_G_to_G2.end(), -1);
    {
        for (const auto i : water_sources)
            mapping_G_to_G2[i] = G2_water_source_idx;
        auto n = 0;
        for (auto&& i : mapping_G_to_G2)
            if (i != G2_water_source_idx)
                i = n++;
        if (n != G2_water_source_idx) {
            ERROR();
        }
    }
    auto mapping_G2_to_G = vector<vector<int>>(G2.size());
    for (auto i = 0; i < n_satellites; i++) {
        mapping_G2_to_G[mapping_G_to_G2[i]].push_back(i);
        for (const auto [u, c] : G[i])
            G2[mapping_G_to_G2[i]].push_back({mapping_G_to_G2[u], c});
    }

    for (const auto r : houses)
        R.push_back(mapping_G_to_G2[r]);
    R.push_back(G2_water_source_idx);

    steiner.Solve();

    // 復元
    auto result = vector<Point>();
    for (auto [v_G2, u_G2] : steiner.result_edges) {
        if (v_G2 > u_G2)
            swap(v_G2, u_G2);
        if (mapping_G2_to_G[v_G2].size() != 1) {
            ERROR();
        }
        auto v_G1 = mapping_G2_to_G[v_G2][0];
        auto v = satellites[v_G1];
        auto u_G1 = mapping_G2_to_G[u_G2][0]; // 仮
        if (u_G2 == G2_water_source_idx) {
            // u を 1 つに定める
            auto us_G1 = mapping_G2_to_G[u_G2];
            while (1) {
                auto it =
                    min_element(us_G1.begin(), us_G1.end(),
                                [&satellites, &v](const int l, const int r) {
                                    return (satellites[l] - v).l1_norm() <
                                           (satellites[r] - v).l1_norm();
                                });
                u_G1 = *it;

                auto u = satellites[u_G1];
                if (from[v_G1][u.y][u.x] != 0)
                    break;
                if (from[u_G1][v.y][v.x] != 0)
                    break;
                us_G1.erase(it);
            }
        }

        auto u = satellites[u_G1];
        if (v == u) {
            ERROR();
        }
        if (from[v_G1][u.y][u.x] == 0) {
            swap(v, u);
            swap(v_G1, u_G1);
        }
        if (from[v_G1][u.y][u.x] == 0) {
            ERROR();
        }
        while (u != v) {
            result.push_back(u);
            u -= kDirections[from[v_G1][u.y][u.x] - 1];
        }
    }

    // Python のリストにする
    PyObject* result_py = PyList_New(result.size());
    if (result_py == NULL)
        return NULL;
    for (auto i = 0; i < (int)result.size(); i++) {
        PyObject* const item = Py_BuildValue("[ii]", result[i].y, result[i].x);
        if (item == NULL)
            return NULL; // 本当は result_py を DECREF する必要が
        PyList_SET_ITEM(result_py, i, item);
    }
    return result_py;
}

static PyMethodDef steiner_methods[] = {
    {"solve_dijkstra_and_steiner", (PyCFunction)SolveDijkstraAndSteiner,
     METH_VARARGS, "solve_steiner"},
    {NULL},
};

static PyModuleDef steiner_module = {PyModuleDef_HEAD_INIT, "steiner", NULL, -1,
                                     steiner_methods};

PyMODINIT_FUNC PyInit_steiner(void) { return PyModule_Create(&steiner_module); }
"""

SETUP_SOURCE = r"""
from distutils.core import setup, Extension

module = Extension(
    "steiner",
    sources=["steiner_.cpp"],
    extra_compile_args=["-O3", "-march=native", "-std=c++17"],
)
setup(
    name="steiner",
    version="0.1.0",
    description="Steiner tree problem solver.",
    ext_modules=[module],
)
"""

if sys.argv[-1] == "ONLINE_JUDGE":  # or os.getcwd() != "/imojudge/sandbox":
    with open("steiner_.cpp", "w") as f:
        f.write(STEINER_SOURCE)
    with open("setup_.py", "w") as f:
        f.write(SETUP_SOURCE)
    os.system(f"{sys.executable} setup_.py build_ext --inplace > /dev/null")

from steiner import solve_dijkstra_and_steiner


class UnionFind:
    # 検証: https://atcoder.jp/contests/practice2/submissions/17594179
    def __init__(self, N):
        self.p = list(range(N))
        self.size = [1] * N

    def root(self, x):
        p = self.p
        while x != p[x]:
            x = p[x]
        return x

    def same(self, x, y):
        return self.root(x) == self.root(y)

    def unite(self, x, y):
        u = self.root(x)
        v = self.root(y)
        if u == v:
            return
        size = self.size
        if size[u] < size[v]:
            self.p[u] = v
            size[v] += size[u]
            size[u] = 0
        else:
            self.p[v] = u
            size[u] += size[v]
            size[v] = 0

    def count(self, x):
        return self.size[self.root(x)]


class GaussianProcess:
    # 修正コレスキー分解使えるかも
    def __init__(self, X, y, n_max_data, sq_sigma_noise, sq_sigma_rbf):
        """
        Args:
            X: [n_data, 2]
            y: [n_data]
            n_max_data (int):
            sq_sigma_noise (float):
            sq_sigma_rbf (float):
        """
        assert len(X) == len(y)
        n_data = len(X)
        assert n_data <= n_max_data
        assert X.shape[1] == 2
        self.gamma = 0.5 / sq_sigma_rbf
        self.K = np.empty((n_max_data, n_max_data), dtype=np.float32)
        K = self.kernel_func(X, X)
        K.flat[:: n_data + 1] += sq_sigma_noise
        self.K[:n_data, :n_data] = K
        self.K_inv = np.empty((n_max_data, n_max_data), dtype=np.float32)
        self.K_inv[:n_data, :n_data] = np.linalg.inv(self.K[:n_data, :n_data])
        self.sq_sigma_noise = sq_sigma_noise
        self.n_data = n_data
        self.X = np.empty((n_max_data, 2), dtype=np.float32)
        self.X[:n_data] = X
        self.y = np.empty(n_max_data, dtype=np.float32)
        self.y[:n_data] = y

    def add_data(self, x, y):
        """
        Args:
            x: [2]
            y (float):
        """
        self.X[self.n_data] = x
        self.y[self.n_data] = y
        k = self.kernel_func(x.reshape(1, 2), self.X[: self.n_data]).reshape(-1)
        self.K[self.n_data, : self.n_data] = k
        self.K[: self.n_data, self.n_data] = k
        self.K[self.n_data, self.n_data] = d = (
            self.kernel_func(x.reshape(1, 2)).reshape(tuple()) + self.sq_sigma_noise
        )
        v = self.K_inv[: self.n_data, : self.n_data] @ k
        t = 1.0 / (d - np.dot(k, v))
        tv = t * v
        self.K_inv[: self.n_data, : self.n_data] += tv[:, None] * v[None, :]
        self.K_inv[self.n_data, : self.n_data] = -tv
        self.K_inv[: self.n_data, self.n_data] = -tv
        self.K_inv[self.n_data, self.n_data] = t
        self.n_data += 1
        return self.n_data - 1

    def modify_data(self, idx, y):
        assert idx < self.n_data
        self.y[idx] = y

    def kernel_func(self, A, B=None, perf=False):
        """
        Args:
            A (np.ndarray): [n_data_a, 2]
            B: [n_data_b, 2]
        """
        if B is None:
            return np.ones(len(A))
        assert A.shape[1] == B.shape[1] == 2
        D = A[:, None, :].astype(np.float32) - B[None, :, :].astype(np.float32)
        np.square(D, out=D)
        K = D.sum(2)
        K *= -self.gamma
        np.exp(K, out=K)
        return K

    def predict(self, X, mu, sq_sigma, return_var=True):
        """
        Args:
            X: [n_data_pred, 2]
            mu (float):
            sq_sigma (float):
        """
        assert X.shape[1] == 2
        # [n_data_pred, n_data]  O(n_data_pred * n_data)
        k_star_T = self.kernel_func(X, self.X[: self.n_data], perf=not return_var)
        if not return_var:
            K_inv_y = self.K_inv[: self.n_data, : self.n_data] @ (
                self.y[: self.n_data] - mu
            )
            mean = k_star_T @ K_inv_y + mu
            return mean
        # [n_data_pred, n_data]  O(n_data_pred * n_data * n_data)
        k_star_T_K_inv = (
            k_star_T.astype(np.float64) @ self.K_inv[: self.n_data, : self.n_data]
        ).astype(
            np.float32
        )  # なぜか float64 の方が速い

        # [n_data_pred]
        mean = k_star_T_K_inv @ (self.y[: self.n_data] - mu) + mu
        # [n_data_pred]
        k_star_star = self.kernel_func(X)
        # [n_data_pred]
        var = (k_star_star - (k_star_T_K_inv * k_star_T).sum(1)) * sq_sigma
        return mean, var


N = 200

# 11.583815975676561
params = {
    "BASE_P_COEF": 3.296632677298496,
    "COEF_INITIAL_P_STD": 2.2668289592070243,
    "LEFT_TAIL_COEF": 1.3406903870719762,
    "MIN_STEINER_MU": 1529.2380472596885,
    "MU_END": 1774.213864087602,
    "MU_START": 1038.8354211939154,
    "NOISE_BASE_P_RATIO": 1.0710366976729258,
    "N_COLS": 14,
    "PRIORITY_COEF_STURDINESS_STD": 0.8676753588059648,
    "PRIORITY_COEF_SYSTEM_SIZE": 1.659149647205586,
    "PRIORITY_COEF_SYSTEM_SIZE_K": 4.964558435838279,
    "RECOVERY_P_COEF": 3.7691914814945653,
    "SIGMA": 989.3815053777757,
    "SIGMA_RBF": 12.972765642673519,
    "STOP_SIGMA": -0.11428275672454605,
    "TEMPORAL_PREDICTION_COEF": 0.32962113142577304,
}


def main():
    global input

    BASE_P_COEF = params["BASE_P_COEF"]
    RECOVERY_P_COEF = params["RECOVERY_P_COEF"]
    STOP_SIGMA = params["STOP_SIGMA"]
    TEMPORAL_PREDICTION_COEF = params["TEMPORAL_PREDICTION_COEF"]
    MU_START = params["MU_START"]
    MU_END = params["MU_END"]
    SIGMA = params["SIGMA"]
    NOISE_BASE_P_RATIO = params["NOISE_BASE_P_RATIO"]
    SIGMA_RBF = params["SIGMA_RBF"]
    N_COLS = params["N_COLS"]
    LEFT_TAIL_COEF = params["LEFT_TAIL_COEF"]
    PRIORITY_COEF_STURDINESS_STD = params["PRIORITY_COEF_STURDINESS_STD"]
    PRIORITY_COEF_SYSTEM_SIZE = params["PRIORITY_COEF_SYSTEM_SIZE"]
    PRIORITY_COEF_SYSTEM_SIZE_K = params["PRIORITY_COEF_SYSTEM_SIZE_K"]
    COEF_INITIAL_P_STD = params["COEF_INITIAL_P_STD"]
    MIN_STEINER_MU = params["MIN_STEINER_MU"]

    SIMULATION = False
    if SIMULATION:
        filename = 0  # "./tools/in/0238.txt"
        f = open(filename)
        input = f.readline

    _, W, K, C = map(int, input().split())
    if SIMULATION:
        true_bedrock = np.array(
            [list(map(int, input().split())) for _ in range(N)], dtype=np.int32
        )
        current_bedrock = true_bedrock.copy()
    consumed_stamina = 0
    water_sources = []
    for _ in range(W):
        y, x = map(int, input().split())
        water_sources.append([y, x])
    houses = []
    for _ in range(K):
        y, x = map(int, input().split())
        houses.append([y, x])

    def interact(y, x, P):
        nonlocal consumed_stamina
        consumed_stamina += C + P
        if SIMULATION:
            if not isinstance(P, int):
                raise TypeError
            if current_bedrock[y, x] <= 0:
                raise ValueError
            true_bedrock[y, x] -= P
            if true_bedrock[y, x] <= 0:
                return 1
            else:
                return 0
        print(y, x, P)
        r = int(input())
        if r == 2 or r == -1:
            exit()
        return r

    base_P = max(1, int(C * BASE_P_COEF))
    recovery_P = max(1, int(C * RECOVERY_P_COEF))

    sq_sigma = SIGMA**2  # 岩盤の頑丈さの事前分布の分散
    sq_sigma_noise = (base_P * NOISE_BASE_P_RATIO) ** 2.0 / sq_sigma  # 岩盤の頑丈さの測定誤差の分散
    sq_sigma_rbf = SIGMA_RBF**2.0  # 頑丈さの測定が周囲どれくらいの範囲の予測に影響するか

    n_rows = int(round(N_COLS * 2 / sqrt(3.0)))

    # 衛星の生成
    satellites = []
    satellites_edges = []
    for row in range(n_rows + 1):
        y = (N - 1) * row / n_rows
        if row % 2 == 0:
            for col in range(N_COLS):
                x = (N - 1) * (0.5 + col) / N_COLS
                v = len(satellites)
                satellites.append([y, x])
                if col != 0:
                    satellites_edges.append([v, v - 1])
                if row != 0:
                    satellites_edges.append([v, v - N_COLS - 1])
                    satellites_edges.append([v, v - N_COLS])
        else:
            for col in range(N_COLS + 1):
                x = (N - 1) * col / N_COLS
                v = len(satellites)
                satellites.append([y, x])
                if col != 0:
                    satellites_edges.append([v, v - 1])
                    satellites_edges.append([v, v - N_COLS - 1])
                if col != N_COLS:
                    satellites_edges.append([v, v - N_COLS])
    n_satellites = len(satellites)
    satellites_graph = [[] for _ in range(n_satellites)]
    for u, v in satellites_edges:
        satellites_graph[u].append(v)
        satellites_graph[v].append(u)

    # 衛星の家と水源への割り当て
    used_satellites = [False] * n_satellites
    house_and_water_source_satellites_indices = []
    dys = []
    dxs = []
    for hy, hx in houses + water_sources:
        best_distance = 1e300
        best_satellite_idx = -100
        for i, (cy, cx) in enumerate(satellites):
            if used_satellites[i]:
                continue
            distance = hypot(hy - cy, hx - cx)
            if distance < best_distance:
                best_distance = distance
                best_satellite_idx = i
        house_and_water_source_satellites_indices.append(best_satellite_idx)
        used_satellites[best_satellite_idx] = True
        cy, cx = satellites[best_satellite_idx]
        dys.append(hy - cy)
        dxs.append(hx - cx)

    # 衛星の位置の修正
    yxs = np.array(houses + water_sources)
    dys = np.array(dys)
    dxs = np.array(dxs)
    sq_sigma_dyx = (N / N_COLS * 0.25) ** 2
    gp_dy = GaussianProcess(
        yxs,
        dys,
        len(yxs),
        sq_sigma_noise=0.5**2 / sq_sigma_dyx,
        sq_sigma_rbf=40.0**2.0,
    )
    gp_dx = GaussianProcess(
        yxs,
        dxs,
        len(yxs),
        sq_sigma_noise=0.5**2 / sq_sigma_dyx,
        sq_sigma_rbf=40.0**2.0,
    )
    satellites_np = np.array(satellites)
    dys_all, _ = gp_dy.predict(satellites_np, 0.0, sq_sigma_dyx)
    dxs_all, _ = gp_dx.predict(satellites_np, 0.0, sq_sigma_dyx)
    satellites = [
        [max(0, min(N - 1, int(round(y + dy)))), max(0, min(N - 1, int(round(x + dx))))]
        for (y, x), dy, dx in zip(satellites, dys_all.tolist(), dxs_all.tolist())
    ]
    for idx_satellites, yx in zip(
        house_and_water_source_satellites_indices, houses + water_sources
    ):
        satellites[idx_satellites] = yx
    del dys_all, dxs_all, satellites_np, gp_dy, gp_dx, sq_sigma_dyx, dys, dxs, yxs

    def excavate(y, x, initial_P, P, current_sum_P=0, stop_sum_P=5000):
        # 閉区間で返す
        assert (10 <= initial_P <= 5000 and current_sum_P == 0) or (
            initial_P == 0 and current_sum_P >= 10
        ), initial_P
        assert 1 <= P <= 5000, P
        assert 0 <= y < N, y
        assert 0 <= x < N, x
        assert current_sum_P or initial_P
        if current_sum_P:
            sum_P = current_sum_P
        else:
            r = interact(y, x, initial_P)
            if r == 1:
                return 10, initial_P
            sum_P = initial_P
        while True:
            r = interact(y, x, P)
            if r == 1:
                return sum_P + 1, sum_P + P
            sum_P += P
            if stop_sum_P <= sum_P:
                return sum_P + 1, None

    # 掘削準備
    uf = UnionFind(K + 1)
    gp = GaussianProcess(
        np.zeros((0, 2)),
        np.zeros(0),
        n_max_data=2000,
        sq_sigma_noise=sq_sigma_noise,
        sq_sigma_rbf=sq_sigma_rbf,
    )
    excavated = [False] * (N * N)
    excavated_coords = []
    stopped = [False] * (N * N)
    satellites_np = np.array(satellites)
    satellite_states = np.zeros(n_satellites, dtype=np.int32)  # 0: 未到達, 1: 解放, 2: 閉鎖
    satellite_owners = [-100] * n_satellites  # state が 1 になったときにセットされる
    satellites_uf = UnionFind(n_satellites)
    for i, satellite_index in enumerate(house_and_water_source_satellites_indices[:K]):
        assert satellite_states[satellite_index] == 0
        satellite_states[satellite_index] = 1
        satellite_owners[satellite_index] = i
    for satellite_index in house_and_water_source_satellites_indices[K:]:
        assert satellite_states[satellite_index] == 0
        satellite_states[satellite_index] = 1
        satellite_owners[satellite_index] = K
        satellites_uf.unite(
            house_and_water_source_satellites_indices[K], satellite_index
        )

    n_excavated = 0
    stop_info = [None] * (N * N)

    def excavate_and_postprocess(
        satellite_index, initial_P, P, stop_sum_P=5000, stop_pred=None
    ):
        nonlocal n_excavated
        # 再開の時は initial_P いらない
        if stop_sum_P < 5000:
            assert stop_pred is not None
        y, x = satellites[satellite_index]
        v = y * N + x
        assert not excavated[v]
        current_sum_P = stop_info[v][0] if stopped[v] else 0
        mi, ma = excavate(y, x, initial_P, P, current_sum_P, stop_sum_P)
        if ma is None:
            gp_idx = gp.add_data(np.array([y, x]), float(max(stop_pred, mi)))
            stopped[v] = True
            stop_info[v] = (
                mi - 1,
                gp_idx,
            )
            return False
        excavated[v] = True
        excavated_coords.append([y, x])
        n_excavated += 1
        if mi <= ma - P:
            mi = max(10, ma - P * LEFT_TAIL_COEF)
        estimation = (ma + mi) * 0.5
        if stopped[v]:
            gp.modify_data(stop_info[v][1], estimation)
        else:
            gp.add_data(np.array([y, x]), estimation)

        assert satellite_states[satellite_index] != 2
        satellite_states[satellite_index] = 2
        owner = satellite_owners[satellite_index]
        assert owner != -100
        for satellite_u in satellites_graph[satellite_index]:
            if satellite_states[satellite_u] == 0:
                satellite_states[satellite_u] = 1
                assert satellite_owners[satellite_u] == -100
                satellite_owners[satellite_u] = owner
            elif satellite_states[satellite_u] == 1:
                # これだけだと逆転した時に困る
                if satellites_uf.count(satellite_index) < satellites_uf.count(
                    house_and_water_source_satellites_indices[
                        satellite_owners[satellite_u]
                    ]
                ):
                    satellite_owners[satellite_u] = owner
            elif satellite_states[satellite_u] == 2:
                uf.unite(owner, satellite_owners[satellite_u])
                satellites_uf.unite(satellite_index, satellite_u)
            else:
                assert False
        if uf.count(0) == K + 1:
            return True
        return False

    # 水源と家を掘る
    for i in house_and_water_source_satellites_indices:
        excavate_and_postprocess(i, 10 + base_P, base_P)

    # 優先度順に衛星を掘る
    while True:
        (candidate_indices,) = np.where(satellite_states == 1)
        mu = MU_START + (MU_END - MU_START) * (n_excavated / n_satellites)
        sturdiness_mean, sturdiness_var = gp.predict(
            satellites_np[candidate_indices], mu, sq_sigma
        )
        sturdiness_var = np.maximum(1e-2, sturdiness_var)  # なぜか負になる……
        sturdiness_std = np.sqrt(sturdiness_var)
        # TODO: これ修正
        # 小ささになってくれない
        system_size = np.array(
            [
                satellites_uf.count(
                    house_and_water_source_satellites_indices[satellite_owners[i]]
                )
                for i in candidate_indices.tolist()
            ]
        )

        inv_sizes = [
            1 / satellites_uf.count(i_sat)
            for i, i_sat in enumerate(
                house_and_water_source_satellites_indices[: K + 1]
            )
            if i == uf.root(i)
        ]
        sum_inv_sizes = sum(inv_sizes)

        # 小さいほど優先
        priority = (
            sturdiness_mean
            - PRIORITY_COEF_STURDINESS_STD * sturdiness_std
            - PRIORITY_COEF_SYSTEM_SIZE
            * np.mean(sturdiness_mean)
            * (1 + PRIORITY_COEF_SYSTEM_SIZE_K)
            / (system_size + PRIORITY_COEF_SYSTEM_SIZE_K)
            / sum_inv_sizes
        )

        best_candidate_idx = np.argmin(priority)
        best_satellite_idx = candidate_indices[best_candidate_idx]
        y, x = satellites[best_satellite_idx]
        v = y * N + x

        initial_P = (
            0
            if stopped[v]
            else max(
                10 + base_P,
                min(
                    5000,
                    int(
                        round(
                            sturdiness_mean[best_candidate_idx]
                            - COEF_INITIAL_P_STD * sturdiness_std[best_candidate_idx]
                        )
                    ),
                ),
            )
        )
        if stopped[v]:
            stop_sum_P = 5000
            stop_pred = None
        else:
            stop_sum_P = (
                sturdiness_mean[best_candidate_idx]
                + STOP_SIGMA * sturdiness_std[best_candidate_idx]
            )
            stop_pred = (
                sturdiness_mean[best_candidate_idx]
                + TEMPORAL_PREDICTION_COEF * sturdiness_std[best_candidate_idx]
            )
        finished = excavate_and_postprocess(
            best_satellite_idx, initial_P, base_P, stop_sum_P, stop_pred
        )
        if finished:
            break

    # シュタイナー

    mu = max(
        MIN_STEINER_MU, MU_START + (MU_END - MU_START) * (n_excavated / n_satellites)
    )
    X_pred = np.array([[y, x] for y in range(N) for x in range(N)])
    all_preds = gp.predict(X_pred, mu, sq_sigma, return_var=False)
    all_preds = np.clip(all_preds, 10.0, 5000.0)

    points = solve_dijkstra_and_steiner(
        all_preds.reshape(N, N).tolist(),
        excavated_coords,
        list(range(K)),
        list(range(K, K + W)),
    )

    # 掘る
    # TODO: これも予測しながら？

    for y, x in points:
        v = y * N + x
        if excavated[v]:
            continue
        excavated[v] = True
        initial_P = (
            0
            if stopped[v]
            else max(
                15,
                min(
                    5000,
                    int(round(all_preds[v])),
                ),
            )
        )
        if stopped[v]:
            current_sum_P = stop_info[v][0]
        else:
            current_sum_P = 0

        excavate(y, x, initial_P, recovery_P, current_sum_P)


main()
