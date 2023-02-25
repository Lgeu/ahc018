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
    vector<vector<Edge>> G; //グラフ
    vector<int> R;          // ターミナル集合

    array<array<array<Index, 2>, kMaxNNodes>, 1 << (kMaxNTerminals - 1)> b;
    vector<pair<int, int>> result_edges; // ここに結果が格納される

    void Solve() {
        auto os = ofstream("log2.txt");
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
        os << "ok1" << endl;
        os << "G.size()=" << G.size() << endl;
        os << "R.size()=" << R.size() << endl;
        while (N.size()) {
            const auto [distance_Iv, Iv] = N.top();
            const auto& [I, v] = Iv;
            os << "d,I,v=" << distance_Iv << "," << I << "," << v << endl;
            N.pop();
            if (l[I][v] != distance_Iv)
                continue;
            if (I == full && v == R.back())
                break;
            P[I][v] = true;
            for (const auto& [w, c] : G[v]) {
                os << "w,c=" << w << ","
                   << "c" << endl;
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
        os << "ok2" << endl;
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
        ofstream os("error_log.txt");                                          \
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
    const auto water_sources = PyIterableToVector<int>(houses_py, parse_i);
    for (const auto& s : satellites)
        board[s.y][s.x] = 0;

    auto steiner = SteinerTree();
    auto& G = steiner.G;
    auto& R = steiner.R;

    // 各衛星に対して
    auto target_ids = array<array<int, 200>, 200>();
    for (auto&& t : target_ids)
        fill(t.begin(), t.end(), -1);
    const auto n_satellites = (int)satellites.size();
    const auto n_targets = min(20, n_satellites);
    auto satellite_numbers = vector<int>(n_satellites);
    iota(satellite_numbers.begin(), satellite_numbers.end(), 0);

    auto distances = array<array<float, 200>, 200>();
    static auto from = array<array<array<signed char, 200>, 200>, 2000>();
    for (auto&& d : distances)
        fill(d.begin(), d.end(), numeric_limits<float>::infinity());
    static constexpr auto kDirections =
        array<Point, 4>{Point{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    G.resize(n_satellites);
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
                    if ((u - start).l1_norm() > max_l_distance)
                        continue;
                    const auto u_distance = v_distance + board[u.y][u.x];
                    if (u_distance < distances[u.y][u.x]) {
                        distances[u.y][u.x] = u_distance;
                        from[idx_satellites][u.y][u.x] = (signed char)(i + 1);
                        q.emplace(u_distance, u);
                    }
                }
                if (n_found_satellites != n_targets) {
                    ERROR();
                }
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
}

static PyObject* Solve(PyObject* /* self */, PyObject* args) {
    auto os = ofstream("log.txt");
    os << "ok1" << endl;
    // グラフとターミナル集合を受け取る
    auto steiner = SteinerTree();
    PyObject *G_py, *R_py, *G_py_iter, *Gv_py, *Gv_py_iter, *Gvi_py,
        *Gvi_py_iter, *R_py_iter, *Ri_py, *item;
    PARSE_ARGS("OO", &G_py, &R_py);
    G_py_iter = PyObject_GetIter(G_py);
    if (G_py_iter == NULL)
        return NULL;
    while ((Gv_py = PyIter_Next(G_py_iter))) {
        auto Gv = vector<SteinerTree::Edge>();
        Gv_py_iter = PyObject_GetIter(Gv_py);
        if (Gv_py_iter == NULL)
            return NULL;
        while ((Gvi_py = PyIter_Next(Gv_py_iter))) {
            Gvi_py_iter = PyObject_GetIter(Gvi_py);
            if (Gvi_py_iter == NULL)
                return NULL;
            item = PyIter_Next(Gvi_py_iter);
            if (item == NULL)
                return NULL;
            const auto to = (int)PyLong_AsLong(item);
            Py_DECREF(item);
            item = PyIter_Next(Gvi_py_iter);
            if (item == NULL)
                return NULL;
            const auto cost = (float)PyFloat_AsDouble(item);
            Py_DECREF(item);
            os << to << " " << cost << endl;
            Gv.push_back({to, cost});
            Py_DECREF(Gvi_py_iter);
            Py_DECREF(Gvi_py);
        }
        Py_DECREF(Gv_py_iter);
        if (PyErr_Occurred())
            return NULL;
        steiner.G.push_back(Gv);
        Py_DECREF(Gv_py);
    }
    Py_DECREF(G_py_iter);
    if (PyErr_Occurred())
        return NULL;
    R_py_iter = PyObject_GetIter(R_py);
    if (R_py_iter == NULL)
        return NULL;
    while ((Ri_py = PyIter_Next(R_py_iter))) {
        const auto r = (int)PyLong_AsLong(Ri_py);
        steiner.R.push_back(r);
        Py_DECREF(Ri_py);
    }
    Py_DECREF(R_py_iter);
    if (PyErr_Occurred())
        return NULL;

    os << "ok2" << endl;
    steiner.Solve();
    os << "ok3" << endl;

    PyObject* result_py = PyList_New(steiner.result_edges.size());
    if (result_py == NULL)
        return NULL;
    for (auto i = 0; i < (int)steiner.result_edges.size(); i++) {
        item = Py_BuildValue("[ii]", steiner.result_edges[i].first,
                             steiner.result_edges[i].second);
        if (item == NULL)
            return NULL; // 本当は result_py を DECREF する必要が
        PyList_SET_ITEM(result_py, i, item);
    }
    return result_py;
}

static PyMethodDef steiner_methods[] = {
    {"solve_steiner", (PyCFunction)Solve, METH_VARARGS, "solve_steiner"},
    {NULL},
};

static PyModuleDef steiner_module = {PyModuleDef_HEAD_INIT, "steiner", NULL, -1,
                                     steiner_methods};

PyMODINIT_FUNC PyInit_steiner(void) { return PyModule_Create(&steiner_module); }
