#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <fstream>
#include <functional>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

using namespace std;

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

static PyObject* Solve(PyObject* self, PyObject* args) {
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
    while (Gv_py = PyIter_Next(G_py_iter)) {
        auto Gv = vector<SteinerTree::Edge>();
        Gv_py_iter = PyObject_GetIter(Gv_py);
        if (Gv_py_iter == NULL)
            return NULL;
        while (Gvi_py = PyIter_Next(Gv_py_iter)) {
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
    while (Ri_py = PyIter_Next(R_py_iter)) {
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
