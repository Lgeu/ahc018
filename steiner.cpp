#include <algorithm>
#include <array>
#include <cassert>
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

    vector<vector<Edge>> G;
    vector<int> R; // ターミナル集合

    array<array<array<Index, 2>, kMaxNNodes>, 1 << (kMaxNTerminals - 1)> b;
    vector<pair<int, int>> result_edges; // ここに結果が格納される

    void Solve() {
        array<array<T, kMaxNNodes>, 1 << (kMaxNTerminals - 1)> l; // 距離
        const auto comp = [](const pair<T, Index>& a, const pair<T, Index>& b) {
            return a.first > b.first;
        };
        auto N = priority_queue<pair<T, Index>, vector<pair<T, Index>>,
                                decltype(comp)>(comp); // キュー
        auto P = array<array<bool, kMaxNNodes>,
                       1 << (kMaxNTerminals - 1)>(); // 確定したかどうか
        for (auto&& li : l)
            fill(li.begin(), li.end(), numeric_limits<T>::infinity());
        for (auto i = 0; i < (int)R.size() - 1; i++) {
            auto r = R[i];
            l[1 << i][r] = 0;
            N.emplace((T)0, Index{(short)(1 << i), (unsigned short)r});
        }
        for (auto v = 0; v < (int)G.size(); v++) {
            l[0][v] = 0;
            P[0][v] = true;
        }
        for (auto&& bi : b)
            for (auto&& bij : bi)
                bij = {Index{-1, 0}, {-1, 0}};
        const auto full = (short)((1 << ((int)R.size() - 1)) - 1);
        while (true) {
            const auto& [distance_Iv, Iv] = N.top();
            const auto [I, v] = Iv;
            if (l[I][v] != distance_Iv)
                continue;
            if (I == full && v == R.back())
                break;
            N.pop();
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