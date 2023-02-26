import sys
from math import sqrt, hypot
from time import time

import numpy as np

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
    # TODO: 修正コレスキー分解
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
        if perf:
            t0 = time()
        D = A[:, None, :].astype(np.float32) - B[None, :, :].astype(np.float32)
        # if perf:
        #     t_kernel_func_D = time() - t0
        #     print(f"t_kernel_func_D={t_kernel_func_D}", file=sys.stderr)
        np.square(D, out=D)
        K = D.sum(2)
        K *= -self.gamma
        np.exp(K, out=K)
        # if perf:
        #     t_kernel_func = time() - t0
        #     print(f"t_kernel_func={t_kernel_func}", file=sys.stderr)
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

params = dict(
    BASE_P_COEF=3.0,
    RECOVERY_P_COEF=3.0,
    STOP_SIGMA=0.5,
    TEMPORAL_PREDICTION_COEF=1.5,
    MU_START=1500.0,  # [500, 2500]  # 岩盤の頑丈さの事前分布の平均
    MU_END=2500.0,  # [1500, 4000]
    SIGMA=1000.0,  # 岩盤の頑丈さの事前分布の標準偏差
    NOISE_BASE_P_RATIO=1.0,
    SIGMA_RBF=10.0,
    N_COLS=18,  # [10, 25]
    LEFT_TAIL_COEF=1.0,
    PRIORITY_COEF_STURDINESS_STD=0.0,  # [-1, 1]
    PRIORITY_COEF_SYSTEM_SIZE=1.0,
    PRIORITY_COEF_SYSTEM_SIZE_K=2.0,  # [0, 10]
    COEF_INITIAL_P_STD=2.0,  # [0.5, 4]
    MIN_STEINER_MU=2000.0,  # [1000.0, 3000.0]
)

# 11.636111155963222
params = {
    "BASE_P_COEF": 4.086361827749314,
    "COEF_INITIAL_P_STD": 2.716238958014798,
    "LEFT_TAIL_COEF": 2.1152027331538013,
    "MIN_STEINER_MU": 1044.017407551411,
    "MU_END": 1980.4716569162727,
    "MU_START": 1195.425808024243,
    "NOISE_BASE_P_RATIO": 0.5629917195152011,
    "N_COLS": 15,
    "PRIORITY_COEF_STURDINESS_STD": 0.2602150750825655,
    "PRIORITY_COEF_SYSTEM_SIZE": 1.990928304497543,
    "PRIORITY_COEF_SYSTEM_SIZE_K": 2.383998528482987,
    "RECOVERY_P_COEF": 3.7019778138973525,
    "SIGMA": 1592.7288608682672,
    "SIGMA_RBF": 13.157213004689595,
    "STOP_SIGMA": 0.14420014220340238,
    "TEMPORAL_PREDICTION_COEF": 0.2361066425020843,
}

# 11.583815975676561s
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

    for arg in sys.argv[1:]:
        if "=" in arg:
            arg = arg.split("=")
            if len(arg) == 2:
                l, r = arg
                try:
                    params[l] = eval(r)
                except:
                    pass

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
        filename = "./tools/in/0165.txt"
        f = open(filename)
        input = f.readline
    # python3 main.py
    # ./tools/target/release/tester python3 main.py < ./tools/in/0006.txt > out.txt

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
        # print(y, x, P, file=sys.stderr)
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

    # 確認
    if False:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 8))
        satellites_np = np.array(satellites)
        plt.scatter(satellites_np[:, 0], satellites_np[:, 1])
        house_and_water_sources_np = np.array(houses + water_sources)
        plt.scatter(house_and_water_sources_np[:, 0], house_and_water_sources_np[:, 1])
        for u, v in satellites_edges:
            uy, ux = satellites[u]
            vy, vx = satellites[v]
            plt.plot([uy, vy], [ux, vx])
        plt.show()

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

    T0 = time()
    t_predict = 0.0

    # 優先度順に衛星を掘る
    while True:
        (candidate_indices,) = np.where(satellite_states == 1)
        mu = MU_START + (MU_END - MU_START) * (n_excavated / n_satellites)
        t0 = time()
        sturdiness_mean, sturdiness_var = gp.predict(
            satellites_np[candidate_indices], mu, sq_sigma
        )
        sturdiness_var = np.maximum(1e-2, sturdiness_var)  # なぜか負になる……
        t_predict += time() - t0
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

    t_satellite = time() - T0

    # print(f"t_satellite={t_satellite}", file=sys.stderr)
    # print(f"`- t_predict={t_predict}", file=sys.stderr)

    # 検証
    if False:
        import matplotlib.pyplot as plt

        print(f"mu={mu}", file=sys.stderr)
        cmap = "coolwarm"
        X_pred = np.array([[y, x] for y in range(N) for x in range(N)])
        mean = gp.predict(X_pred, mu, sq_sigma, return_var=False)
        mean = mean.reshape(N, N)
        excavated_np = np.array(excavated)
        X = X_pred[excavated_np]
        plt.figure(figsize=(6, 6))
        plt.imshow(mean, cmap=cmap, vmin=10, vmax=5000)
        plt.colorbar()
        plt.scatter(X[:, 1], X[:, 0], s=4, c="white", marker="x")
        plt.show()

    # シュタイナー

    t0 = time()
    mu = max(
        MIN_STEINER_MU, MU_START + (MU_END - MU_START) * (n_excavated / n_satellites)
    )
    X_pred = np.array([[y, x] for y in range(N) for x in range(N)])
    all_preds = gp.predict(X_pred, mu, sq_sigma, return_var=False)
    all_preds = np.clip(all_preds, 10.0, 5000.0)
    t_all_prediction = time() - t0
    # print(f"t_all_prediction={t_all_prediction}", file=sys.stderr)

    points = solve_dijkstra_and_steiner(
        all_preds.reshape(N, N).tolist(),
        excavated_coords,
        list(range(K)),
        list(range(K, K + W)),
    )

    if False:
        import matplotlib.pyplot as plt

        points_np = np.array(points)
        cmap = "coolwarm"
        X = X_pred[np.array(excavated)]
        plt.figure(figsize=(6, 6))
        plt.imshow(all_preds.reshape(N, N), cmap=cmap, vmin=10, vmax=5000)
        plt.colorbar()
        plt.scatter(points_np[:, 1], points_np[:, 0], s=2, c="yellow")
        plt.scatter(X[:, 1], X[:, 0], s=4, c="white", marker="x")
        plt.show()

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

    # TODO: 信頼度が高い/低い順に掘る？
    # TODO: 累乗にする


main()
