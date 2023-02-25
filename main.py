import sys
from math import sqrt, hypot
from time import time

import numpy as np

from steiner import solve_steiner


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
        if perf:
            t_kernel_func_D = time() - t0
            print(f"t_kernel_func_D={t_kernel_func_D}", file=sys.stderr)
        np.square(D, out=D)
        K = D.sum(2)
        K *= -self.gamma
        np.exp(K, out=K)
        if perf:
            t_kernel_func = time() - t0
            print(f"t_kernel_func={t_kernel_func}", file=sys.stderr)
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

SIMULATION = False
if SIMULATION:
    filename = "./tools/in/0006.txt"
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
    global consumed_stamina
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


BASE_P_COEF = 3.0
RECOVERY_P_COEF = 3.0

base_P = max(1, int(C * BASE_P_COEF))
recovery_P = max(1, int(C * RECOVERY_P_COEF))

# mu = 2505.0  # 岩盤の頑丈さの事前分布の平均
MU_START = 2000.0  # [1000, 3000]
MU_END = 4000  # [2000, 4500]
sq_sigma = 1000.0**2  # 岩盤の頑丈さの事前分布の分散
sq_sigma_noise = (base_P * 1.0) ** 2 / sq_sigma  # 岩盤の頑丈さの測定誤差の分散
sq_sigma_rbf = 10.0**2.0  # 頑丈さの測定が周囲どれくらいの範囲の予測に影響するか

sqrt3 = sqrt(3.0)

n_cols = 20
n_rows = int(round(n_cols * 2 / sqrt3))

# 衛星の生成
satellites = []
satellites_edges = []
for row in range(n_rows + 1):
    y = (N - 1) * row / n_rows
    if row % 2 == 0:
        for col in range(n_cols):
            x = (N - 1) * (0.5 + col) / n_cols
            v = len(satellites)
            satellites.append([y, x])
            if col != 0:
                satellites_edges.append([v, v - 1])
            if row != 0:
                satellites_edges.append([v, v - n_cols - 1])
                satellites_edges.append([v, v - n_cols])
    else:
        for col in range(n_cols + 1):
            x = (N - 1) * col / n_cols
            v = len(satellites)
            satellites.append([y, x])
            if col != 0:
                satellites_edges.append([v, v - 1])
                satellites_edges.append([v, v - n_cols - 1])
            if col != n_cols:
                satellites_edges.append([v, v - n_cols])
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
sq_sigma_dyx = (N / n_cols * 0.25) ** 2
gp_dy = GaussianProcess(
    yxs, dys, len(yxs), sq_sigma_noise=0.5**2 / sq_sigma_dyx, sq_sigma_rbf=40.0**2.0
)
gp_dx = GaussianProcess(
    yxs, dxs, len(yxs), sq_sigma_noise=0.5**2 / sq_sigma_dyx, sq_sigma_rbf=40.0**2.0
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


def excavate(y, x, initial_P, P):
    # 閉区間で返す
    assert 10 <= initial_P <= 5000, initial_P
    assert 1 <= P <= 5000, P
    assert 0 <= y < N, y
    assert 0 <= x < N, x
    r = interact(y, x, initial_P)
    if r == 1:
        return 10, initial_P
    sum_P = initial_P
    while True:
        r = interact(y, x, P)
        if r == 1:
            return sum_P + 1, sum_P + P
        sum_P += P


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
    satellites_uf.unite(house_and_water_source_satellites_indices[K], satellite_index)

n_excavated = 0


def excavate_and_postprocess(satellite_index, initial_P, P):
    global n_excavated
    y, x = satellites[satellite_index]
    v = y * N + x
    assert not excavated[v]
    excavated[v] = True
    n_excavated += 1
    mi, ma = excavate(y, x, initial_P, P)
    left_tail_coef = 1.0
    if mi <= ma - P:
        mi = max(10, ma - P * left_tail_coef)
    estimation = (ma + mi) * 0.5
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
            if satellites_uf.count(satellite_index) < satellites_uf.count(satellite_u):
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
    # closed_indices, = np.where(satellite_states == 2)
    (candidate_indices,) = np.where(satellite_states == 1)
    mu = MU_START + (MU_END - MU_START) * (n_excavated / n_satellites)
    t0 = time()
    sturdiness_mean, sturdiness_var = gp.predict(
        satellites_np[candidate_indices], mu, sq_sigma
    )
    t_predict += time() - t0
    sturdiness_std = np.sqrt(sturdiness_var)
    # TODO: これ修正
    # 小ささになってくれない
    system_size = np.array([satellites_uf.count(i) for i in candidate_indices.tolist()])

    # 小さいほど優先
    coef_sturdiness_std = 0.2
    coef_system_size = 1.0
    coef_system_size_k = 2.0  # パラメータ [0, 10]
    priority = (
        sturdiness_mean
        - coef_sturdiness_std * sturdiness_std
        - coef_system_size
        * np.mean(sturdiness_mean)
        * (1.0 + coef_system_size_k)
        / (system_size + coef_system_size_k)
    )

    best_candidate_idx = np.argmin(priority)
    best_satellite_idx = candidate_indices[best_candidate_idx]
    initial_P = max(
        10 + base_P,
        min(
            5000,
            int(
                round(
                    sturdiness_mean[best_candidate_idx]
                    - 2.0 * sturdiness_std[best_candidate_idx]
                )
            ),
        ),
    )  # パラメータ
    finished = excavate_and_postprocess(best_satellite_idx, initial_P, base_P)
    if finished:
        break

t_satellite = time() - T0

print(f"t_satellite={t_satellite}", file=sys.stderr)
print(f"`- t_predict={t_predict}", file=sys.stderr)

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

# 各地点からの
# シュタイナー

# これもパラメータにした方が良いのでは
t0 = time()
mu = max(2505.0, MU_START + (MU_END - MU_START) * (n_excavated / n_satellites))
X_pred = np.array([[y, x] for y in range(N) for x in range(N)])
all_preds = gp.predict(X_pred, mu, sq_sigma, return_var=False)
t_all_prediction = time() - t0
print(f"t_all_prediction={t_all_prediction}", file=sys.stderr)

# solve_steiner()

# TODO: 信頼度が高い/低い順に掘る？


# 1. 水源と家を掘る

# 2. 予測をし、以下を繰り返す
#   a. 何らかの基準で候補地をソート
#   b. 最も良さそうな候補を、何らかの閾値まで掘り進める
#   c. 掘り切れなかった場合は次のループへ
#   d. 予測を更新し、全域木を構築できるようになった場合はループを抜ける

# 3. シュタイナー木を構築し、その通りにやる？

# クラスカルっぽくやるのが多分良い
# いやブルーフカっぽく？

# 左手法はやや深めの部分を掘ることになるので微妙？

# 山を下る必要がある場合
# * 上下左右を間隔開けて 1 つ抜けるまで掘る
# * 抜けた部分のからの上下左右も 1 つ抜けるまで掘る
#   * コストが増えるしかなかったら元のも探す
# 山を下るのはそもそも効率悪い？
# * 山スタートをどの程度許容するかパラメータにできないか？
# 予測は？

# 何らかの基準で掘る場所と閾値を決める
# * 既に掘り進めていること
# * 予測値が小さいこと
# * 未知の家に近いこと (所属する連結成分の小ささ)
# * 既に掘った場所に囲まれていないこと
# * 座標が中央に近いこと

# 誤読！！！！！！！！！！！！！！！！！！！！！
