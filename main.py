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
        self.gamma = 0.5 / sq_sigma_rbf**2
        self.K = np.empty((n_max_data, n_max_data))
        K = self.kernel_func(X, X)
        K.flat[:: n_data + 1] += sq_sigma_noise
        self.K[:n_data, :n_data] = K
        self.K_inv = np.empty((n_max_data, n_max_data))
        self.K_inv[:n_data, :n_data] = np.linalg.inv(self.K[:n_data, :n_data])
        self.sq_sigma_noise = sq_sigma_noise
        self.n_data = n_data
        self.X = np.empty((n_max_data, 2), dtype=np.int32)
        self.X[:n_data] = X
        self.y = np.empty(n_max_data)
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

    def kernel_func(self, A, B=None):
        """
        Args:
            A (np.ndarray): [n_data_a, 2]
            B: [n_data_b, 2]
        """
        if B is None:
            return np.ones(len(A))
        assert A.shape[1] == B.shape[1] == 2
        D = A[:, None, :] - B[None, :, :]
        K = np.hypot(D[:, :, 0], D[:, :, 1])
        K = np.exp(-self.gamma * K)
        return K

    def predict(self, X, mu, sq_sigma):
        """
        Args:
            X: [n_data_pred, 2]
            mu (float):
            sq_sigma (float):
        """
        assert X.shape[1] == 2
        # [n_data_pred, n_data]  O(n_data_pred * n_data)
        k_star_T = self.kernel_func(X, self.X[: self.n_data])
        # [n_data_pred, n_data]  O(n_data_pred * n_data * n_data)
        k_star_T_K_inv = k_star_T @ self.K_inv[: self.n_data, : self.n_data]
        # [n_data_pred]
        mean = k_star_T_K_inv @ (self.y[: self.n_data] - mu) + mu
        # [n_data_pred]
        k_star_star = self.kernel_func(X)
        # [n_data_pred]
        var = (k_star_star - (k_star_T_K_inv * k_star_T).sum(1)) * sq_sigma
        return mean, var


N = 200

filename = 0
with open(filename) as f:
    _, K, W, C = map(int, f.readline().split())
    # true_bedrock = np.array([list(map(int, f.readline().split())) for _ in range(N)], dtype=np.int32)
    water_sources = []
    for _ in range(W):
        y, x = map(int, f.readline().split())
        water_sources.append([y, x])
    houses = []
    for _ in range(K):
        y, x = map(int, f.readline().split())
        houses.append([y, x])


def interact(y, x, P):
    print(y, x, P)
    r = int(f.readline())
    if r == 2 or r == -1:
        exit()
    return r


BASE_P_COEF = 3.0
RECOVERY_P_COEF = 3.0

base_P = max(1, int(C * BASE_P_COEF))
recovery_P = max(1, int(C * RECOVERY_P_COEF))

mu = 2505.0  # 岩盤の頑丈さの事前分布の平均
sq_sigma = 1000.0**2  # 岩盤の頑丈さの事前分布の分散
sq_sigma_noise = (10.0 / mu) ** 2  # 岩盤の頑丈さの測定誤差の分散
sq_sigma_rbf = 10.0  # 頑丈さの測定が周囲どれくらいの範囲の予測に影響するか

dominance_radius = 10.0  # 連結とみなす範囲
dominance_deltas = []
candidate_dominance_deltas = []

# TODO: 6 方向
for y in range(-int(dominance_radius), int(dominance_radius) + 1):
    for x in range(-int(dominance_radius), int(dominance_radius) + 1):
        if y == 0 and x == 0:
            continue
        if y * y + x * x <= dominance_radius * dominance_radius:
            dominance_deltas.append((y, x))
            if y * y + x * x <= dominance_radius * dominance_radius * 0.4 * 0.4:
                # 新しく候補を置くのを避ける範囲
                candidate_dominance_deltas.append((y, x))

#

dominator = [-1] * (N * N)
excavated = [False] * (N * N)
house_and_water_source_board = [-1] * (N * N)
candidate_satellites = set()
candidate_dominance_board = [False] * (N * N)
for i, (y, x) in enumerate(houses):
    dominator[y * N + x] = i
    house_and_water_source_board[y * N + x] = i
for y, x in enumerate(water_sources):
    dominator[y * N + x] = K
    house_and_water_source_board[y * N + x] = K

# TODO: 既存の候補に近すぎるの (距離 2 とか) は入れないようにする


def excavate(
    y,
    x,
    initial_P,
    P,
):
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


uf = UnionFind(K + 1)
gp = GaussianProcess(
    np.zeros(0, 2),
    np.zeros(0),
    n_max_data=2000,
    sq_sigma_noise=sq_sigma_noise,
    sq_sigma_rbf=sq_sigma_rbf,
)


def excavate_and_postprocess(y, x, initial_P, P):
    mi, ma = excavate(y, x, initial_P, P)
    left_tail_coef = 1.0
    if mi <= ma - P:
        mi = max(10, ma - P * left_tail_coef)
    estimation = (ma + mi) * 0.5
    gp.add_data(np.array([y, x]), estimation)

    v = y * N + x
    excavated[v] = True
    # TODO: ここで6方向？

    for dy, dx in dominance_deltas:
        uy = y + dy
        ux = x + dx
        if 0 <= uy < N and 0 <= ux < N:
            u = uy * N + ux
            if house_and_water_source_board[u] != -1:
                uf.unite(dominator[v], house_and_water_source_board[u])
                continue
            if dominator[u] == -1:
                dominator[u] = dominator[v]

    if uf.count(0) == len(houses) + 1:
        return True
    return False


# 水源と家を掘る
for y, x in water_sources:
    excavate_and_postprocess(y, x, 10 + base_P, base_P)
for y, x in houses:
    excavate_and_postprocess(y, x, 10 + base_P, base_P)

# 優先度順に掘る
while True:
    # TODO: 近場を既に掘っていれば候補から外す
    for satellite in satellites:
        pass


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

# 誤読！！！！！！！！！！！！！！！！！！！！！
