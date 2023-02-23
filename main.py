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

# 実は DFS っぽくしたほうがいいのでは

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
