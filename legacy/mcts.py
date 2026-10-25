# mcts.py

import numpy as np
import torch # モデルがPyTorchの場合

class MCTSNode:
    def __init__(self, prior, parent=None):
        self.parent = parent
        self.children = {}  # action_index -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior

    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_child(self, c_puct):
        best_score = -float('inf')
        best_action = -1
        best_child = None

        # 訪問回数の平方根を一度だけ計算（高速化）
        sqrt_total_visit = np.sqrt(self.visit_count)

        for action, child in self.children.items():
            # PUCT公式: Q + U
            u_score = c_puct * child.prior * sqrt_total_visit / (1 + child.visit_count)
            score = child.value + u_score

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def expand(self, action_probs):
        """
        action_probs: (action_index, probability) のリスト
        """
        for action, prob in action_probs:
            if action not in self.children:
                self.children[action] = MCTSNode(prior=prob, parent=self)

    def is_expanded(self):
        return len(self.children) > 0


class MCTS:
    def __init__(self, model, c_puct=1.0):
        self.model = model
        self.c_puct = c_puct

    def search(self, board, node):
        """1回のシミュレーションを実行"""
        
        # 1. Selection (選択)
        # 葉ノード（まだ展開されていないノード）に辿り着くまで潜る
        path = []
        while node.is_expanded():
            action, node = node.select_child(self.c_puct)
            board.make_move(decode_move(action)) # board.pyで定義したdecodeを使用
            path.append(action)

        # 2. Evaluation & Expansion (評価と展開)
        # NNに盤面を入力して評価を得る
        state_tensor = torch.FloatTensor(board.to_tensor()).unsqueeze(0) # (1, 4, 9, 9)
        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)
        
        # 合法手のみを抽出して正規化
        mask = board.get_action_mask()
        probs = torch.softmax(policy_logits, dim=1).numpy()[0]
        valid_probs = probs * mask
        if valid_probs.sum() > 0:
            valid_probs /= valid_probs.sum()
        
        # 葉ノードを展開
        action_probs = [(i, p) for i, p in enumerate(valid_probs) if p > 0]
        node.expand(action_probs)

        # 3. Backup (逆伝播)
        # Value（-1~1）を親に伝播させる。
        # Quoridorは交互着手なので、1ステップごとに符号を反転させる必要がある点に注意
        v = value.item()
        self._backpropagate(node, v)

        # 探索のために動かした盤面を元に戻す
        for _ in range(len(path)):
            board.undo_move()

    def _backpropagate(self, node, v):
        """親ノードに向かって価値を更新"""
        node.value_sum += v
        node.visit_count += 1
        
        if node.parent is not None:
            # 相手の番での評価は自分にとってマイナス
            self._backpropagate(node.parent, -v)
            

if __name__ == "__main__":
    from board import Board, decode_move
    import torch
    import torch.nn as nn

    # 簡易的なダミーモデル (入力: 4ch, 9x9 -> 出力: 209, 1)
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(4, 16, kernel_size=3, padding=1)
            self.policy_head = nn.Linear(16 * 9 * 9, 209)
            self.value_head = nn.Linear(16 * 9 * 9, 1)

        def forward(self, x):
            x = torch.relu(self.conv(x))
            x = x.view(x.size(0), -1)
            return self.policy_head(x), torch.tanh(self.value_head(x))

    # 初期化
    board = Board()
    model = DummyModel()
    mcts = MCTS(model)
    root = MCTSNode(prior=1.0)

    print("探索開始...")
    # 100回シミュレーションを回してみる
    for i in range(100):
        mcts.search(board, root)
        if (i + 1) % 10 == 0:
            print(f"Simulation {i+1} done.")

    # 最も訪問回数が多い手を選択
    best_action = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
    print(f"\n決定した指し手: {decode_move(best_action)}")
    print(f"ルートノードの訪問回数: {root.visit_count}")