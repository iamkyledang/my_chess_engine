# Superhuman Chess Engine (UCI) using CNN + MCTS
import sys
import torch
import chess
import numpy as np
import time
from collections import deque
import random

# --- Device ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Hyperparameters ---
IN_CHANNELS = 12
BOARD_SIZE = 8
NUM_RES_BLOCKS = 5
MCTS_SIMULATIONS = 1000  # Increase for stronger engine
C_PUCT = 1.0             # Exploration factor

# --- CNN Network ---
class ChessResNet(torch.nn.Module):
    def __init__(self, in_channels=12, board_size=8, num_res_blocks=10):
        super().__init__()
        self.conv_in = torch.nn.Conv2d(in_channels, 64, 3, padding=1)
        self.bn_in = torch.nn.BatchNorm2d(64)
        self.res_blocks = torch.nn.ModuleList([self._res_block(64) for _ in range(num_res_blocks)])
        # Policy head
        self.policy_head = torch.nn.Sequential(
            torch.nn.Conv2d(64, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(2*board_size*board_size, 64*64),
            torch.nn.Softmax(dim=-1)
        )
        # Value head
        self.value_head = torch.nn.Sequential(
            torch.nn.Conv2d(64, 1, 1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(board_size*board_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256,1),
            torch.nn.Tanh()
        )

    def _res_block(self, channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, 3, padding=1),
            torch.nn.BatchNorm2d(channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels, channels, 3, padding=1),
            torch.nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        out = torch.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            out = out + block(out)
        policy = self.policy_head(out)
        value = self.value_head(out)
        return policy, value

# --- Board to Tensor ---
def board_to_tensor(board):
    tensor = np.zeros((12,8,8), dtype=np.float32)
    for sq, piece in board.piece_map().items():
        row, col = divmod(sq,8)
        idx = piece.piece_type-1
        if piece.color == chess.BLACK:
            idx +=6
        tensor[idx,row,col] = 1
    return tensor

# --- Move indexing ---
def move_to_index(move):
    return move.from_square*64 + move.to_square

def index_to_move(idx):
    from_sq = idx // 64
    to_sq = idx % 64
    return chess.Move(from_sq, to_sq)

# --- MCTS Node ---
class MCTSNode:
    def __init__(self, board, parent=None, prior=1.0):
        self.board = board.copy()
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0
        self.prior = prior
    def value(self):
        return 0 if self.visits==0 else self.value_sum/self.visits

# --- MCTS Search ---
def mcts(model, root_board, simulations=MCTS_SIMULATIONS, c_puct=C_PUCT):
    root = MCTSNode(root_board)
    legal_moves = list(root_board.legal_moves)

    for _ in range(simulations):
        node = root
        path = [node]

        # Selection
        while node.children:
            ucb_scores = {m: child.value() + c_puct*child.prior*np.sqrt(node.visits)/(1+child.visits)
                          for m, child in node.children.items()}
            move = max(ucb_scores, key=ucb_scores.get)
            node = node.children[move]
            path.append(node)

        # Expansion & Evaluation
        if node.board.is_game_over():
            result = node.board.result()
            if result=='1-0': value=1.0
            elif result=='0-1': value=-1.0
            else: value=0.0
        else:
            x = torch.tensor(board_to_tensor(node.board), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            policy, value = model(x)
            value = value.item()
            policy = policy.detach().cpu().numpy().flatten()
            for move in node.board.legal_moves:
                idx = move_to_index(move)
                prior = policy[idx]
                node.children[move] = MCTSNode(node.board, parent=node, prior=prior)

        # Backpropagation
        for n in path:
            n.visits +=1
            n.value_sum += value

    # Pick most visited move
    best_move = max(root.children.items(), key=lambda x:x[1].visits)[0]
    return best_move

# --- UCI Loop ---
def uci_loop():
    model = ChessResNet(IN_CHANNELS, BOARD_SIZE, NUM_RES_BLOCKS).to(DEVICE)
    model.load_state_dict(torch.load("model_2.pt", map_location=DEVICE))
    model.eval()
    board = chess.Board()

    while True:
        try:
            line = sys.stdin.readline()
            if not line: break
            line = line.strip()

            if line == "uci":
                print("id name MyEngine_V2")
                print("id author kyle Dang")
                print("uciok")
            elif line == "isready":
                print("readyok")
            elif line.startswith("position"):
                if "startpos" in line:
                    board = chess.Board()
                    moves = line.split("moves")
                    if len(moves)>1:
                        for mv in moves[1].strip().split():
                            board.push_uci(mv)
                elif "fen" in line:
                    fen = line.split("fen")[1].split("moves")[0].strip()
                    board = chess.Board(fen)
                    if "moves" in line:
                        for mv in line.split("moves")[1].strip().split():
                            board.push_uci(mv)
            elif line.startswith("go"):
                time_limit = 2.0
                best_move = mcts(model, board, simulations=MCTS_SIMULATIONS)
                if best_move:
                    print(f"bestmove {best_move.uci()}")
                else:
                    print("bestmove 0000")
            elif line == "quit":
                break
            sys.stdout.flush()
        except Exception as e:
            print(f"info string error: {e}")
            sys.stdout.flush()

if __name__=="__main__":
    uci_loop()
