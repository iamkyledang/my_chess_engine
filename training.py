import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import chess
import chess.pgn

# -----------------------------
# --- Hyperparameters ---------
# -----------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

BATCH_SIZE = 1028   # Increase if you have enough GPU memory
EPOCHS = 1
LR = 1e-3
IN_CHANNELS = 12
BOARD_SIZE = 8
NUM_RES_BLOCKS = 5

# Paths
PGN_DATASET_DIR = r"D:\THIRD YEAR\chess_engine\dataset_2"
CHUNK_DIR = "F:/chess_dataset_2"
os.makedirs(CHUNK_DIR, exist_ok=True)
CHUNK_SIZE = 10000  # positions per chunk

# -----------------------------
# --- Chess Board Encoding ----
# -----------------------------
def board_to_tensor(board):
    """Convert a chess.Board into a 12x8x8 tensor."""
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for sq, piece in board.piece_map().items():
        row, col = divmod(sq, 8)
        idx = piece.piece_type - 1
        if piece.color == chess.BLACK:
            idx += 6
        tensor[idx, row, col] = 1
    return tensor

def move_to_index(move):
    """Map a move to an integer index (0..4095)."""
    return move.from_square * 64 + move.to_square

# -----------------------------
# --- Model -------------------
# -----------------------------
class ChessResNet(nn.Module):
    def __init__(self, in_channels, board_size, num_blocks):
        super().__init__()
        self.board_size = board_size

        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(64)

        self.res_blocks = nn.ModuleList([self._res_block(64) for _ in range(num_blocks)])

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, 64 * 64),
            nn.Softmax(dim=-1)
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def _res_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        out = torch.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            out = out + block(out)
        policy = self.policy_head(out)
        value = self.value_head(out)
        return policy, value

# -----------------------------
# --- PGN Preprocessing ------- 
# -----------------------------
def preprocess_pgns(dataset_dir, chunk_dir, chunk_size=10000):
    """Parse PGN files into chunked tensors for training."""
    states, policies, values = [], [], []
    chunk_idx = 0
    total_games, total_positions = 0, 0

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"PGN folder not found: {dataset_dir}")

    print(f"Starting PGN parsing in: {dataset_dir}")
    file_count = 0

    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if not f.endswith(".pgn"):
                continue
            file_count += 1
            path = os.path.join(root, f)
            print(f"Parsing file {file_count}: {path}")

            with open(path, encoding="utf-8", errors="ignore") as f_in:
                game_count = 0
                while True:
                    game = chess.pgn.read_game(f_in)
                    if game is None:
                        break

                    result = game.headers.get("Result", "*")
                    if result == "1-0":
                        value = 1.0
                    elif result == "0-1":
                        value = -1.0
                    elif result == "1/2-1/2":
                        value = 0.0
                    else:
                        continue  # skip games without result

                    total_games += 1
                    game_count += 1

                    board = game.board()
                    for move in game.mainline_moves():
                        states.append(board_to_tensor(board))

                        policy_vec = np.zeros(64 * 64, dtype=np.float32)
                        policy_vec[move_to_index(move)] = 1.0
                        policies.append(policy_vec)
                        values.append(value)

                        board.push(move)
                        total_positions += 1

                        # Save chunk
                        if len(states) >= chunk_size:
                            chunk = {
                                "states": torch.tensor(np.array(states), dtype=torch.float32),
                                "policies": torch.tensor(np.array(policies), dtype=torch.float32),
                                "values": torch.tensor(np.array(values), dtype=torch.float32).unsqueeze(1)
                            }
                            chunk_file = os.path.join(chunk_dir, f"chunk{chunk_idx}.pt")
                            torch.save(chunk, chunk_file)
                            print(f"  Saved chunk {chunk_idx} ({len(states)} positions) to {chunk_file}")
                            chunk_idx += 1
                            states, policies, values = [], [], []

                print(f"  Parsed {game_count} games in current file.")

    # Save leftover positions
    if len(states) > 0:
        chunk = {
            "states": torch.tensor(np.array(states), dtype=torch.float32),
            "policies": torch.tensor(np.array(policies), dtype=torch.float32),
            "values": torch.tensor(np.array(values), dtype=torch.float32).unsqueeze(1)
        }
        chunk_file = os.path.join(chunk_dir, f"chunk{chunk_idx}.pt")
        torch.save(chunk, chunk_file)
        print(f"  Saved final chunk {chunk_idx} ({len(states)} positions) to {chunk_file}")

    print(f"All files parsed. Total games: {total_games}, total positions: {total_positions}")

# -----------------------------
# --- Training ----------------
# -----------------------------
def train_model():
    chunk_files = sorted(glob.glob(os.path.join(CHUNK_DIR, "chunk*.pt")))

    model = ChessResNet(IN_CHANNELS, BOARD_SIZE, NUM_RES_BLOCKS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_batches = 0, 0
        print(f"Epoch {epoch+1}/{EPOCHS} starting...")

        for chunk_idx, chunk_file in enumerate(chunk_files):
            print(f"  Loading chunk {chunk_idx}: {chunk_file}")
            chunk = torch.load(chunk_file)
            X, P, V = chunk["states"], chunk["policies"], chunk["values"]
            train_dataset = TensorDataset(X, P, V)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)

            for xb, pb, vb in train_loader:
                xb, pb, vb = xb.to(DEVICE), pb.to(DEVICE), vb.to(DEVICE)
                optimizer.zero_grad()
                pred_policy, pred_value = model(xb)

                loss_policy = criterion_policy(pred_policy, pb.argmax(dim=1))
                loss_value = criterion_value(pred_value, vb)
                loss = loss_policy + loss_value

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        print(f"Epoch {epoch+1}/{EPOCHS}, Avg Loss: {avg_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), "model_2.pt")
    print("model_2.pt")


if __name__ == "__main__":
    # Step 1: Preprocess PGNs -> chunked dataset
    preprocess_pgns(PGN_DATASET_DIR, CHUNK_DIR, CHUNK_SIZE)

    # Step 2: Train model from chunks
    train_model()
