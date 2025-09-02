# My Chess Engine
This project is a custom chess engine built using **PyTorch**, combining a Convolutional Neural Network (CNN) with **Monte Carlo Tree Search (MCTS)**.  
The engine follows the **UCI (Universal Chess Interface)** standard, which allows it to run inside GUI frontends like **Cute Chess** alongside other engines (e.g., Stockfish, LCZero).

---

## 📌 Project Structure

- **`training.py`**  
  - Preprocesses chess game data (from PGN files).  
  - Converts board states into tensors.  
  - Trains a CNN on supervised data to predict policy (moves) and value (position evaluation).  
  - Saves the trained model as a `.pt` file (PyTorch checkpoint).  

- **`myengine.py`**  
  - Loads the `.pt` model file.  
  - Wraps the neural network with an **MCTS search loop**.  
  - Exposes the engine through a **UCI protocol loop**.  
  - Can be compiled into a `.exe` binary so that GUI programs can run it just like Stockfish.  

---

## 🚀 Workflow

1. **Train the Model**
   ```bash
   python training.py

2. **Run the engine**
   ```bash
   python myengine.py

3. **Compile to executable**
    ```bash
    pyinstaller --onefile myengine.py

    This will generate a standalone executable:
    ```bash
    dist/myengine.exe

4. ***Test in GUI software***
    ```bash
    Download and install Cute Chess GUI

    Open the engine manager and add myengine.exe as a UCI engine.


