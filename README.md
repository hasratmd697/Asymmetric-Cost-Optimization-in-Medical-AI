# Asymmetric Cost Optimization in Medical AI: Chest X-ray Thoracic Pathology Classification

Automated classification of chest X-ray images into 20 thoracic pathologies (e.g., Atelectasis, Cardiomegaly, Effusion, Pneumothorax). This repository implements a robust deep learning pipeline optimized for **asymmetric misdiagnosis costs**, achieving a leaderboard score of **-4.34**.

##  Problem & Clinical Context
In clinical AI, misdiagnoses carry asymmetric consequences. A missed pathology (False Negative) is significantly more dangerous to patient outcomes than a false alarm (False Positive). 

To model this clinical reality, the model was evaluated under a strict asymmetric score $S_c$ per class:
$$S_c = \frac{TP_c - FP_c - 5 \cdot FN_c}{N_c}$$
Where a **False Negative is penalised 5 times more heavily** than a False Positive. The dataset also exhibits severe class imbalance, dominated by "No Finding," while critical pathologies like Hernia are extremely rare.

---

##  Key Methodologies & Tech Stack
* **Deep Learning Framework:** PyTorch & Torchvision
* **Model Architecture:** **DenseNet-121** fine-tuned with a custom classification head:
  `BatchNorm1d ➜ Dropout(0.4) ➜ Linear(1024 ➜ 512) ➜ ReLU ➜ BatchNorm1d ➜ Dropout(0.2) ➜ Linear(512 ➜ 20)`.
* **Hardware Accelerator:** PyTorch Mixed Precision (`autocast` / FP16) on CUDA.

### 1. Decoupled Training & Decision Boundary Tuning
Directly training a network on highly asymmetric cost functions degrades representation learning, forcing the model to collapse toward predicting rare classes. Instead, our pipeline decouples these objectives:
1. **Representational Learning:** Train the model for balanced class probabilities.
2. **Decision Calibration:** Apply asymmetric cost adjustments post-hoc at inference.

### 2. Class Imbalance Mitigation (Training Time)
* **Sqrt-Inverse-Frequency Weighted Sampling:** Applied PyTorch's `WeightedRandomSampler` with weights scaled as $1 / \sqrt{N_c}$. Sqrt scaling acts as a regulariser; full inverse frequency ($1/N_c$) over-boosts rare classes, producing high false positives.
* **Balanced Cross-Entropy Loss:** Scaled loss weights normalising class frequencies.
* **Label Smoothing ($\epsilon=0.05$):** Prevented the network from over-calibrating to extreme probabilities, crucial for post-hoc threshold adjustment.

### 3. Ensembling & Test-Time Augmentation (TTA)
* **5-Fold Stratified Cross-Validation:** Trained 3 folds to build a diverse model ensemble.
* **TTA Pipeline:** Performed 5 rounds of stochastic TTA (horizontal flips + minor rotations) per fold.
* **Forward Passes:** Each test prediction averaged probabilities across $3\text{ models} \times (1\text{ base} + 5\text{ TTA}) = 18$ forward passes to smooth decision boundaries.

### 4. Post-Hoc Cost-Sensitive Exponent Scaling (The Math)
For a single sample, maximizing the expected clinical score simplifies to selecting the class $c$ maximizing:
$$\text{Adjusted Prob}(c) = P(c) \cdot \left(\frac{1}{N_c + 1}\right)^\alpha$$
Where $N_c$ is the training sample count for class $c$, and $\alpha \in [0.0, 1.0]$ is a validation-tuned exponent:
* $\alpha = 0.0$ yields standard argmax classification.
* $\alpha = 1.0$ yields full frequency-based cost adjustment.
Grid-searching $\alpha$ on our validation folds yielded the optimal tradeoff, boosting the leaderboard score to **-4.34**.

---

##  Project Structure
```bash
.
├── train.py                  # Custom training script with PyTorch pipeline
├── README.md                 # Project documentation
├── .gitignore                # Git ignore file (excludes local data/academic markers)
└── data/                     # (Create this directory for datasets)
    ├── train.csv
    ├── test.csv
    └── images/
```

##  Setup & Execution
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/hasratmd697/Asymmetric-Cost-Optimization-in-Medical-AI.git
   cd Asymmetric-Cost-Optimization-in-Medical-AI
   ```
2. **Install Dependencies:**
   ```bash
   pip install torch torchvision pandas numpy scikit-learn pillow tqdm
   ```
3. **Download & Structure Data:** Place the competition images and CSVs inside a `./data` folder in the root directory.
4. **Run Training:**
   ```bash
   python train.py
   ```
