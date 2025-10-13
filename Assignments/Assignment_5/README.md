# MLOps Assignment 5 - Complete Beginner's Guide
# Named Entity Recognition (NER) & Weak Supervision with Snorkel AI

## 📋 Assignment Overview

This assignment explores **weak supervision** and **transfer learning** in machine learning through:
- **Named Entity Recognition (NER)** using the CoNLL-2003 dataset
- **Snorkel AI** for programmatic data labeling
- **CIFAR-10/100** sequential training experiments 
- **Weights & Biases (W&B)** for experiment tracking

### 🎯 Learning Objectives
- Understand weak supervision vs traditional supervised learning
- Learn to create programmatic labeling functions
- Explore transfer learning effects between similar datasets
- Practice MLOps with experiment tracking

---

## 🚀 Complete Setup Guide for Beginners

### Prerequisites
- **Python 3.8+** installed on your system
- **Internet connection** for downloading datasets (~2GB)
- **Weights & Biases account** (free at https://wandb.ai/)
- **Basic command line knowledge**

### Step 1: Navigate to Assignment Folder
```powershell
# Open PowerShell and navigate to assignment directory
cd "c:\Users\BHUPATHI VARUN\Desktop\SEM_7\MLOPS\mlops2025w_142201034\Assignments\Assignment_5"

# Verify you're in the right directory
ls
# You should see: src/, pyproject.toml, README.md, etc.
```

### Step 2: Create Python Virtual Environment
```powershell
# Create isolated Python environment (recommended)
python -m venv .venv

# Activate the environment (IMPORTANT: Do this every time)
.\.venv\Scripts\Activate.ps1

# Your prompt should now show (.venv) at the beginning
# If you get execution policy error, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 3: Install Dependencies
```powershell
# Install all required packages from pyproject.toml
pip install -e .

# This installs:
# - datasets (HuggingFace datasets library)
# - torch & torchvision (PyTorch for deep learning)
# - transformers (NLP models)
# - wandb (experiment tracking)
# - snorkel (weak supervision framework)
# - numpy, pandas, scikit-learn (data science)
# - matplotlib, seaborn (visualization)
```

### Step 4: Setup Weights & Biases
```powershell
# Login to W&B (one-time setup)
wandb login

# This opens browser → copy your API key → paste in terminal
# Get API key from: https://wandb.ai/authorize
```

### Step 5: Verify Installation
```powershell
# Test all imports work correctly
python -c "import wandb, snorkel, datasets, torch, torchvision; print('✅ All packages installed successfully!')"
```

---

## 📁 Directory Structure Explained

```
Assignment_5/                           # Main assignment folder
├── 📄 README.md                       # This documentation file
├── 📄 pyproject.toml                  # Python dependencies & project config
├── 📄 Assignment 5.pdf                # Original assignment instructions
├── 📄 .gitignore                      # Files ignored by Git (see below)
├── 📄 uv.lock                         # Dependency lock file
│
├── 📁 src/                            # Source code directory
│   ├── 🐍 question1_conll_ner.py     # Q1: Dataset analysis & W&B logging
│   ├── 🐍 question2_snorkel_labeling.py  # Q2: Labeling functions
│   ├── 🐍 question3_label_aggregation.py # Q3: Label voting & aggregation  
│   └── 🐍 question4_cifar_experiments.py # Q4: Transfer learning experiments
│
├── 📁 .venv/                          # Python virtual environment (created by you)
├── 📁 data/                           # Downloaded datasets (auto-created)
│   ├── 📁 cifar-10-batches-py/        # CIFAR-10 dataset files
│   └── 📁 cifar-100-python/           # CIFAR-100 dataset files
│
├── 📁 outputs/                        # Generated results & models
│   ├── 📊 *.png                      # Visualization plots
│   ├── 💾 *.pkl                      # Serialized Python objects
│   ├── 📋 *.npy                      # NumPy arrays
│   └── 🧠 *.pth                      # PyTorch model weights
│
└── 📁 wandb/                          # W&B experiment logs (auto-created)
    └── 📁 run-*/                     # Individual experiment runs
```

### 🚫 Files Ignored by Git (.gitignore)
The following files/folders are **NOT tracked** in version control:
- **`.venv/`** - Virtual environment (too large, user-specific)
- **`data/`** - Datasets (downloaded automatically, very large)
- **`outputs/`** - Generated results (can be recreated)
- **`wandb/`** - W&B logs (stored in cloud)
- **`__pycache__/`** - Python bytecode (auto-generated)
- **`*.pyc`** - Compiled Python files
- **`.ipynb_checkpoints/`** - Jupyter notebook checkpoints
- **`.DS_Store`** - macOS system files
- **`.vscode/`** - VS Code settings
- **`.env`** - Environment variables (may contain secrets)

---

## 📚 Question-by-Question Breakdown

### 🔍 Question 1: CoNLL-2003 Dataset Analysis
**File**: `src/question1_conll_ner.py`

#### What It Does:
- Loads the famous **CoNLL-2003 Named Entity Recognition** dataset
- Analyzes entity distribution and dataset statistics
- Logs all metrics to **Weights & Biases** for tracking

#### Why It's Important:
- **CoNLL-2003** is the gold standard benchmark for NER tasks
- Understanding data distribution is crucial before building models
- **W&B logging** enables experiment reproducibility

#### How It Works:
1. **Downloads dataset** from HuggingFace (automatic)
2. **Parses BIO tagging** (B-eginning, I-nside, O-utside entity tags)
3. **Counts entity types**: PER (Person), LOC (Location), ORG (Organization), MISC (Miscellaneous)
4. **Creates visualizations** of entity distributions
5. **Logs to W&B** with project name "Q1-weak-supervision-ner"

#### Entity Types Explained:
- **PER**: Person names → "Barack Obama", "John Smith"
- **LOC**: Geographic locations → "New York", "Paris", "Mount Everest"  
- **ORG**: Organizations → "Google", "United Nations", "Harvard University"
- **MISC**: Miscellaneous entities → "Olympics", "Nobel Prize", "World War II"

#### Run Command:
```powershell
python src/question1_conll_ner.py
```

---

### 🏷️ Question 2: Snorkel Labeling Functions
**File**: `src/question2_snorkel_labeling.py`

#### What It Does:
- Implements **programmatic labeling functions** using Snorkel AI
- Creates two heuristic rules to automatically label text data
- Evaluates labeling function performance and coverage

#### Why It's Revolutionary:
- **Traditional approach**: Humans manually label thousands of examples (expensive, slow)
- **Weak supervision**: Write code that labels data automatically (fast, scalable)
- **Domain expertise**: Capture human knowledge as programmatic rules

#### The Two Labeling Functions:

1. **Year Detection Function**:
   - **Purpose**: Identify years (1900-2099) as potential DATE entities
   - **Logic**: Uses regex pattern `\b(19|20)\d{2}\b`
   - **Examples**: "2023" → MISC, "1995" → MISC, "hello" → ABSTAIN

2. **Organization Detection Function**:
   - **Purpose**: Identify companies by common suffixes  
   - **Logic**: Looks for "Inc.", "Corp.", "Ltd.", "LLC", "Co."
   - **Examples**: "Apple Inc." → MISC, "random word" → NOT_ENTITY

#### How Weak Supervision Works:
```
Text: "Apple Inc. was founded in 1976"
├── Year LF: Finds "1976" → Label: MISC
├── Org LF: Finds "Apple Inc." → Label: MISC  
└── Combined: Multiple signals → Higher confidence
```

#### Output Files:
- **`label_matrix_Q2.npy`**: Matrix of all labeling function outputs
- **`dataframe_Q2.pkl`**: Processed text data with labels
- **`question2_labeling_functions.png`**: Performance visualization

#### Run Command:
```powershell
python src/question2_snorkel_labeling.py
```

---

### 🗳️ Question 3: Label Aggregation with Majority Voting
**File**: `src/question3_label_aggregation.py`

#### What It Does:
- Combines multiple noisy labeling functions into clean training labels
- Implements **Majority Label Voter** and **Snorkel Label Model**
- Compares different aggregation strategies

#### The Challenge:
- Multiple labeling functions may **disagree** on the same text
- Some functions are more **accurate** than others
- Some functions may be **correlated** (not independent)

#### Two Aggregation Methods:

1. **Majority Label Voter** (Simple):
   - **Logic**: Most common label wins
   - **Pros**: Fast, interpretable
   - **Cons**: Treats all functions equally

2. **Snorkel Label Model** (Sophisticated):
   - **Logic**: Learns function accuracies and correlations automatically
   - **Pros**: Weights reliable functions higher
   - **Cons**: More complex, requires training

#### Example Aggregation:
```
Text: "Microsoft Corp. established in 1975"

Function Votes:
├── Year LF: "1975" → MISC
├── Org LF: "Microsoft Corp." → MISC
└── Random LF: → NOT_ENTITY

Majority Vote: MISC (2 vs 1)
Label Model: MISC (higher confidence due to function reliability)
```

#### Output Files:
- **`final_predictions_Q3.npy`**: Hard predictions (0 or 1)
- **`final_probabilities_Q3.npy`**: Soft probabilities (0.0 to 1.0)
- **`final_aggregated_labels_Q3.pkl`**: Complete aggregation results
- **`question3_label_aggregation.png`**: Comparison visualization

#### Run Command:
```powershell
python src/question3_label_aggregation.py
```

---

### 🧠 Question 4: CIFAR Sequential Training Experiments  
**File**: `src/question4_cifar_experiments.py`

#### What It Does:
- Studies **transfer learning** effects between CIFAR-10 and CIFAR-100
- Trains models sequentially: Dataset A → Dataset B
- Measures **catastrophic forgetting** and adaptation

#### The Datasets:
- **CIFAR-10**: 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **CIFAR-100**: 100 classes organized into 20 superclasses (more challenging)
- **Both**: 32×32 RGB images, 50K training + 10K test samples

#### Two Experiments:

1. **Experiment A**: CIFAR-100 → CIFAR-10
   - Train on complex dataset first (100 classes)
   - Then adapt to simpler dataset (10 classes)
   - **Hypothesis**: Should be easier (going from hard to easy)

2. **Experiment B**: CIFAR-10 → CIFAR-100  
   - Train on simple dataset first (10 classes)
   - Then adapt to complex dataset (100 classes)
   - **Hypothesis**: Should be harder (going from easy to hard)

#### Key ML Concepts:

**Transfer Learning**:
- Use pretrained **ResNet18** (trained on ImageNet)
- Adapt to smaller CIFAR images (32×32 vs 224×224)
- Fine-tune features for new tasks

**Catastrophic Forgetting**:
- When learning new task, model "forgets" previous task
- **Natural phenomenon** in neural networks
- Measured by testing on original dataset after new training

#### Architecture Details:
```
ResNet18 Backbone (Pretrained)
├── Feature Extractor: Conv layers (shared)
├── Classifier A: FC layer for CIFAR-100 (100 classes)
└── Classifier B: FC layer for CIFAR-10 (10 classes)
```

#### Training Strategy:
- **Phase 1**: Train backbone + classifier A (100 epochs)
- **Phase 2**: Keep backbone, train classifier B (100 epochs)
- **Evaluation**: Test both classifiers to measure forgetting

#### Output Files:
- **Model weights**: `*.pth` files for each phase
- **W&B logs**: Training curves, accuracy metrics, loss plots
- **Experiments**: Compare A→B vs B→A performance

#### Run Command:
```powershell
python src/question4_cifar_experiments.py
# Note: Takes ~2-4 hours depending on hardware (GPU recommended but not required)
```

---

## 🛠️ Running All Questions (Complete Workflow)

### Option 1: Run Each Question Separately (Recommended)
```powershell
# Make sure virtual environment is active
.\.venv\Scripts\Activate.ps1

# Q1: Dataset analysis (5-10 minutes)
python src/question1_conll_ner.py

# Q2: Labeling functions (10-15 minutes) 
python src/question2_snorkel_labeling.py

# Q3: Label aggregation (5-10 minutes)
python src/question3_label_aggregation.py

# Q4: CIFAR experiments (2-4 hours!)
python src/question4_cifar_experiments.py
```

### Option 2: Run All Together
```powershell
# Run complete pipeline (will take several hours)
cd src
python question1_conll_ner.py && python question2_snorkel_labeling.py && python question3_label_aggregation.py && python question4_cifar_experiments.py
```

---

## 📊 Understanding the Outputs

### Generated Files After Running:
```
outputs/
├── 📊 question2_labeling_functions.png     # LF performance plots
├── 📊 question3_label_aggregation.png      # Aggregation comparison
├── 💾 dataframe_Q2.pkl                    # Processed text data
├── 📋 label_matrix_Q2.npy                 # LF output matrix
├── 📋 final_predictions_Q3.npy            # Hard labels
├── 📋 final_probabilities_Q3.npy          # Soft probabilities
├── 💾 final_aggregated_labels_Q3.pkl      # Aggregation results
├── 🧠 best_model_CIFAR10_phase_1.pth      # CIFAR-10 model weights
├── 🧠 best_model_CIFAR100_phase_1.pth     # CIFAR-100 model weights
└── 🧠 model_experiment_*.pth               # Sequential training results
```

### Weights & Biases Dashboard:
- Visit https://wandb.ai/ → Your projects
- **Project**: "Q1-weak-supervision-ner"
- **Metrics**: Accuracy, loss curves, dataset statistics
- **Visualizations**: Confusion matrices, training progress

---

## 🔧 Troubleshooting Guide

### 🚨 Common Issues & Solutions:

#### 1. Virtual Environment Problems
```powershell
# Error: "cannot be loaded because running scripts is disabled"
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Error: "No module named 'wandb'"
# Make sure virtual environment is activated:
.\.venv\Scripts\Activate.ps1
pip install -e .
```

#### 2. Dataset Download Issues
```powershell
# Error: "HTTPSConnectionPool timeout"
# Solution: Check internet connection, try again
# Datasets auto-download on first run (may take time)
```

#### 3. CUDA/GPU Issues  
```powershell
# Error: "CUDA out of memory"
# Solution: Code works on CPU automatically
# Edit question4_cifar_experiments.py → reduce batch_size if needed
```

#### 4. W&B Authentication
```powershell
# Error: "wandb login required" 
wandb login
# Paste API key from: https://wandb.ai/authorize
```

#### 5. Import Errors
```powershell
# Error: "ModuleNotFoundError: No module named 'snorkel'"
pip install -e .
# Reinstall all dependencies
```

---

## 🎓 Key Learning Outcomes

### After Completing This Assignment:

#### Technical Skills:
- ✅ **Weak Supervision**: Create programmatic labeling functions
- ✅ **NER Understanding**: Work with entity recognition datasets  
- ✅ **Transfer Learning**: Train models sequentially across tasks
- ✅ **Experiment Tracking**: Use W&B for MLOps workflows
- ✅ **Deep Learning**: Implement CNN architectures with PyTorch

#### Conceptual Understanding:
- ✅ **Data Efficiency**: Generate labels programmatically vs manual annotation
- ✅ **Model Evaluation**: Measure catastrophic forgetting and adaptation
- ✅ **ML Pipeline**: End-to-end workflow from data to deployment
- ✅ **Best Practices**: Virtual environments, dependency management, version control

#### Real-World Applications:
- **Industry Problem**: Manual labeling costs millions, takes months
- **Snorkel Solution**: Programmatic labeling cuts time by 10-100x
- **Transfer Learning**: Reuse pretrained models for new domains
- **MLOps**: Track experiments, reproduce results, collaborate effectively

---

## 📖 Additional Resources

### Documentation:
- **Snorkel AI**: https://snorkel.org/
- **W&B Docs**: https://docs.wandb.ai/
- **HuggingFace Datasets**: https://huggingface.co/datasets/eriktks/conll2003
- **PyTorch Tutorials**: https://pytorch.org/tutorials/

### Research Papers:
- **Snorkel Paper**: "Snorkel: Rapid Training Data Creation with Weak Supervision"
- **CoNLL-2003**: "Introduction to the CoNLL-2003 Shared Task"
- **Transfer Learning**: "How transferable are features in deep neural networks?"

### Video Resources:
- **W&B Course**: https://www.kdnuggets.com/weights-biases-a-kdnuggets-crash-course
- **Snorkel Tutorial**: https://www.youtube.com/playlist?list=PLZePYakcDhmgkczq3OTRxLbe1BTXRXF4l

---

## 🆘 Getting Help

### If You're Stuck:
1. **Check error messages** carefully - they often contain the solution
2. **Verify virtual environment** is activated (see prompt)  
3. **Check internet connection** for dataset downloads
4. **Read console output** for progress indicators
5. **Check W&B dashboard** for logged metrics
6. **Ask for help** with specific error messages and context

### Success Indicators:
- ✅ All imports work without errors
- ✅ W&B dashboard shows your experiments  
- ✅ Output files generated in `outputs/` folder
- ✅ Training loss decreases over epochs
- ✅ No Python exceptions during execution

**Good luck with your MLOps journey! 🚀**
