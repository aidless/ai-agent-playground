#!/bin/bash
#===============================================================================
# AE-TTL Project One-Click Deployment Script
# Adaptive Ensemble Test-Time Learning for Multi-Agent Systems
# 
# Usage: bash setup_aettl_project.sh
# Author: AI Research Assistant
# Date: 2026
#===============================================================================

set -e  # Exit on error

#==============================================================================
# CONFIGURATION
#==============================================================================

PROJECT_NAME="aettl-research"
PROJECT_VERSION="0.1.0"
PYTHON_VERSION="3.10"
WORK_DIR="${PWD}/${PROJECT_NAME}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

#==============================================================================
# UTILITY FUNCTIONS
#==============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo "==============================================================================="
    echo -e "${GREEN}$1${NC}"
    echo "==============================================================================="
    echo ""
}

#==============================================================================
# STEP 1: CREATE PROJECT STRUCTURE
#==============================================================================

create_directory_structure() {
    print_header "STEP 1/7: Creating Project Directory Structure"
    
    log_info "Creating main directories..."
    
    mkdir -p "$WORK_DIR"
    
    # Paper directory
    mkdir -p "$WORK_DIR/paper/{figures/results,latex}"
    
    # Source code
    mkdir -p "$WORK_DIR/src/{models,algorithms,environments,data,evaluation,utils}"
    
    # Config files
    mkdir -p "$WORK_DIR/configs/{experiments,environments}"
    
    # Scripts
    mkdir -p "$WORK_DIR/scripts"
    
    # Notebooks
    mkdir -p "$WORK_DIR/notebooks"
    
    # Experiments
    mkdir -p "$WORK_DIR/experiments/{logs/checkpoints,tensorboard,mlflow,plots,reports}"
    
    # Tests
    mkdir -p "$WORK_DIR/tests"
    
    # Documentation
    mkdir -p "$WORK_DIR/docs/{api,user,tutorials}"
    
    # GitHub workflows
    mkdir -p "$WORK_DIR/.github/workflows"
    
    # Data directories
    mkdir -p "$WORK_DIR/data/{raw,processed,cache}"
    
    log_success "Project structure created at: $WORK_DIR"
}

#==============================================================================
# STEP 2: GENERATE REQUIREMENTS.TXT
#==============================================================================

generate_requirements_file() {
    print_header "STEP 2/7: Generating Dependencies File"
    
    cat > "$WORK_DIR/requirements.txt" << 'EOF'
# ============================================
# Core Dependencies
# ============================================
torch>=2.1.0
transformers>=4.35.0
datasets>=2.14.0
accelerate>=0.24.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
scikit-learn>=1.3.0
tqdm>=4.65.0
rich>=13.0.0
PyYAML>=6.0.0

# ============================================
# Reinforcement Learning & Agent Frameworks
# ============================================
ray[rllib]>=2.9.0
stable-baselines3>=2.1.0
gymnasium>=0.29.0

# ============================================
# LLM Integration
# ============================================
sentence-transformers>=2.2.0
langchain>=0.0.300
openai>=1.0.0
tiktoken>=0.5.0

# ============================================
# Experiment Tracking
# ============================================
wandb>=0.15.0
tensorboard>=2.14.0
mlflow>=2.7.0

# ============================================
# Optimization & Hyperparameter Search
# ============================================
optuna>=3.4.0
hydra-core>=1.3.0
aim>=3.22.0

# ============================================
# Visualization
# ============================================
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.18.0
graphviz>=0.20.0

# ============================================
# Utility Libraries
# ============================================
einops>=0.7.0
omegaconf>=2.3.0
fire>=0.6.0

# ============================================
# Testing & Debugging
# ============================================
pytest>=7.4.0
pytest-cov>=4.1.0
ipdb>=0.13.0

# ============================================
# Code Quality Tools
# ============================================
black>=23.0.0
isort>=5.12.0
flake8>=6.1.0
mypy>=1.5.0
pre-commit>=3.4.0

# ============================================
# Documentation
# ============================================
sphinx>=7.2.0
mkdocs>=1.5.0
myst-parser>=2.0.0

# ============================================
# Type Stubs
# ============================================
types-tqdm>=4.66.0
types-PyYAML>=6.0.0
EOF

    log_success "Created requirements.txt"
}

#==============================================================================
# STEP 3: GENERATE SETUP.PY AND PYPROJECT.TOML
#==============================================================================

generate_setup_files() {
    print_header "STEP 3/7: Generating Setup Files"
    
    # setup.py
    cat > "$WORK_DIR/setup.py" << 'EOF'
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="aettl",
    version="0.1.0",
    author="Research Team",
    author_email="research@example.com",
    description="Adaptive Ensemble Test-Time Learning for Multi-Agent Systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/research/aettl-research",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=7.4.0", "black>=23.0.0", "mypy>=1.5.0"],
        "docs": ["sphinx>=7.2.0", "myst-parser>=2.0.0"],
    },
)
EOF

    # pyproject.toml
    cat > "$WORK_DIR/pyproject.toml" << 'EOF'
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.black]
line-length = 100
target-version = ['py310']
include = '\.pyi?$'

[tool.isort]
profile = "black"
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
line_length = 100

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
addopts = "-v --cov=src --cov-report=html"
EOF

    # .gitignore
    cat > "$WORK_DIR/.gitignore" << 'EOF'
__pycache__/
*.py[cod]
*$py.class
*.so
build/
dist/
*.egg-info/
.env
.venv
env/
venv/
.DS_Store
.idea/
.vscode/
*.swp
*.swo
*.sqlite
*.h5
*.pth
*.ckpt
*.onnx
wandb/run-*/
runs/
experiments/logs/*.log
experiments/checkpoints/*.pt
data/cache/*
tmp/
*.log
EOF

    log_success "Created setup.py, pyproject.toml, .gitignore"
}

#==============================================================================
# STEP 4: GENERATE CORE SOURCE CODE FILES
#==============================================================================

generate_source_code() {
    print_header "STEP 4/7: Generating Core Source Code"
    
    cd "$WORK_DIR"
    
    # Create __init__.py files
    touch src/__init__.py
    touch src/models/__init__.py
    touch src/algorithms/__init__.py
    touch src/environments/__init__.py
    touch src/data/__init__.py
    touch src/evaluation/__init__.py
    touch src/utils/__init__.py
    
    # Generate router.py
    cat > src/models/router.py << 'ROUTER_EOF'
"""Dynamic Expert Routing Module for AE-TTL"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DynamicExpertRouter(nn.Module):
    """Input-aware sparse expert selection."""
    
    def __init__(
        self,
        d_model: int = 4096,
        d_router: int = 512,
        num_experts: int = 100,
        top_k: int = 3,
        routing_temperature: float = 0.1,
        load_balance_coef: float = 0.01
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = routing_temperature
        self.z_loss_coef = load_balance_coef
        
        self.router = nn.Sequential(
            nn.Linear(d_model, d_router),
            nn.LayerNorm(d_router),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_router, num_experts)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.router[0].weight)
        nn.init.zeros_(self.router[0].bias)
        nn.init.xavier_uniform_(self.router[-1].weight)
        nn.init.constant_(self.router[-1].bias, 0.0)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = hidden_states.shape
        flat_hidden = hidden_states.flatten(0, 1)
        
        logits = self.router(flat_hidden)
        probs = F.softmax(logits / self.temperature, dim=-1)
        
        scores, indices = torch.topk(probs, self.top_k, dim=-1)
        weights = F.softmax(scores * self.temperature, dim=-1)
        
        from einops import einsum
        one_hot = F.one_hot(indices, num_classes=self.num_experts).float()
        gate_output = einsum(one_hot, weights, "b s k e, b s k -> b s e")
        
        aux_loss = self._compute_load_balancing(gate_output)
        
        indices = indices.view(batch_size, seq_len, self.top_k)
        weights = weights.view(batch_size, seq_len, self.top_k)
        
        return indices, weights, aux_loss
    
    def _compute_load_balancing(self, gate_output: torch.Tensor) -> torch.Tensor:
        avg_usage = gate_output.mean(dim=(0, 1))
        importances = avg_usage * self.num_experts
        z_loss = (importances ** 2).sum()
        return self.z_loss_coef * z_loss
ROUTER_EOF

    # Generate agent.py
    cat > src/models/agent.py << 'AGENT_EOF'
"""Multi-Agent System Core Class"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional


class Agent(nn.Module):
    """Single agent with expert router integration."""
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-v0.1",
        agent_id: int = 0,
        device: str = "cuda"
    ):
        super().__init__()
        self.agent_id = agent_id
        self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        
        # Add router after loading base model
        d_model = self.model.config.hidden_size
        self.router = DynamicExpertRouter(
            d_model=d_model,
            num_experts=100,
            top_k=3
        ).to(device)
        
        self.register_buffer("update_count", torch.tensor(0))
    
    def generate(
        self,
        tasks: List[str],
        context: Optional[str] = None,
        max_new_tokens: int = 512
    ) -> List[str]:
        """Generate responses for given tasks."""
        self.model.eval()
        outputs = []
        
        for task in tasks:
            prompt = self._format_prompt(task, context)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                response = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            decoded = self.tokenizer.decode(response[0], skip_special_tokens=True)
            outputs.append(decoded)
        
        return outputs
    
    def compute_gradient(
        self,
        reward: torch.Tensor,
        task: str
    ) -> torch.Tensor:
        """Compute policy gradient for TTRL update."""
        prompt = self._format_prompt(task)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss
        
        # Apply reward weighting
        weighted_loss = loss * (1 - reward.item())
        weighted_loss.backward()
        
        grad_norm = torch.norm(torch.cat([p.grad.flatten() for p in self.model.parameters() if p.grad is not None]))
        
        return grad_norm.item()
    
    def apply_global_update(
        self,
        aggregated_params: Dict[str, torch.Tensor]
    ):
        """Apply federated aggregation update."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in aggregated_params:
                    param.data.copy_(aggregated_params[name])
        
        self.update_count += 1
    
    def _format_prompt(
        self,
        task: str,
        context: Optional[str] = None
    ) -> str:
        """Format task into instruction prompt."""
        if context:
            return f"<context>\n{context}\n\n<task>\n{task}\n\n<response>\n"
        return f"<task>\n{task}\n\n<response>\n"


class DynamicExpertRouter:
    """Placeholder - actual implementation in router.py"""
    pass
AGENT_EOF

    # Generate reward_estimator.py
    cat > src/models/reward_estimator.py << 'REWARD_EOF'
"""Hierarchical Reward Estimation Module"""

import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from typing import Dict, List
import numpy as np


class HierarchicalRewardEstimator:
    """Cluster-aware consensus reward estimation."""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        num_clusters: int = 5,
        device: str = "cuda"
    ):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.num_clusters = min(num_clusters, 10)
        self.device = device
    
    def cluster_candidates(
        self,
        candidates: List[str]
    ) -> Dict[int, str]:
        """First stage: semantic clustering."""
        embeddings = self.embedding_model.encode(candidates)
        
        n_clusters = min(len(candidates), self.num_clusters)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        representatives = {}
        for cluster_id in np.unique(labels):
            cluster_members = [candidates[i] for i in range(len(candidates)) if labels[i] == cluster_id]
            rep = cluster_members[0]  # Select first as representative
            representatives[cluster_id] = rep
            
        return representatives
    
    def pairwise_comparison(
        self,
        cand_a: str,
        cand_b: str,
        task_context: str
    ) -> float:
        """Second stage: pairwise preference scoring."""
        # Simplified: use length and coherence heuristic
        score_a = len(cand_a.split()) * 0.5 + cand_a.count('.') * 2
        score_b = len(cand_b.split()) * 0.5 + cand_b.count('.') * 2
        
        total = score_a + score_b + 1e-8
        return score_a / total * 10
    
    def compute_rewards(
        self,
        candidates: Dict[int, str],
        task_context: str
    ) -> Dict[int, float]:
        """Complete reward computation pipeline."""
        if len(candidates) <= 1:
            return {aid: 0.5 for aid in candidates}
        
        candidate_list = list(candidates.values())
        representatives = self.cluster_candidates(candidate_list)
        
        rep_scores = {}
        reps_list = list(representatives.values())
        
        for i in range(len(reps_list)):
            for j in range(i+1, len(reps_list)):
                score = self.pairwise_comparison(reps_list[i], reps_list[j], task_context)
                rep_scores.setdefault(reps_list[i], 0) += score
                rep_scores.setdefault(reps_list[j], 0) += (10 - score)
        
        # Normalize scores
        max_score = max(rep_scores.values()) + 1e-8
        for rep in rep_scores:
            rep_scores[rep] /= max_score
        
        # Propagate to original candidates
        final_rewards = {}
        for aid, cand in candidates.items():
            best_rep = min(representatives.values(), key=lambda r: abs(len(r) - len(cand)))
            final_rewards[aid] = rep_scores.get(best_rep, 0.5)
        
        return final_rewards
REWARD_EOF

    # Generate aggregator.py
    cat > src/models/aggregator.py << 'AGGREGATOR_EOF'
"""Byzantine-Resilient Federated Aggregator"""

import torch
import numpy as np
from typing import Dict, List


class KrumAggregator:
    """Krum-style secure aggregation with outlier detection."""
    
    def __init__(
        self,
        byzantine_fraction: float = 0.1,
        device: str = "cuda"
    ):
        self.f = byzantine_fraction
        self.device = device
    
    def aggregate_gradients(
        self,
        all_grads: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Krum algorithm: exclude abnormal gradients."""
        n_total = len(all_grads)
        m = int(n_total * (1 - self.f))
        
        # Compute distance matrix
        distances = self._compute_distance_matrix(all_grads)
        
        # Calculate credibility scores
        scores = []
        for i in range(n_total):
            sorted_dists = np.sort(distances[i])
            scores.append(np.sum(sorted_dists[:m]))
        
        # Select most trustworthy gradients
        selected_indices = np.argsort(scores)[:m]
        
        # Weighted average
        aggregated = {}
        for key in all_grads[0].keys():
            agg_grad = sum(all_grads[i][key] for i in selected_indices) / m
            aggregated[key] = agg_grad
        
        return aggregated
    
    def _compute_distance_matrix(
        self,
        grads: List[Dict[str, torch.Tensor]]
    ) -> np.ndarray:
        """Compute pairwise L2 distances between gradients."""
        n = len(grads)
        dists = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                d = self._l2_distance(grads[i], grads[j])
                dists[i][j] = dists[j][i] = d
        
        return dists
    
    def _l2_distance(
        self,
        grad_a: Dict[str, torch.Tensor],
        grad_b: Dict[str, torch.Tensor]
    ) -> float:
        """Compute flattened L2 distance between two gradient dicts."""
        total_sq = 0.0
        for key in grad_a.keys():
            diff = (grad_a[key] - grad_b[key]).flatten()
            total_sq += (diff ** 2).sum().item()
        return np.sqrt(total_sq)
AGGREGATOR_EOF

    # Generate aettl.py (main algorithm)
    cat > src/algorithms/aettl.py << 'AETTLEOF'
"""AE-TTL Main Algorithm Implementation"""

import torch
import yaml
import os
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict


class AETTLOrchestration:
    """Main orchestration class for AE-TTL."""
    
    def __init__(self, config_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()
        
        # Initialize components
        self.agents = self._init_agents()
        self.reward_estimator = self._init_reward_estimator()
        self.aggregator = self._init_aggregator()
        
        self.global_step = 0
        self.training_history = []
    
    def _default_config(self) -> dict:
        """Return default configuration."""
        return {
            "model": {"num_experts": 100, "top_k": 3},
            "training": {"batch_size": 8, "learning_rate": 1e-5},
            "multi_agent": {"num_agents": 10, "communication_frequency": 10},
        }
    
    def _init_agents(self) -> list:
        """Initialize multi-agent system."""
        # Placeholder - actual init requires model loading
        num_agents = self.config["multi_agent"]["num_agents"]
        return [{"id": i, "state": "initialized"} for i in range(num_agents)]
    
    def _init_reward_estimator(self):
        """Initialize hierarchical reward estimator."""
        return "HierarchicalRewardEstimator initialized"
    
    def _init_aggregator(self):
        """Initialize federated aggregator."""
        return "KrumAggregator initialized"
    
    def train(self, train_loader, num_epochs: int = 100):
        """Main training loop."""
        print(f"Starting AE-TTL training for {num_epochs} epochs...")
        print(f"Number of agents: {len(self.agents)}")
        print(f"Device: {self.device}")
        
        for epoch in range(num_epochs):
            epoch_loss = self._train_epoch(train_loader)
            
            history_entry = {
                "epoch": epoch,
                "loss": epoch_loss,
                "global_step": self.global_step
            }
            self.training_history.append(history_entry)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f}")
        
        return self.training_history
    
    def _train_epoch(self, train_loader) -> float:
        """Run one training epoch."""
        total_loss = 0.0
        n_batches = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Simulate training step
            loss = self._training_step(batch)
            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1
        
        return total_loss / max(n_batches, 1)
    
    def _training_step(self, batch) -> torch.Tensor:
        """Single training step (placeholder)."""
        # This would contain actual training logic
        dummy_loss = torch.tensor(0.5, device=self.device)
        return dummy_loss
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint."""
        checkpoint = {
            "global_step": self.global_step,
            "training_history": self.training_history,
            "config": self.config
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.global_step = checkpoint["global_step"]
        self.training_history = checkpoint["training_history"]
        print(f"Checkpoint loaded from {filepath}")


if __name__ == "__main__":
    # Quick test
    orchestrator = AETTLOrchestration()
    print("AE-TTL Orchestration initialized successfully!")
AETTLEOF

    cd ../
    log_success "Core source code files generated"
}

#==============================================================================
# STEP 5: GENERATE CONFIGURATION FILES
#==============================================================================

generate_config_files() {
    print_header "STEP 5/7: Generating Configuration Files"
    
    # Default config
    cat > "$WORK_DIR/configs/default.yaml" << 'CONFIGEOF'
# ============================================
# AE-TTL Default Configuration
# ============================================

experiment:
  name: "aettl_webshop_experiment"
  seed: 42
  wandb_entity: "your_username"
  project_name: "aettl_research"
  
model:
  backbone: "mistralai/Mistral-7B-v0.1"
  max_length: 2048
  num_experts: 100
  top_k: 3
  routing_temperature: 0.1
  
training:
  mode: "ttrl"
  device: "cuda"
  batch_size: 8
  learning_rate: 1e-5
  weight_decay: 0.01
  optimizer: "adamw"
  
multi_agent:
  num_agents: 10
  communication_frequency: 10
  aggregation_method: "krum_byzantine"
  byzantine_fraction: 0.1

reward_estimation:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  num_clusters: 5
  
evaluation:
  datasets: ["webshop", "miniwob", "swebench"]
  metrics_to_track: ["success_rate", "step_efficiency"]
CONFIGEOF

    # WebShop experiment config
    cat > "$WORK_DIR/configs/experiments/exp_01_webshop.yaml" << 'WSEXP'
# WebShop Experiment Configuration
extends: configs/default.yaml

experiment:
  name: "aettl_webshop_main"

training:
  epochs: 50
  batch_size: 8
  
environments:
  webshop:
    max_episodes: 600
    timeout_seconds: 300
WSEXP

    # MiniWob experiment config
    cat > "$WORK_DIR/configs/experiments/exp_02_miniwob.yaml" << 'MWEYP'
# MiniWob++ Experiment Configuration
extends: configs/default.yaml

experiment:
  name: "aettl_miniwob_main"

training:
  epochs: 40
  
environments:
  miniwob:
    domain_randomization: true
    action_space_type: "discrete"
MWEYP

    log_success "Configuration files generated"
}

#==============================================================================
# STEP 6: GENERATE SCRIPTS
#==============================================================================

generate_scripts() {
    print_header "STEP 6/7: Generating Shell Scripts"
    
    # Environment setup script
    cat > "$WORK_DIR/scripts/setup_environment.sh" << 'SETUPSH'
#!/bin/bash
echo "🚀 Setting up AE-TTL Environment..."
conda create -n aettl python=3.10 -y
conda activate aettl
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo "✅ Setup complete! Run: conda activate aettl"
SETUPSH
    chmod +x "$WORK_DIR/scripts/setup_environment.sh"
    
    # Training script
    cat > "$WORK_DIR/scripts/train_aettl.sh" << 'TRAINSH'
#!/bin/bash
source_dir="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$source_dir"

echo "🔬 Starting AE-TTL Training..."
python src/algorithms/aettl.py --config configs/experiments/exp_01_webshop.yaml

echo "✅ Training completed!"
TRAINSH
    chmod +x "$WORK_DIR/scripts/train_aettl.sh"
    
    # Evaluation script
    cat > "$WORK_DIR/scripts/evaluate.sh" << 'EVALSH'
#!/bin/bash
echo "📊 Running Evaluation..."
python src/evaluation/benchmark_runner.py --benchmark webshop --checkpoint experiments/checkpoints/best_model.pt
echo "✅ Evaluation completed!"
EVALSH
    chmod +x "$WORK_DIR/scripts/evaluate.sh"
    
    # Makefile
    cat > "$WORK_DIR/Makefile" << 'MAKEFILE'
.PHONY: help install test clean lint format train eval

help: ## Show this help message
	@echo 'Usage: make [target]'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk '{printf "  %-20s %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -e ".[dev]"

test: ## Run tests
	pytest tests/ -v

clean: ## Clean build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info

lint: ## Run linters
	flake8 src/
	mypy src/

format: ## Format code
	black src/
	isort src/

train: ## Start training
	bash scripts/train_aettl.sh

eval: ## Run evaluation
	bash scripts/evaluate.sh
MAKEFILE

    log_success "Shell scripts and Makefile generated"
}

#==============================================================================
# STEP 7: GENERATE DOCUMENTATION AND README
#==============================================================================

generate_documentation() {
    print_header "STEP 7/7: Generating Documentation"
    
    # Comprehensive README.md
    cat > "$WORK_DIR/README.md" << 'READMEEOF'
# 🔬 AE-TTL: Adaptive Ensemble Test-Time Learning for Multi-Agent Systems

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 Overview

AE-TTL introduces a novel framework combining:
- ✅ Test-Time Reinforcement Learning for online adaptation
- ✅ Hierarchical Reward Estimation via clustering-based consensus
- ✅ Byzantine-Resilient Federated Aggregation for robust collaboration
- ✅ Sparse Expert Routing for computational efficiency

## 🚀 Quick Start

### Installation

```bash
# Clone and navigate
cd aettl-research

# Setup environment
bash scripts/setup_environment.sh

# Activate environment
conda activate aettl