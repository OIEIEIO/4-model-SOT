#@title Load All Models to GPU Memory h2o

# ===========================================
# GPU Model Loader - Simultaneous Loading
# Loads all cached models directly to GPU for fast inference
# Updated with Llama-3.2-1B-Instruct replacing TinyLlama
# ===========================================

import torch
import time
import gc
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

print("[gpu-loader] Loading all models to GPU memory...")

# ===========================================
# Device and Memory Setup
# ===========================================

# Check CUDA availability and setup device
if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[gpu-loader] Device: {gpu_name}")
    print(f"[gpu-loader] Total VRAM: {total_memory:.1f} GB")
else:
    device = torch.device("cpu")
    print("[gpu-loader] WARNING: CUDA not available, using CPU")

# Clear any existing GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

# ===========================================
# Model Configuration
# ===========================================

# Local cache directory from previous download
MODEL_CACHE_DIR = Path("models/cache")

# Agent model configurations
AGENT_MODELS = {
    "architect": {
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "role": "architect",
        "color": "\x1b[96m",  # Cyan
        "system_prompt": "You are the Architect agent in a multi-agent drone design team."
    },
    "ops": {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "role": "ops",
        "color": "\x1b[93m",  # Yellow
        "system_prompt": "You are the Operations agent in a multi-agent drone design team."
    },
    "manager": {
        "model_id": "google/gemma-2-2b-it",
        "role": "manager",
        "color": "\x1b[92m",  # Green
        "system_prompt": "You are the Manager agent in a multi-agent drone design team."
    },
    "reasoning": {
        "model_id": "h2oai/h2o-danube2-1.8b-base",
        "role": "reasoning_specialist",
        "color": "\x1b[95m",  # Magenta
        "system_prompt": "You are the Reasoning Specialist agent in a multi-agent drone design team."
    }
}

# Embedding model configuration
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# ===========================================
# Model Loading Functions
# ===========================================

def load_language_model(model_id, agent_name):
    """Load a language model and tokenizer to GPU"""
    print(f"[gpu-loader] Loading {agent_name} model: {model_id}")
    start_time = time.time()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=MODEL_CACHE_DIR,
        trust_remote_code=True
    )

    # Set pad token if missing
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model with appropriate dtype
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=MODEL_CACHE_DIR,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=None  # We'll manually move to device
    )

    # Move to GPU
    model = model.to(device)
    model.eval()

    load_time = time.time() - start_time

    # Check model memory usage
    if torch.cuda.is_available():
        model_memory = torch.cuda.memory_allocated() / 1e9
        print(f"[gpu-loader] ✓ {agent_name} loaded in {load_time:.1f}s | GPU memory: {model_memory:.1f} GB")
    else:
        print(f"[gpu-loader] ✓ {agent_name} loaded in {load_time:.1f}s")

    return tokenizer, model

def load_embedding_model():
    """Load sentence transformer model"""
    print(f"[gpu-loader] Loading embedding model: {EMBEDDING_MODEL_ID}")
    start_time = time.time()

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_ID,
        cache_folder=MODEL_CACHE_DIR,
        device=device
    )

    load_time = time.time() - start_time
    print(f"[gpu-loader] ✓ Embedding model loaded in {load_time:.1f}s")

    return embedding_model

# ===========================================
# Load All Models Simultaneously
# ===========================================

print(f"[gpu-loader] Starting simultaneous model loading...")
print(f"[gpu-loader] Updated: Using Llama-3.2-1B for better instruction following")
total_start_time = time.time()

# Dictionary to store loaded models
loaded_models = {}

# Load each agent's language model
for agent_name, config in AGENT_MODELS.items():
    model_id = config["model_id"]
    tokenizer, model = load_language_model(model_id, agent_name)

    loaded_models[agent_name] = {
        "tokenizer": tokenizer,
        "model": model,
        "model_id": model_id,
        "config": config
    }

# Load embedding model
embedding_model = load_embedding_model()
loaded_models["embedding"] = embedding_model

total_load_time = time.time() - total_start_time

# ===========================================
# Memory Usage Report
# ===========================================

print(f"\n{'='*60}")
print("GPU MODEL LOADING COMPLETE")
print(f"{'='*60}")

if torch.cuda.is_available():
    current_memory = torch.cuda.memory_allocated() / 1e9
    max_memory = torch.cuda.max_memory_allocated() / 1e9
    memory_percent = (current_memory / total_memory) * 100

    print(f"GPU Memory Usage:")
    print(f"  Current: {current_memory:.2f} GB / {total_memory:.1f} GB ({memory_percent:.1f}%)")
    print(f"  Peak: {max_memory:.2f} GB")
    print(f"  Available: {total_memory - current_memory:.2f} GB")

print(f"\nLoaded Models:")
for agent_name, model_data in loaded_models.items():
    if agent_name != "embedding":
        model_id = model_data["model_id"]
        print(f"  {agent_name.title()}: {model_id}")
    else:
        print(f"  Embedding: {EMBEDDING_MODEL_ID}")

print(f"\nTotal Loading Time: {total_load_time:.1f} seconds")
print(f"Models ready for fast inference!")

# ===========================================
# Model Verification Test
# ===========================================

print(f"\n[gpu-loader] Running verification tests...")

def test_model_inference(agent_name, model_data):
    """Quick inference test to verify model works"""
    try:
        tokenizer = model_data["tokenizer"]
        model = model_data["model"]

        # Simple test prompt
        test_prompt = "Hello, I am the"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"[gpu-loader] ✓ {agent_name.title()} inference test passed")
        return True

    except Exception as e:
        print(f"[gpu-loader] ✗ {agent_name.title()} inference test failed: {e}")
        return False

# Test each language model
all_tests_passed = True
for agent_name, model_data in loaded_models.items():
    if agent_name != "embedding":
        test_passed = test_model_inference(agent_name, model_data)
        all_tests_passed = all_tests_passed and test_passed

# Test embedding model
try:
    test_embedding = loaded_models["embedding"].encode(["Test sentence"], normalize_embeddings=True)
    print(f"[gpu-loader] ✓ Embedding model inference test passed")
except Exception as e:
    print(f"[gpu-loader] ✗ Embedding model inference test failed: {e}")
    all_tests_passed = False

if all_tests_passed:
    print(f"\n[gpu-loader] All models verified and ready for multi-agent system!")
    print(f"[gpu-loader] Llama-3.2-1B should provide much better conversation quality")
else:
    print(f"\n[gpu-loader] WARNING: Some models failed verification tests")

# ===========================================
# Global Variables for Main System
# ===========================================

# Make models available globally for the main system
MODEL_CACHE = loaded_models
EMBEDDING_MODEL = embedding_model
DEVICE = device

print(f"\n[gpu-loader] Global variables set: MODEL_CACHE, EMBEDDING_MODEL, DEVICE")
print(f"[gpu-loader] MODEL_CACHE keys: {list(MODEL_CACHE.keys())}")
print(f"[gpu-loader] Ready for main multi-agent system execution!")

# ===========================================
# Model Upgrade Summary
# ===========================================

print(f"\n{'='*60}")
print("MODEL CONFIGURATION UPDATE")
print(f"{'='*60}")
print(f"OPS MODEL UPDATED:")
print(f"  OLD: TinyLlama/TinyLlama-1.1B-Chat-v1.0")
print(f"  NEW: meta-llama/Llama-3.2-1B-Instruct")
print(f"")
print(f"Expected improvements:")
print(f"  ✓ Better instruction following")
print(f"  ✓ Reduced template repetition")
print(f"  ✓ More natural multi-agent dialogue")
print(f"  ✓ Better reasoning capability")

#@title Just the run

# ===========================================
# Filename: src/multi_agent_core.py
# Tree Location: /project_root/src/multi_agent_core.py
# Version: 10.0
# Description: Core infrastructure with metabolic constraint wallet system
# Dependencies: transformers, sentence-transformers, torch, numpy, peft
# ===========================================

import json
import csv
import time
import uuid
import re
import threading
import textwrap
import math
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Set, Any
from transformers import TextIteratorStreamer, GenerationConfig
from sklearn.metrics.pairwise import cosine_similarity

# ===========================================
# CONFIGURATION LOADER
# ===========================================

def load_config(config_path: str = "config.json") -> Dict:
    """Load configuration from JSON file with validation"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        print(f"📋 Loaded configuration from {config_path}")
        print(f"🔧 Config version: {config.get('metadata', {}).get('config_version', 'unknown')}")
        print(f"🎯 Project: {config.get('metadata', {}).get('project_name', 'unknown')}")

        return config
    except FileNotFoundError:
        print(f"❌ ERROR: Configuration file {config_path} not found!")
        raise RuntimeError(f"Configuration file {config_path} must exist")
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON in {config_path}: {e}")
        raise RuntimeError(f"Configuration file {config_path} contains invalid JSON")

# Load global configuration
CONFIG = load_config()

print("🚀 Starting Enhanced Multi-Agent System v10.0 with Metabolic Constraints...")
print("✨ Features: Manager-Led Flow | Proportional Stages | Stage Cycling | Wallet System | Smart Routing")

# ===========================================
# Prerequisites Check
# ===========================================

try:
    print(f"🔍 Checking pre-loaded models...")
    print(f"📦 Available models: {list(MODEL_CACHE.keys())}")
    print(f"🎯 Device: {DEVICE}")
    print(f"🧠 Embedding model: {type(EMBEDDING_MODEL).__name__}")
except NameError:
    print("❌ ERROR: Models not loaded. Run GPU loader block first!")
    raise RuntimeError("GPU loader block must be executed first")

# ===========================================
# Enhanced Colors and Visual Elements
# ===========================================

class Colors:
    # Agent colors
    CYAN = "\x1b[96m"      # Technical
    YELLOW = "\x1b[93m"    # Practical
    GREEN = "\x1b[92m"     # Strategic
    MAGENTA = "\x1b[95m"   # Reasoning

    # Status colors
    BLUE = "\x1b[94m"      # System
    RED = "\x1b[91m"       # Errors
    WHITE = "\x1b[97m"     # Highlights

    # Formatting
    RESET = "\x1b[0m"
    BOLD = "\x1b[1m"
    DIM = "\x1b[2m"

    # Visual elements
    CHECKMARK = "✓"
    CROSSMARK = "✗"
    ARROW = "→"
    BULLET = "•"
    CHECKBOX_EMPTY = "☐"
    CHECKBOX_FILLED = "☑"

# Map config color names to ANSI codes
COLOR_MAP = {
    "cyan": Colors.CYAN,
    "yellow": Colors.YELLOW,
    "green": Colors.GREEN,
    "magenta": Colors.MAGENTA
}

# Build model colors from config
MODEL_COLORS = {}
for pool_name, color_name in CONFIG.get("display", {}).get("colors", {}).items():
    model_key = CONFIG.get("models", {}).get("model_pools", {}).get(pool_name, "architect")
    MODEL_COLORS[model_key] = COLOR_MAP.get(color_name, Colors.CYAN)

# Ensure reasoning color is set
MODEL_COLORS["reasoning"] = Colors.MAGENTA

def wrap_text(text: str, width: int = None) -> str:
    """Wrap text for better readability"""
    if width is None:
        width = CONFIG.get("display", {}).get("line_width", 80)

    lines = text.split('\n')
    wrapped_lines = []

    for line in lines:
        if len(line) <= width:
            wrapped_lines.append(line)
        else:
            wrapped_lines.extend(textwrap.wrap(line, width=width,
                                             break_long_words=False,
                                             break_on_hyphens=False))

    return '\n'.join(wrapped_lines)

def format_header(title: str, char: str = "=", width: int = 70) -> str:
    """Create formatted headers"""
    padding = (width - len(title) - 2) // 2
    return f"{char * padding} {title} {char * padding}"

# ===========================================
# Metabolic Constraint Wallet System
# ===========================================

class AgentWallet:
    """Individual agent wallet with metabolic constraints"""

    def __init__(self, agent_id: str, starting_balance: float = 100.0):
        self.agent_id = agent_id
        self.balance = starting_balance
        self.starting_balance = starting_balance
        self.last_decay = time.time()

        # Get wallet config from main config
        wallet_config = CONFIG.get("wallet", {})
        self.decay_rate = wallet_config.get("decay_rate", 0.1)  # 10% per minute
        self.min_balance = wallet_config.get("min_balance", 5.0)
        self.max_balance = wallet_config.get("max_balance", 500.0)

        # Transaction tracking
        self.transaction_log = []
        self.total_earned = 0.0
        self.total_spent = 0.0

        # Operational costs from config
        costs_config = wallet_config.get("costs", {})
        self.operation_costs = {
            "basic_response": costs_config.get("basic_response", 10.0),
            "enhanced_response": costs_config.get("enhanced_response", 25.0),
            "component_search": costs_config.get("component_search", 5.0),
            "priority_routing": costs_config.get("priority_routing", 15.0)
        }

        # Earning rates from config
        rewards_config = wallet_config.get("rewards", {})
        self.reward_rates = {
            "completion_item": rewards_config.get("completion_item", 50.0),
            "peer_verification": rewards_config.get("peer_verification", 20.0),
            "domain_terms": rewards_config.get("domain_terms", 2.0),
            "component_usage": rewards_config.get("component_usage", 5.0),
            "quality_bonus": rewards_config.get("quality_bonus", 10.0)
        }

    def decay_balance(self) -> float:
        """Apply metabolic decay to balance"""
        now = time.time()
        minutes_elapsed = (now - self.last_decay) / 60.0

        if minutes_elapsed > 0:
            # Exponential decay to simulate metabolic consumption
            decay_factor = math.exp(-self.decay_rate * minutes_elapsed)
            old_balance = self.balance
            self.balance = max(self.min_balance, self.balance * decay_factor)
            decay_amount = old_balance - self.balance

            if decay_amount > 0.01:  # Only log significant decay
                self.log_transaction("decay", -decay_amount, f"Metabolic decay over {minutes_elapsed:.1f}m")

            self.last_decay = now
            return decay_amount

        return 0.0

    def can_afford(self, operation: str) -> bool:
        """Check if agent can afford an operation"""
        self.decay_balance()
        cost = self.operation_costs.get(operation, 0.0)
        return self.balance >= cost

    def spend(self, operation: str, multiplier: float = 1.0) -> bool:
        """Spend resources for an operation"""
        self.decay_balance()
        cost = self.operation_costs.get(operation, 0.0) * multiplier

        if self.balance >= cost:
            self.balance -= cost
            self.total_spent += cost
            self.log_transaction("spend", -cost, f"{operation} (x{multiplier:.2f})")
            return True

        return False

    def earn(self, reward_type: str, multiplier: float = 1.0, description: str = "") -> float:
        """Earn resources for positive contributions"""
        amount = self.reward_rates.get(reward_type, 0.0) * multiplier

        if amount > 0:
            self.balance = min(self.max_balance, self.balance + amount)
            self.total_earned += amount
            desc = description or f"{reward_type} reward (x{multiplier:.2f})"
            self.log_transaction("earn", amount, desc)

        return amount

    def log_transaction(self, transaction_type: str, amount: float, description: str):
        """Log wallet transaction"""
        self.transaction_log.append({
            "timestamp": time.time(),
            "type": transaction_type,
            "amount": amount,
            "balance_after": self.balance,
            "description": description
        })

        # Keep only recent transactions to prevent memory bloat
        if len(self.transaction_log) > 100:
            self.transaction_log = self.transaction_log[-50:]

    def get_status(self) -> Dict:
        """Get current wallet status"""
        self.decay_balance()

        return {
            "agent_id": self.agent_id,
            "balance": self.balance,
            "balance_pct": (self.balance / self.starting_balance) * 100,
            "total_earned": self.total_earned,
            "total_spent": self.total_spent,
            "net_change": self.total_earned - self.total_spent,
            "last_decay": self.last_decay,
            "can_speak": self.balance >= self.operation_costs["basic_response"],
            "can_enhanced": self.balance >= self.operation_costs["enhanced_response"],
            "transactions": len(self.transaction_log)
        }

    def get_display_status(self) -> str:
        """Get compact display status"""
        self.decay_balance()
        balance_icon = "💰" if self.balance > 50 else "💸" if self.balance > 20 else "🚨"
        return f"{balance_icon}{self.balance:.0f}"

class WalletManager:
    """Manages all agent wallets with metabolic constraints"""

    def __init__(self):
        self.wallets = {}  # agent_id -> AgentWallet

        # Get wallet system config
        wallet_config = CONFIG.get("wallet", {})
        self.enabled = wallet_config.get("enabled", True)

        if not self.enabled:
            print("💸 Wallet system disabled in config")
            return

        print("💰 Initializing metabolic constraint wallet system...")

    def initialize_wallets(self, active_team: List[Dict]):
        """Initialize wallets for all active team members"""
        if not self.enabled:
            return

        wallet_config = CONFIG.get("wallet", {})
        starting_balance = wallet_config.get("starting_balance", 100.0)

        for agent in active_team:
            agent_id = agent["agent_id"]
            self.wallets[agent_id] = AgentWallet(agent_id, starting_balance)
            print(f"💳 Created wallet for {agent['agent_name']}: {starting_balance} credits")

        print(f"✅ Initialized {len(self.wallets)} agent wallets")

    def get_wallet(self, agent_id: str) -> Optional[AgentWallet]:
        """Get wallet for specific agent"""
        return self.wallets.get(agent_id) if self.enabled else None

    def can_agent_speak(self, agent_id: str, enhanced: bool = False) -> bool:
        """Check if agent has sufficient resources to speak"""
        if not self.enabled:
            return True

        wallet = self.get_wallet(agent_id)
        if not wallet:
            return True

        operation = "enhanced_response" if enhanced else "basic_response"
        return wallet.can_afford(operation)

    def charge_for_response(self, agent_id: str, enhanced: bool = False, generation_time: float = 0.0) -> bool:
        """Charge agent for generating a response"""
        if not self.enabled:
            return True

        wallet = self.get_wallet(agent_id)
        if not wallet:
            return True

        # Base cost
        operation = "enhanced_response" if enhanced else "basic_response"

        # Time-based multiplier (longer responses cost more)
        time_multiplier = 1.0 + (generation_time / 10.0)  # +10% per 10 seconds

        return wallet.spend(operation, time_multiplier)

    def reward_completion(self, agent_id: str, completion_items: int, quality_metrics: Dict):
        """Reward agent for completing project criteria"""
        if not self.enabled:
            return

        wallet = self.get_wallet(agent_id)
        if not wallet:
            return

        # Base completion rewards
        if completion_items > 0:
            wallet.earn("completion_item", completion_items, f"Completed {completion_items} criteria")

        # Quality bonuses
        if quality_metrics.get("domain_terms_used", 0) > 3:
            wallet.earn("domain_terms", quality_metrics["domain_terms_used"] / 3.0, "Domain expertise")

        if quality_metrics.get("components_mentioned", 0) > 0:
            wallet.earn("component_usage", quality_metrics["components_mentioned"], "Component integration")

        # Validation bonus
        req_score = sum(1 for k, v in quality_metrics.items() if k.endswith("_ok") and v)
        if req_score > 0:
            wallet.earn("quality_bonus", req_score * 0.5, "Requirements compliance")

    def get_available_agents(self, all_agents: List[Dict], enhanced_needed: bool = False) -> List[Dict]:
        """Filter agents by wallet balance - metabolic selection pressure"""
        if not self.enabled:
            return all_agents

        available = []
        for agent in all_agents:
            if self.can_agent_speak(agent["agent_id"], enhanced_needed):
                available.append(agent)

        return available

    def get_wallet_summary(self) -> str:
        """Get summary of all wallet statuses"""
        if not self.enabled:
            return "Wallet system disabled"

        if not self.wallets:
            return "No wallets initialized"

        statuses = []
        for agent_id, wallet in self.wallets.items():
            status = wallet.get_status()
            name = agent_id.replace("_1", "").replace("_", " ").title()
            balance = status["balance"]
            pct = status["balance_pct"]

            # Status indicator
            if balance >= 75:
                indicator = "🟢"
            elif balance >= 40:
                indicator = "🟡"
            elif balance >= 20:
                indicator = "🟠"
            else:
                indicator = "🔴"

            statuses.append(f"{indicator}{name}({balance:.0f}|{pct:.0f}%)")

        return " | ".join(statuses)

    def get_detailed_wallet_report(self) -> str:
        """Generate detailed wallet report"""
        if not self.enabled:
            return "💸 Wallet system disabled"

        if not self.wallets:
            return "💳 No wallets active"

        lines = ["💰 WALLET STATUS REPORT"]

        total_balance = 0
        total_earned = 0
        total_spent = 0

        for agent_id, wallet in self.wallets.items():
            status = wallet.get_status()
            name = agent_id.replace("_1", "").replace("_", " ").title()

            # Accumulate totals
            total_balance += status["balance"]
            total_earned += status["total_earned"]
            total_spent += status["total_spent"]

            # Agent status
            balance_bar = "█" * int(status["balance"] / 10) + "░" * (10 - int(status["balance"] / 10))
            lines.append(f"  {name}: {status['balance']:.1f} [{balance_bar}] "
                        f"E:{status['total_earned']:.0f} S:{status['total_spent']:.0f}")

        # Summary
        lines.append(f"💹 Economy: Total Balance: {total_balance:.0f} | "
                    f"Earned: {total_earned:.0f} | Spent: {total_spent:.0f}")

        return "\n".join(lines)

# Initialize global wallet manager
wallet_manager = WalletManager()

# ===========================================
# Completion Tracking System with Wallet Integration
# ===========================================

class CompletionTracker:
    """Tracks completion of project criteria and deliverables with wallet rewards"""

    def __init__(self):
        self.completion_items = {}  # id -> {category, description, completed, evidence}
        self.categories = {}  # category -> {items_count, completed_count}

    def add_completion_item(self, item_id: str, category: str, description: str):
        """Add a trackable completion item"""
        self.completion_items[item_id] = {
            "category": category,
            "description": description,
            "completed": False,
            "evidence": "",
            "completed_by": "",
            "completed_at": None
        }

        if category not in self.categories:
            self.categories[category] = {"items_count": 0, "completed_count": 0}
        self.categories[category]["items_count"] += 1

    def mark_completed(self, item_id: str, evidence: str = "", completed_by: str = "", agent_id: str = ""):
        """Mark an item as completed with evidence and wallet reward"""
        if item_id in self.completion_items:
            item = self.completion_items[item_id]
            if not item["completed"]:
                item["completed"] = True
                item["evidence"] = evidence
                item["completed_by"] = completed_by
                item["completed_at"] = datetime.now().isoformat()

                category = item["category"]
                self.categories[category]["completed_count"] += 1

                # Wallet reward for completion
                if agent_id and wallet_manager.enabled:
                    wallet = wallet_manager.get_wallet(agent_id)
                    if wallet:
                        wallet.earn("completion_item", 1.0, f"Completed: {item['description'][:50]}")

                return True
        return False

    def get_completion_percentage(self) -> float:
        """Get overall completion percentage"""
        total_items = len(self.completion_items)
        if total_items == 0:
            return 0.0
        completed_items = sum(1 for item in self.completion_items.values() if item["completed"])
        return (completed_items / total_items) * 100

    def get_category_progress(self, category: str) -> Dict:
        """Get progress for a specific category"""
        if category not in self.categories:
            return {"completed": 0, "total": 0, "percentage": 0.0}

        cat_data = self.categories[category]
        percentage = (cat_data["completed_count"] / cat_data["items_count"]) * 100 if cat_data["items_count"] > 0 else 0

        return {
            "completed": cat_data["completed_count"],
            "total": cat_data["items_count"],
            "percentage": percentage
        }

    def get_incomplete_items(self, category: str = None) -> List[Dict]:
        """Get list of incomplete items, optionally filtered by category"""
        incomplete = []
        for item_id, item in self.completion_items.items():
            if not item["completed"]:
                if category is None or item["category"] == category:
                    incomplete.append({
                        "id": item_id,
                        "category": item["category"],
                        "description": item["description"]
                    })
        return incomplete

    def get_completion_summary(self) -> str:
        """Get visual summary of completion status"""
        if not self.completion_items:
            return "No completion criteria defined"

        overall_pct = self.get_completion_percentage()

        summary_parts = [f"Overall: {overall_pct:.1f}%"]

        for category in sorted(self.categories.keys()):
            progress = self.get_category_progress(category)
            indicator = "🟢" if progress["percentage"] == 100 else "🟡" if progress["percentage"] > 50 else "🔴"
            summary_parts.append(f"{indicator} {category.title()}: {progress['completed']}/{progress['total']}")

        return " | ".join(summary_parts)

    def is_project_complete(self) -> bool:
        """Check if all completion criteria are met"""
        return self.get_completion_percentage() >= 100.0

    def get_detailed_completion_report(self) -> str:
        """Generate detailed completion report with proper checkboxes"""
        if not self.completion_items:
            return "No completion criteria defined"

        report_lines = []

        # Overall status
        overall_pct = self.get_completion_percentage()
        report_lines.append(f"📈 Project Completion: {overall_pct:.1f}%")

        # Category breakdown
        for category in sorted(self.categories.keys()):
            progress = self.get_category_progress(category)
            report_lines.append(f"\n{category.replace('_', ' ').title()} ({progress['completed']}/{progress['total']}):")

            # Show all items in category with proper checkboxes
            category_items = [(item_id, item) for item_id, item in self.completion_items.items()
                            if item["category"] == category]

            for item_id, item in category_items:
                checkbox = Colors.CHECKBOX_FILLED if item["completed"] else Colors.CHECKBOX_EMPTY
                description = item["description"][:60] + "..." if len(item["description"]) > 60 else item["description"]
                report_lines.append(f"  {checkbox} {description}")

                if item["completed"]:
                    report_lines.append(f"    ✓ Completed by: {item['completed_by']}")

        return "\n".join(report_lines)

# ===========================================
# LoRA Adapter Management
# ===========================================

class AdapterManager:
    """Manages LoRA adapter loading and caching"""

    def __init__(self):
        self.adapters_path = Path(CONFIG.get("adapters", {}).get("adapters_path", "adapters/lora"))
        self.adapter_cache = {}

        try:
            from peft import PeftModel
            self.peft_available = True
            print("🧩 PEFT library available")
        except ImportError:
            self.peft_available = False
            print("⚠️  PEFT not available - adapters disabled")

    def find_latest_adapter(self, model_key: str) -> Optional[str]:
        """Find the most recent adapter for a model"""
        if not self.adapters_path.exists():
            print(f"📁 Adapters directory missing: {self.adapters_path}")
            return None

        pattern = f"{model_key}_*"
        matching_dirs = list(self.adapters_path.glob(pattern))

        if CONFIG.get("display", {}).get("verbose_logging", False):
            print(f"🔍 Looking for {model_key} adapters in {self.adapters_path}")
            print(f"🎯 Pattern: {pattern}")
            print(f"📋 Found: {[d.name for d in matching_dirs]}")

        if not matching_dirs:
            return None

        matching_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest = matching_dirs[0]

        config_file = latest / "adapter_config.json"
        if config_file.exists():
            print(f"✅ Valid adapter found: {latest.name}")
            return str(latest)
        else:
            print(f"❌ Invalid adapter (no config): {latest.name}")
            return None

    def get_adapter_path(self, model_key: str) -> Optional[str]:
        """Get adapter path based on configuration"""
        adapters_config = CONFIG.get("adapters", {})

        if not adapters_config.get("use_adapters", False) or not self.peft_available:
            return None

        adapter_mode = adapters_config.get("adapter_mode", "latest")

        if adapter_mode == "none":
            return None
        elif adapter_mode == "specific":
            return adapters_config.get("specific_adapters", {}).get(model_key)
        elif adapter_mode == "latest":
            return self.find_latest_adapter(model_key)

        return None

    def load_model_with_adapter(self, model_key: str):
        """Load model with optional adapter"""
        cache_key = f"{model_key}_with_adapter"
        if cache_key in self.adapter_cache:
            return self.adapter_cache[cache_key]

        base_model_data = MODEL_CACHE[model_key]
        base_model = base_model_data["model"]
        tokenizer = base_model_data["tokenizer"]

        adapter_path = self.get_adapter_path(model_key)

        if adapter_path and Path(adapter_path).exists():
            try:
                from peft import PeftModel
                print(f"🔧 Loading adapter for {model_key}: {Path(adapter_path).name}")

                model_with_adapter = PeftModel.from_pretrained(base_model, adapter_path)

                enhanced_model_data = {
                    "model": model_with_adapter,
                    "tokenizer": tokenizer,
                    "adapter_path": adapter_path,
                    "enhanced": True
                }

                self.adapter_cache[cache_key] = enhanced_model_data
                return enhanced_model_data

            except Exception as e:
                print(f"⚠️  Failed to load adapter {adapter_path}: {e}")
                print(f"🔄 Falling back to base model for {model_key}")

        base_model_enhanced = base_model_data.copy()
        base_model_enhanced["enhanced"] = False
        base_model_enhanced["adapter_path"] = None

        self.adapter_cache[cache_key] = base_model_enhanced
        return base_model_enhanced

adapter_manager = AdapterManager()

# ===========================================
# Enhanced KB Parser with Completion Tracking
# ===========================================

class EnhancedKBParser:
    """Enhanced parser with semantic search, improved component matching, and completion tracking"""

    def __init__(self):
        kb_config = CONFIG.get("knowledge_base", {})
        self.kb_dir = Path(kb_config.get("kb_directory", "kb"))

        # Project context
        self.project_context = {}
        self.requirements = {}
        self.constraints = []
        self.banned_terms = []
        self.success_metrics = []
        self.acceptance_criteria = []
        self.domain_terminology = set()
        self.project_type = "unknown"
        self.kb_version = "1.0"

        # Completion tracking
        self.completion_tracker = CompletionTracker()

        # Agent definitions
        self.agent_roles = {}
        self.model_pools = CONFIG.get("models", {}).get("model_pools", {})
        self.active_team = self._process_active_team()

        # Initialize wallets for active team
        wallet_manager.initialize_wallets(self.active_team)

        # Components and embeddings
        self.components_db = self._load_components_database()
        self.component_embeddings = None
        self._cache_component_embeddings()

        self._parse_knowledge_base()
        self._parse_agent_definitions()

        print(f"🎯 Initialized with {len(self.agent_roles)} role types")
        print(f"👥 Active team: {len(self.active_team)} agents")
        print(f"🔧 Components database: {len(self.components_db)} items")
        print(f"🧠 Semantic search: {'Enabled' if self.component_embeddings is not None else 'Disabled'}")
        print(f"📋 Completion tracking: {len(self.completion_tracker.completion_items)} criteria")
        print(f"💰 Wallet system: {'Enabled' if wallet_manager.enabled else 'Disabled'}")

    def _load_components_database(self) -> List[Dict]:
        """Load components database with enhanced error handling"""
        kb_config = CONFIG.get("knowledge_base", {})
        components_file = Path(kb_config.get("components_file", "kb/components.csv"))

        if not components_file.exists():
            print("📋 No components.csv found")
            return []

        try:
            import pandas as pd
            df = pd.read_csv(components_file, on_bad_lines='skip', engine='python')
            components = df.to_dict('records')
            print(f"✅ Loaded {len(components)} components from database")
            return components
        except Exception as e:
            print(f"⚠️  Failed to load components database: {e}")
            print("🔄 Trying manual CSV parsing...")

            try:
                import csv
                components = []
                with open(components_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for i, row in enumerate(reader):
                        try:
                            components.append(dict(row))
                        except Exception as row_error:
                            print(f"⚠️  Skipping malformed row {i+2}: {row_error}")
                            continue

                print(f"✅ Loaded {len(components)} components via fallback parsing")
                return components
            except Exception as fallback_error:
                print(f"❌ Fallback parsing failed: {fallback_error}")
                return []

    def _cache_component_embeddings(self):
        """Pre-compute embeddings for all components"""
        if not self.components_db:
            return

        try:
            print("🧠 Computing component embeddings for semantic search...")
            component_texts = []

            for component in self.components_db:
                # Create rich text representation
                text_parts = [
                    str(component.get('part_name', '')),
                    str(component.get('component_type', '')),
                    str(component.get('description', '')),
                    str(component.get('manufacturer', '')),
                    str(component.get('specs', ''))
                ]
                component_text = ' '.join([part for part in text_parts if part and part != 'nan'])
                component_texts.append(component_text)

            # Compute embeddings using the global embedding model
            self.component_embeddings = EMBEDDING_MODEL.encode(component_texts)
            print(f"✅ Cached {len(component_texts)} component embeddings")

        except Exception as e:
            print(f"⚠️  Failed to compute component embeddings: {e}")
            self.component_embeddings = None

    def get_relevant_components_semantic(self, query_text: str, max_results: int = 5) -> List[Dict]:
        """Get relevant components using semantic search with wallet integration"""
        if not self.components_db or self.component_embeddings is None:
            return self.get_relevant_components_keyword(query_text, max_results)

        try:
            # Embed the query
            query_embedding = EMBEDDING_MODEL.encode([query_text])

            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.component_embeddings)[0]

            # Get top matches
            top_indices = similarities.argsort()[-max_results:][::-1]

            relevant_components = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    component = self.components_db[idx].copy()
                    component['relevance_score'] = float(similarities[idx])
                    relevant_components.append(component)

            return relevant_components

        except Exception as e:
            print(f"⚠️  Semantic search failed: {e}, falling back to keyword search")
            return self.get_relevant_components_keyword(query_text, max_results)

    def get_relevant_components_keyword(self, query_text: str, max_results: int = 5) -> List[Dict]:
        """Fallback keyword-based component search"""
        if not self.components_db:
            return []

        query_lower = query_text.lower()
        relevant = []

        for component in self.components_db:
            relevance_score = 0
            component_text = " ".join([
                str(component.get('part_name', '')),
                str(component.get('description', '')),
                str(component.get('component_type', ''))
            ]).lower()

            query_words = query_lower.split()
            for word in query_words:
                if len(word) > 3 and word in component_text:
                    relevance_score += 1

            if relevance_score > 0:
                component_copy = component.copy()
                component_copy['relevance_score'] = relevance_score
                relevant.append(component_copy)

        relevant.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant[:max_results]

    def get_relevant_components(self, query_text: str, max_results: int = 5) -> List[Dict]:
        """Main component search method - uses semantic search if available"""
        return self.get_relevant_components_semantic(query_text, max_results)

    def _process_active_team(self) -> List[Dict]:
        """Process active team with auto-numbering"""
        raw_team = CONFIG.get("agents", {}).get("active_team", [])
        processed_team = []
        role_counters = {}

        for agent_spec in raw_team:
            role_type = agent_spec["role_type"]
            agent_name = agent_spec["agent_name"]

            if role_type not in role_counters:
                role_counters[role_type] = 1
            else:
                role_counters[role_type] += 1

            instance_num = role_counters[role_type]
            agent_id = f"{agent_name.lower()}_{instance_num}"

            processed_agent = {
                "agent_id": agent_id,
                "role_type": role_type,
                "agent_name": agent_name,
                "instance_number": instance_num,
                "turn_count": 0  # Track participation
            }

            processed_team.append(processed_agent)

        return processed_team

    def _parse_knowledge_base(self):
        """Parse project-specific KB files with completion tracking"""
        print(f"📚 Loading project knowledge base from {self.kb_dir}/")

        kb_files = [f for f in self.kb_dir.glob("*.md")]
        print(f"📄 Found {len(kb_files)} knowledge files")

        for kb_file in kb_files:
            file_content = kb_file.read_text(encoding="utf-8")
            file_type = kb_file.stem.lower()

            print(f"📖 Parsing {kb_file.name}...")

            if file_type in ["sot", "source_of_truth"]:
                self._parse_sot_file(file_content)
            elif file_type in ["acceptance", "criteria"]:
                self._parse_acceptance_file(file_content)
            elif file_type in ["notes", "design_notes", "decisions"]:
                self._parse_notes_file(file_content)
            elif file_type in ["tests", "test_protocols", "testing", "tests_template"]:
                self._parse_tests_file(file_content)
            elif file_type != "agents":  # Skip agents.md as it's parsed separately
                self._parse_generic_completion_file(file_content, file_type)

            self._extract_domain_terminology(file_content)

        self._infer_project_type()

        print(f"📋 Project parsing complete:")
        print(f"  📁 Project type: {self.project_type}")
        print(f"  🎯 Topic: {self.project_context.get('topic', 'Not extracted')}")
        print(f"  📊 Requirements: {len(self.requirements)}")
        print(f"  🚫 Constraints: {len(self.constraints)}")
        print(f"  🎪 Domain terms: {len(self.domain_terminology)}")
        print(f"  ✅ Completion items: {len(self.completion_tracker.completion_items)}")

    def _parse_acceptance_file(self, content: str):
        """Parse acceptance criteria with completion tracking"""
        # Extract main sections
        sections = re.findall(r'##\s+([^#\n]+)\n((?:(?!##).)*)', content, re.DOTALL)

        for section_name, section_content in sections:
            section_clean = section_name.strip().lower().replace(' ', '_')

            # Extract bullet points as completion items
            criteria_items = re.findall(r'-\s*(.+?)(?=\n-|\n\n|\Z)', section_content, re.DOTALL)

            for i, item in enumerate(criteria_items):
                item_clean = item.strip()
                if item_clean and len(item_clean) > 10:  # Filter out very short items
                    item_id = f"acceptance_{section_clean}_{i+1}"
                    self.completion_tracker.add_completion_item(
                        item_id=item_id,
                        category=section_clean,
                        description=item_clean
                    )

            self.acceptance_criteria.extend([item.strip() for item in criteria_items if item.strip()])

    def _parse_tests_file(self, content: str):
        """Parse test protocols with completion tracking"""
        # Extract test sections
        sections = re.findall(r'##\s+([^#\n]+)\n((?:(?!##).)*)', content, re.DOTALL)

        for section_name, section_content in sections:
            section_clean = section_name.strip().lower().replace(' ', '_')

            # Extract test items as completion criteria
            test_items = re.findall(r'-\s*(.+?)(?=\n-|\n\n|\Z)', section_content, re.DOTALL)

            for i, item in enumerate(test_items):
                item_clean = item.strip()
                if item_clean and len(item_clean) > 10:
                    item_id = f"test_{section_clean}_{i+1}"
                    self.completion_tracker.add_completion_item(
                        item_id=item_id,
                        category="testing",
                        description=item_clean
                    )

            # Store in project context
            self.project_context[f"test_{section_clean}"] = [item.strip() for item in test_items if item.strip()]

    def _parse_notes_file(self, content: str):
        """Parse design notes with open issues as completion items"""
        # Extract decision patterns
        decision_patterns = re.findall(r"(?:Decision|Choice|Selected):\s*(.+?)(?=\n|$)", content, re.IGNORECASE)
        for decision in decision_patterns:
            self.project_context[f"decision_{len(self.project_context)}"] = decision.strip()

        # Extract open issues as completion items
        open_issues_section = re.search(r'##\s*Open Issues\s*\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
        if open_issues_section:
            issues = re.findall(r'-\s*(.+?)(?=\n-|\n\n|\Z)', open_issues_section.group(1), re.DOTALL)
            for i, issue in enumerate(issues):
                issue_clean = issue.strip()
                if issue_clean:
                    item_id = f"open_issue_{i+1}"
                    self.completion_tracker.add_completion_item(
                        item_id=item_id,
                        category="design_decisions",
                        description=f"Resolve: {issue_clean}"
                    )

    def _parse_generic_completion_file(self, content: str, file_type: str):
        """Parse any markdown file for completion items"""
        # Extract bullet points as potential completion items
        sections = re.findall(r'##\s+([^#\n]+)\n((?:(?!##).)*)', content, re.DOTALL)

        for section_name, section_content in sections:
            section_clean = section_name.strip().lower().replace(' ', '_')
            items = re.findall(r'-\s*(.+?)(?=\n-|\n\n|\Z)', section_content, re.DOTALL)

            for i, item in enumerate(items):
                item_clean = item.strip()
                if item_clean and len(item_clean) > 15:  # Only substantial items
                    item_id = f"{file_type}_{section_clean}_{i+1}"
                    self.completion_tracker.add_completion_item(
                        item_id=item_id,
                        category=file_type,
                        description=item_clean
                    )

    def _parse_agent_definitions(self):
        """Parse agent role definitions from kb/agents.md"""
        agents_file = self.kb_dir / "agents.md"

        if not agents_file.exists():
            print("📝 No agents.md found, using default roles")
            self._create_default_agent_roles()
            return

        print(f"👥 Loading agent definitions from {agents_file}")
        content = agents_file.read_text(encoding="utf-8")

        role_sections = re.findall(r'##\s+(\w+)\s*\n(.*?)(?=\n##|\Z)', content, re.DOTALL)

        for role_name, role_content in role_sections:
            role_def = {"role_type": role_name.lower()}

            # Extract structured fields with improved regex
            fields = {
                "model_pool": r'\*\*Model Pool\*\*:\s*(.+?)(?=\n\*\*|\n\n|\Z)',
                "personality": r'\*\*Personality\*\*:\s*(.+?)(?=\n\*\*|\n\n|\Z)',
                "responsibilities": r'\*\*Responsibilities\*\*:\s*(.+?)(?=\n\*\*|\n\n|\Z)',
                "expertise": r'\*\*Expertise\*\*:\s*(.+?)(?=\n\*\*|\n\n|\Z)',
                "collaborates_with": r'\*\*Collaborates With\*\*:\s*(.+?)(?=\n\*\*|\n\n|\Z)',
                "introduction_template": r'\*\*Introduction Template\*\*:\s*"(.+?)"'
            }

            for field, pattern in fields.items():
                match = re.search(pattern, role_content, re.DOTALL)
                if match:
                    role_def[field] = match.group(1).strip()

            # Use explicit model pool if defined, otherwise use default assignment
            if "model_pool" not in role_def:
                model_pool = self._get_default_model_pool(role_name.lower())
                role_def["model_pool"] = model_pool

            self.agent_roles[role_name.lower()] = role_def
            print(f"👤 Loaded role: {role_name.lower()} (pool: {role_def['model_pool']})")

        print(f"✅ Loaded {len(self.agent_roles)} agent role definitions")

    def _get_default_model_pool(self, role_type: str) -> str:
        """Assign default model pool based on role type"""
        technical_roles = ["electrical_engineer", "mechanical_engineer", "flight_controls_engineer",
                          "software_engineer", "avionics_engineer"]
        practical_roles = ["test_engineer", "welder", "machinist", "assembler", "field_engineer"]
        strategic_roles = ["manager", "systems_engineer", "safety_engineer", "finance_analyst"]
        reasoning_roles = ["reasoning_specialist"]

        if role_type in technical_roles:
            return "technical"
        elif role_type in practical_roles:
            return "practical"
        elif role_type in strategic_roles:
            return "strategic"
        elif role_type in reasoning_roles:
            return "reasoning"
        else:
            return "technical"

    def _create_default_agent_roles(self):
        """Create default agent roles if no agents.md exists"""
        default_roles = {
            "manager": {
                "role_type": "manager",
                "personality": "Strategic, collaborative, coordination-focused",
                "responsibilities": "Project coordination, resource management, decision synthesis",
                "expertise": "Project management and stakeholder coordination",
                "collaborates_with": "all team members",
                "introduction_template": "I'm {agent_name}, the project manager coordinating this team.",
                "model_pool": "strategic"
            }
        }
        self.agent_roles = default_roles
        print(f"📝 Created {len(default_roles)} default agent roles")

    def _parse_sot_file(self, content: str):
        """Parse SOT file with improved patterns"""
        topic_patterns = [
            r"##?\s*(?:Mission\s+Requirements?|Goal|Objective|Topic)\s*[:\n]\s*(.*?)(?=\n##|\n-|\Z)",
            r"Design\s+a\s+(.*?)(?=:|\.|\n)"
        ]

        for pattern in topic_patterns:
            topic_match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if topic_match:
                topic_text = re.sub(r'\s+', ' ', topic_match.group(1).strip())
                if len(topic_text) > 10:
                    self.project_context["topic"] = topic_text
                    break

        req_pattern = r"-\s*\*\*([^*]+)\*\*:\s*([^-\n]+)"
        matches = re.findall(req_pattern, content)

        for req_name, req_value in matches:
            req_name = req_name.strip().lower()
            req_value = req_value.strip()

            # Enhanced requirement parsing
            if any(unit in req_value.lower() for unit in ['kg', 'km', 'gb', 'mb', 'hours', 'minutes']):
                num_match = re.search(r'(\d+(?:\.\d+)?)\s*([a-zA-Z]+)', req_value)
                if num_match:
                    value, unit = num_match.groups()
                    self.requirements[req_name] = {
                        "value": value,
                        "unit": unit,
                        "type": "numerical",
                        "pattern": rf"\b{re.escape(value)}\s*{re.escape(unit)}\b",
                        "description": f"{req_name.title()}: {value} {unit}"
                    }
            elif '$' in req_value or 'usd' in req_value.lower():
                money_match = re.search(r'[\$]?\s*(\d+(?:,\d+)*)', req_value)
                if money_match:
                    value = money_match.group(1).replace(',', '')
                    self.requirements[req_name] = {
                        "value": value,
                        "unit": "USD",
                        "type": "monetary",
                        "pattern": rf"(\$\s*{re.escape(value)}|\b{re.escape(value)}\s*(?:usd|\$)?)\b",
                        "description": f"{req_name.title()}: ${value}"
                    }
            else:
                self.requirements[req_name] = {
                    "value": req_value,
                    "unit": "",
                    "type": "descriptive",
                    "pattern": rf"\b{re.escape(req_value.lower())}\b",
                    "description": f"{req_name.title()}: {req_value}"
                }

        # Extract banned terms
        banned_section = re.search(r'##\s*Banned Terms\s*\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
        if banned_section:
            banned_text = banned_section.group(1).strip()
            self.banned_terms = [term.strip() for term in re.split(r'[,\n]', banned_text) if term.strip()]

    def _extract_domain_terminology(self, content: str):
        """Extract domain-specific terminology"""
        technical_terms = re.findall(r'\b[A-Z][A-Za-z]{3,}\b', content)
        compound_terms = re.findall(r'\b(?:[A-Za-z]+-[A-Za-z]+|[A-Z][A-Za-z]*\s[A-Z][A-Za-z]*)\b', content)

        self.domain_terminology.update(technical_terms)
        self.domain_terminology.update(compound_terms)

        common_words = {"This", "That", "With", "From", "They", "Will", "When", "Where", "What", "Which",
                       "Design", "System", "Project", "Version", "Initial", "Future", "Technical", "Analysis"}
        self.domain_terminology = self.domain_terminology - common_words

    def _infer_project_type(self):
        """Infer project type from content"""
        content_text = " ".join([
            self.project_context.get("topic", ""),
            " ".join(self.requirements.keys()),
            " ".join(self.constraints),
            " ".join(self.domain_terminology)
        ]).lower()

        type_indicators = {
            "software": ["software", "application", "api", "database", "code"],
            "hardware": ["hardware", "device", "circuit", "component", "drone", "motor"],
            "engineering": ["design", "engineering", "system", "technical", "battery"],
            "business": ["business", "strategy", "market", "revenue", "customer"],
            "research": ["research", "study", "analysis", "investigation", "experiment"],
            "product": ["product", "feature", "user", "interface", "experience"]
        }

        scores = {}
        for proj_type, indicators in type_indicators.items():
            score = sum(1 for indicator in indicators if indicator in content_text)
            if score > 0:
                scores[proj_type] = score

        self.project_type = max(scores, key=scores.get) if scores else "general"

    def check_completion_in_response(self, response: str, agent_name: str, agent_id: str = "") -> List[str]:
        """Check if response indicates completion of any criteria with wallet rewards"""
        completed_items = []
        response_lower = response.lower()

        # Look for completion indicators
        completion_phrases = [
            r"completed?\s+(.{10,100})",
            r"finished\s+(.{10,100})",
            r"done\s+with\s+(.{10,100})",
            r"accomplished\s+(.{10,100})",
            r"✓\s*(.{10,100})",
            r"checked?\s+(.{10,100})"
        ]

        for item_id, item in self.completion_tracker.completion_items.items():
            if not item["completed"]:
                item_desc_lower = item["description"].lower()

                # Check for direct mentions of the completion item
                for phrase_pattern in completion_phrases:
                    matches = re.findall(phrase_pattern, response_lower)
                    for match in matches:
                        # Simple similarity check
                        if any(word in item_desc_lower for word in match.split() if len(word) > 3):
                            self.completion_tracker.mark_completed(
                                item_id=item_id,
                                evidence=match.strip(),
                                completed_by=agent_name,
                                agent_id=agent_id
                            )
                            completed_items.append(item_id)
                            break

        return completed_items

    def get_agent_system_prompt(self, agent_instance: Dict) -> str:
        """Generate enhanced system prompt for agent with completion and wallet context"""
        role_def = self.agent_roles.get(agent_instance["role_type"], {})

        # Get incomplete items that this agent could work on
        incomplete_items = self.completion_tracker.get_incomplete_items()
        completion_context = ""

        if incomplete_items:
            relevant_items = [item for item in incomplete_items[:5]]  # Show top 5
            if relevant_items:
                completion_context = "\n\n📋 Outstanding Completion Items:\n"
                for item in relevant_items:
                    completion_context += f"• {item['description']}\n"
                completion_context += "\nWhen you complete any criteria, clearly state what was accomplished."

        # Wallet context
        wallet_context = ""
        if wallet_manager.enabled:
            wallet = wallet_manager.get_wallet(agent_instance["agent_id"])
            if wallet:
                status = wallet.get_status()
                wallet_context = f"\n\n💰 Resource Status: {status['balance']:.0f} credits available"
                if status['balance'] < 30:
                    wallet_context += " ⚠️ Low resources - focus on high-value contributions"

        prompt_parts = [
            f"You are {agent_instance['agent_name']}, a {agent_instance['role_type']} in a collaborative project team.",
            "",
            f"🎭 Personality: {role_def.get('personality', 'Professional and focused')}",
            f"📋 Responsibilities: {role_def.get('responsibilities', 'Contribute to project success')}",
            f"🔧 Expertise: {role_def.get('expertise', 'General project knowledge')}",
            f"🤝 Collaborates with: {role_def.get('collaborates_with', 'other team members')}",
            "",
            "📜 Instructions:",
            "• Address other agents by name when asking questions or making requests",
            "• Build on previous contributions rather than repeating them",
            "• Stay within project requirements and constraints",
            "• Be specific and actionable in your responses",
            "• Work collaboratively to achieve project objectives",
            "• Reference specific components from the database when relevant",
            "• Clearly indicate when you complete any criteria or deliverables",
            completion_context,
            wallet_context
        ]

        return "\n".join(prompt_parts)

    def get_introduction_text(self, agent_instance: Dict) -> str:
        """Generate introduction text for agent"""
        role_def = self.agent_roles.get(agent_instance["role_type"], {})
        template = role_def.get("introduction_template",
                               "I'm {agent_name}, working as {role_type} on this project.")

        return template.format(
            agent_name=agent_instance["agent_name"],
            role_type=agent_instance["role_type"],
            expertise=role_def.get("expertise", "project work")
        )

    def get_model_for_agent(self, agent_instance: Dict) -> str:
        """Get model ID for agent instance"""
        role_def = self.agent_roles.get(agent_instance["role_type"], {})
        model_pool = role_def.get("model_pool", "technical")

        # Make sure model_pools maps correctly
        model_key = self.model_pools.get(model_pool, "architect")
        return model_key

    def get_agent_color(self, agent_instance: Dict) -> str:
        """Get color for agent based on model assignment"""
        model_key = self.get_model_for_agent(agent_instance)
        color = MODEL_COLORS.get(model_key, Colors.CYAN)
        return color

    def get_project_topic(self) -> str:
        """Get the main project topic/objective"""
        topic = self.project_context.get("topic", "")
        if topic:
            return topic

        if self.requirements:
            req_summary = []
            for req_name, req_data in list(self.requirements.items())[:3]:
                if req_data["type"] in ["numerical", "monetary"]:
                    req_summary.append(f"{req_data['value']} {req_data['unit']} {req_name}")
            if req_summary:
                return f"Project with requirements: {', '.join(req_summary)}"

        return "Project collaboration and solution development"

    def get_domain_context_summary(self) -> str:
        """Generate domain context summary for agent prompts"""
        context_parts = []

        if self.requirements:
            req_summary = []
            for req_name, req_data in self.requirements.items():
                req_summary.append(f"• {req_data['description']}")
            context_parts.append("📋 Requirements:\n" + "\n".join(req_summary))

        if self.constraints:
            constraint_summary = [f"• {constraint}" for constraint in self.constraints]
            context_parts.append("🚫 Constraints:\n" + "\n".join(constraint_summary))

        if self.success_metrics:
            context_parts.append("🎯 Success Metrics:\n" + "\n".join([f"• {metric}" for metric in self.success_metrics]))

        return "\n\n".join(context_parts) if context_parts else "Project requirements to be determined from discussion."

    def validate_against_requirements(self, text: str) -> Dict:
        """Validate text against parsed requirements"""
        text_lower = text.lower()
        validation_results = {}

        for req_name, req_data in self.requirements.items():
            pattern = req_data["pattern"]
            field_name = f"req_{req_name.replace(' ', '_').replace('-', '_')}_ok"
            validation_results[field_name] = bool(re.search(pattern, text_lower, re.IGNORECASE))

        banned_count = sum(1 for term in self.banned_terms
                          if re.search(rf"\b{re.escape(term)}\b", text_lower, re.IGNORECASE))
        validation_results["banned_terms"] = banned_count

        domain_term_count = sum(1 for term in self.domain_terminology if term.lower() in text_lower)
        validation_results["domain_terms_used"] = domain_term_count

        return validation_results

kb_parser = EnhancedKBParser()

# ===========================================
# Enhanced CSV Logging
# ===========================================

def ensure_csv_schema(csv_name: str, data: Dict):
    """Ensure CSV file has all required fields from data"""
    data_config = CONFIG.get("data", {})
    data_dir = Path(data_config.get("data_directory", "data"))
    csv_path = data_dir / f"{csv_name}.csv"
    csv_path.parent.mkdir(exist_ok=True)

    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data.keys()), quoting=csv.QUOTE_ALL)
            writer.writeheader()
        return list(data.keys())

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            existing_headers = next(reader)
        except StopIteration:
            existing_headers = []

    new_fields = [field for field in data.keys() if field not in existing_headers]

    if new_fields:
        all_rows = []
        if existing_headers:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                all_rows = list(reader)

        all_headers = existing_headers + new_fields

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_headers, quoting=csv.QUOTE_ALL)
            writer.writeheader()

            for row in all_rows:
                writer.writerow(row)

    return existing_headers + new_fields

def log_to_csv(csv_name: str, data: Dict):
    """Enhanced CSV logging with dynamic schema support"""
    try:
        headers = ensure_csv_schema(csv_name, data)

        data_config = CONFIG.get("data", {})
        data_dir = Path(data_config.get("data_directory", "data"))
        csv_path = data_dir / f"{csv_name}.csv"

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers, quoting=csv.QUOTE_ALL)
            row_data = {field: data.get(field, "") for field in headers}
            writer.writerow(row_data)
    except Exception as e:
        print(f"⚠️  Failed to log to {csv_name}: {e}")

# ===========================================
# Enhanced Response Generation with Wallet Integration
# ===========================================

def generate_streaming_response(agent_instance: Dict, messages: List[Dict], context: Dict) -> Tuple[str, float, Dict]:
    """Generate streaming response with wallet integration and enhanced component integration"""

    # Check wallet balance before generation
    agent_id = agent_instance["agent_id"]
    if not wallet_manager.can_agent_speak(agent_id):
        print(f"💸 {agent_instance['agent_name']} insufficient credits for response")
        return "I need to conserve my resources right now. Let me focus on listening.", 0.1, {"wallet_blocked": True}

    # Get model with adapter
    model_key = kb_parser.get_model_for_agent(agent_instance)
    model_data = adapter_manager.load_model_with_adapter(model_key)
    tokenizer = model_data["tokenizer"]
    model = model_data["model"]
    is_enhanced = model_data["enhanced"]
    adapter_path = model_data["adapter_path"]

    # Check enhanced model costs
    if is_enhanced and not wallet_manager.can_agent_speak(agent_id, enhanced=True):
        print(f"💸 {agent_instance['agent_name']} using base model - insufficient credits for enhanced")
        # Fallback to base model
        base_model_data = MODEL_CACHE[model_key]
        model = base_model_data["model"]
        tokenizer = base_model_data["tokenizer"]
        is_enhanced = False
        adapter_path = None

    # Enhance prompt with relevant components
    enhanced_messages = messages.copy()
    relevant_components = []
    if kb_parser.components_db and len(messages) > 0:
        user_content = messages[-1]["content"] if messages[-1]["role"] == "user" else ""
        relevant_components = kb_parser.get_relevant_components(user_content, max_results=3)

        if relevant_components:
            component_context = "\n\n🔧 Available Components:\n"
            for comp in relevant_components:
                name = comp.get('part_name', 'Unknown')
                comp_type = comp.get('component_type', 'N/A')
                price = comp.get('price_usd', 'N/A')
                desc = str(comp.get('description', 'No description'))[:100]

                component_context += f"• {name} ({comp_type}): {desc}... ${price}\n"

            # Wrap the component context for readability
            component_context = wrap_text(component_context, CONFIG.get("display", {}).get("line_width", 80))
            enhanced_messages[-1]["content"] += component_context

    # Prepare and wrap prompt
    try:
        prompt_text = tokenizer.apply_chat_template(enhanced_messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        system_content = enhanced_messages[0]["content"] if enhanced_messages[0]["role"] == "system" else ""
        user_content = enhanced_messages[-1]["content"] if enhanced_messages[-1]["role"] == "user" else ""
        prompt_text = f"{system_content}\n\n{user_content}\n\nAssistant:"

    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)

    # Get generation parameters from config
    gen_config = CONFIG.get("models", {}).get("generation", {})
    generation_config = GenerationConfig(
        do_sample=gen_config.get("do_sample", True),
        temperature=gen_config.get("temperature", 0.7),
        top_p=gen_config.get("top_p", 0.9),
        top_k=gen_config.get("top_k", 50),
        max_new_tokens=gen_config.get("max_new_tokens", 1200),
        repetition_penalty=gen_config.get("repetition_penalty", 1.05),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {**inputs, "generation_config": generation_config, "streamer": streamer}

    # Enhanced display with wallet status and visual indicators
    color = kb_parser.get_agent_color(agent_instance)
    agent_display = f"{agent_instance['agent_name']} ({agent_instance['role_type']})"
    if agent_instance["instance_number"] > 1:
        agent_display = f"{agent_instance['agent_name']}_{agent_instance['instance_number']} ({agent_instance['role_type']})"

    # Add indicators
    indicators = []

    # Speaker count
    speaker_count = 0  # Will be set by router after response
    indicators.append(f"#{speaker_count}")

    # Wallet status
    if wallet_manager.enabled:
        wallet = wallet_manager.get_wallet(agent_id)
        if wallet:
            indicators.append(wallet.get_display_status())

    # Enhancement indicators
    if is_enhanced:
        adapter_name = Path(adapter_path).name if adapter_path else "enhanced"
        indicators.append(f"🚀{adapter_name}")

    if relevant_components:
        indicators.append(f"🔧{len(relevant_components)}comp")

    agent_display += f" [{', '.join(indicators)}]"

    print(f"\n{color}┌─ {agent_display} ─{Colors.RESET}")
    print(f"{color}│{Colors.RESET} ", end="", flush=True)

    start_time = time.time()
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    response_parts = []
    line_length = 2  # Start after "│ "

    try:
        for new_text in streamer:
            for char in new_text:
                if char == '\n':
                    print(f"\n{color}│{Colors.RESET} ", end="", flush=True)
                    line_length = 2
                else:
                    print(f"{color}{char}{Colors.RESET}", end="", flush=True)
                    line_length += 1

                    # Wrap long lines
                    if line_length > CONFIG.get("display", {}).get("line_width", 80):
                        print(f"\n{color}│{Colors.RESET} ", end="", flush=True)
                        line_length = 2

            response_parts.append(new_text)
    except KeyboardInterrupt:
        print(f"\n{Colors.RED}│ [interrupted]{Colors.RESET}")
    finally:
        thread.join()

    generation_time = time.time() - start_time
    response = "".join(response_parts)

    # Charge wallet for response
    wallet_charged = wallet_manager.charge_for_response(agent_id, is_enhanced, generation_time)

    print(f"\n{color}└─ {generation_time:.1f}s ─{Colors.RESET}\n")

    # Extract metadata
    metadata = {
        "addressed_to": extract_addressees(response, kb_parser.active_team),
        "questions_asked": len(re.findall(r"[?]", response)),
        "references_made": extract_references(response, kb_parser.active_team),
        "domain_terms_used": count_domain_terms(response),
        "components_mentioned": extract_components_mentioned(response, kb_parser.components_db),
        "enhanced_model": is_enhanced,
        "adapter_used": adapter_path,
        "routing_method": context.get("routing_method", "unknown"),
        "speaker_count": context.get("speaker_count", 0),
        "wallet_charged": wallet_charged,
        "generation_cost": wallet_manager.get_wallet(agent_id).operation_costs.get("enhanced_response" if is_enhanced else "basic_response", 0) if wallet_manager.enabled else 0
    }

    return response, generation_time, metadata

def extract_addressees(text: str, team: List[Dict]) -> List[str]:
    """Extract agent names being addressed"""
    addressees = []
    for agent in team:
        agent_name = agent["agent_name"].lower()
        if re.search(rf"@{re.escape(agent_name)}\b", text.lower()):
            addressees.append(agent["agent_name"])

    return list(set(addressees))

def extract_references(text: str, team: List[Dict]) -> List[str]:
    """Extract references to other agents"""
    references = []
    for agent in team:
        agent_name = agent["agent_name"].lower()
        patterns = [
            rf"(?:as|like)\s+{re.escape(agent_name)}\s+(?:mentioned|said|noted)",
            rf"building\s+on\s+{re.escape(agent_name)}",
            rf"{re.escape(agent_name)}\s+(?:pointed out|suggested|proposed)"
        ]
        for pattern in patterns:
            if re.search(pattern, text.lower()):
                references.append(agent["agent_name"])
                break

    return list(set(references))

def count_domain_terms(text: str) -> int:
    """Count usage of domain-specific terminology"""
    text_lower = text.lower()
    count = sum(1 for term in kb_parser.domain_terminology if term.lower() in text_lower)
    return count

def extract_components_mentioned(text: str, components_db: List[Dict]) -> int:
    """Count component names mentioned in response"""
    if not components_db:
        return 0

    count = 0
    text_lower = text.lower()

    for component in components_db:
        part_name = str(component.get('part_name', '')).lower()
        if part_name and len(part_name) > 3 and part_name in text_lower:
            count += 1

    return count

# ===========================================
# Filename: src/multi_agent_core.py
# Tree Location: /project_root/src/multi_agent_core.py
# Creation Date: 2024-12-19 16:45:00 UTC
# ===========================================

# ===========================================
# Filename: src/multi_agent_orchestration.py
# Tree Location: /project_root/src/multi_agent_orchestration.py
# Version: 10.0
# Description: Agent coordination & orchestration with metabolic wallet constraints
# Dependencies: Requires Part 1 (multi_agent_core.py) to be imported first
# ===========================================

# ===========================================
# Smart Agent Router with Wallet-Aware Speaker Tracking
# ===========================================

class SmartAgentRouter:
    """Routes conversations to most appropriate agents using embedding similarity with wallet-aware speaker tracking"""

    def __init__(self, kb_parser):
        self.kb_parser = kb_parser
        self.agent_embeddings = None

        routing_config = CONFIG.get("routing", {})
        self.routing_mode = routing_config.get("routing_mode", "hybrid")
        self.round_robin_index = 0

        # Speaker tracking - executive control system
        self.speaker_stats = {}  # agent_id -> {count, last_turn, role_type, agent_name}
        self.recent_speakers = []  # Recent speaker queue for cooldown
        self.global_turn_count = 0
        self.cooldown_turns = routing_config.get("cooldown_turns", 2)

        self._init_speaker_tracking()
        self._cache_agent_embeddings()

    def _init_speaker_tracking(self):
        """Initialize speaker statistics for all agents"""
        print("🧠 Initializing speaker tracking system...")
        for agent in self.kb_parser.active_team:
            self.speaker_stats[agent["agent_id"]] = {
                "count": 0,
                "last_turn": -1,
                "role_type": agent["role_type"],
                "agent_name": agent["agent_name"],
                "instance_number": agent.get("instance_number", 1)
            }
        print(f"📊 Tracking {len(self.speaker_stats)} agents")

    def _cache_agent_embeddings(self):
        """Pre-compute embeddings for agent expertise"""
        try:
            print("🧠 Computing agent expertise embeddings...")

            agent_texts = []
            for agent in self.kb_parser.active_team:
                role_def = self.kb_parser.agent_roles.get(agent["role_type"], {})

                # Create expertise representation
                text_parts = [
                    agent["role_type"].replace("_", " "),
                    role_def.get("expertise", ""),
                    role_def.get("responsibilities", ""),
                    role_def.get("personality", "")
                ]
                agent_text = ' '.join([part for part in text_parts if part])
                agent_texts.append(agent_text)

            self.agent_embeddings = EMBEDDING_MODEL.encode(agent_texts)
            print(f"✅ Cached {len(agent_texts)} agent expertise embeddings")

        except Exception as e:
            print(f"⚠️  Failed to compute agent embeddings: {e}")
            self.agent_embeddings = None

    def update_speaker_activity(self, agent_instance: Dict):
        """Update tracking after agent speaks - executive memory update"""
        agent_id = agent_instance["agent_id"]
        self.global_turn_count += 1

        # Update statistics
        if agent_id in self.speaker_stats:
            self.speaker_stats[agent_id]["count"] += 1
            self.speaker_stats[agent_id]["last_turn"] = self.global_turn_count

        # Update recent speakers queue for cooldown
        self.recent_speakers.append(agent_id)
        if len(self.recent_speakers) > self.cooldown_turns:
            self.recent_speakers.pop(0)

        print(f"📊 Speaker update: {agent_instance['agent_name']} (#{self.speaker_stats[agent_id]['count']})")

    def get_available_agents(self, all_agents: List[Dict], enhanced_needed: bool = False) -> List[Dict]:
        """Filter agents based on cooldown AND wallet balance - dual constraint system"""
        available = []

        # First filter by cooldown
        for agent in all_agents:
            agent_id = agent["agent_id"]

            # Check cooldown
            if agent_id in self.recent_speakers:
                continue  # Agent in cooldown

            available.append(agent)

        # If everyone is in cooldown, reset and allow all
        if not available:
            print("🔄 All agents in cooldown - resetting")
            self.recent_speakers.clear()
            available = all_agents.copy()

        # Apply wallet constraints - metabolic selection pressure
        wallet_available = wallet_manager.get_available_agents(available, enhanced_needed)

        if len(wallet_available) < len(available):
            broke_agents = [a for a in available if a not in wallet_available]
            broke_names = [a["agent_name"] for a in broke_agents]
            print(f"💸 Resource-constrained agents: {', '.join(broke_names)}")

        # Final check - if wallet constraints eliminate everyone, allow base responses only
        if not wallet_available and enhanced_needed:
            print("🚨 No agents can afford enhanced responses - allowing base responses")
            return wallet_manager.get_available_agents(available, enhanced_needed=False)
        elif not wallet_available:
            print("🚨 All agents broke - emergency reset")
            # Emergency: give minimal credits to least-broke agent
            if wallet_manager.enabled and available:
                richest_agent = max(available, key=lambda a: wallet_manager.get_wallet(a["agent_id"]).balance if wallet_manager.get_wallet(a["agent_id"]) else 0)
                wallet = wallet_manager.get_wallet(richest_agent["agent_id"])
                if wallet:
                    wallet.earn("emergency_bailout", 1.0, "Emergency system bailout")
                    print(f"🆘 Emergency bailout: {richest_agent['agent_name']} received credits")
                return [richest_agent]

        return wallet_available

    def get_underutilized_agents(self, available_agents: List[Dict]) -> List[Dict]:
        """Find agents who have spoken least - fairness balancing"""
        if not available_agents:
            return []

        # Calculate average speaking frequency
        total_turns = sum(stats["count"] for stats in self.speaker_stats.values())
        avg_turns = total_turns / len(self.speaker_stats) if self.speaker_stats else 0

        # Find agents below or at average
        underutilized = []
        for agent in available_agents:
            agent_count = self.speaker_stats.get(agent["agent_id"], {}).get("count", 0)
            if agent_count <= avg_turns:
                underutilized.append(agent)

        if underutilized and len(underutilized) < len(available_agents):
            print(f"⚖️  Prioritizing {len(underutilized)} underutilized agents")
            return underutilized

        return available_agents

    def get_speaker_summary(self) -> str:
        """Get visual summary of speaker activity with wallet status"""
        if not self.speaker_stats:
            return "No speakers yet"

        # Sort by speak count descending
        sorted_speakers = sorted(
            self.speaker_stats.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )

        summary_parts = []
        for agent_id, stats in sorted_speakers:
            name = stats["agent_name"]
            if stats["instance_number"] > 1:
                name = f"{name}_{stats['instance_number']}"

            count = stats["count"]
            role = stats["role_type"].replace("_", " ").title()

            # Visual indicator based on activity
            if count == 0:
                indicator = "⚪"
            elif count <= 2:
                indicator = "🟡"
            elif count <= 4:
                indicator = "🟠"
            else:
                indicator = "🟢"

            # Add wallet status if enabled
            wallet_status = ""
            if wallet_manager.enabled:
                wallet = wallet_manager.get_wallet(agent_id)
                if wallet:
                    wallet_status = f":{wallet.get_display_status()}"

            summary_parts.append(f"{indicator} {name}({count}){wallet_status} [{role}]")

        return " | ".join(summary_parts)

    def find_addressed_agent(self, text: str) -> Optional[Dict]:
        """Find directly addressed agent (@AgentName)"""
        for agent in self.kb_parser.active_team:
            agent_name = agent["agent_name"].lower()
            if re.search(rf"@{re.escape(agent_name)}\b", text.lower()):
                return agent
        return None

    def find_expertise_match(self, context: str, available_agents: List[Dict]) -> Optional[Dict]:
        """Find agent with most relevant expertise from available agents"""
        if self.agent_embeddings is None or not available_agents:
            return None

        try:
            # Get indices of available agents in the main team list
            available_indices = []
            for agent in available_agents:
                for i, team_agent in enumerate(self.kb_parser.active_team):
                    if team_agent["agent_id"] == agent["agent_id"]:
                        available_indices.append(i)
                        break

            if not available_indices:
                return None

            # Embed the context
            context_embedding = EMBEDDING_MODEL.encode([context])

            # Calculate similarities only for available agents
            similarities = cosine_similarity(context_embedding, self.agent_embeddings)[0]
            available_similarities = [(idx, similarities[idx]) for idx in available_indices]

            # Find best match with minimum threshold
            routing_config = CONFIG.get("routing", {})
            threshold = routing_config.get("expertise_threshold", 0.4)

            best_idx, best_score = max(available_similarities, key=lambda x: x[1])
            if best_score > threshold:
                expert_agent = self.kb_parser.active_team[best_idx]
                print(f"🧠 Expertise match: {expert_agent['agent_name']} (score: {best_score:.2f})")
                return expert_agent

        except Exception as e:
            print(f"⚠️  Expertise matching failed: {e}")

        return None

    def get_next_agent(self, conversation_context: str, all_agents: List[Dict], enhanced_needed: bool = False) -> Dict:
        """Executive decision-making with wallet constraints - like prefrontal cortex resource allocation"""

        # 1. Apply dual constraints - cooldown + wallet filtering
        available_agents = self.get_available_agents(all_agents, enhanced_needed)

        print(f"🎯 Available agents: {len(available_agents)}/{len(all_agents)}")

        routing_config = CONFIG.get("routing", {})

        # 2. Check for direct addressing (top priority)
        if routing_config.get("enable_handoffs", True):
            addressed_agent = self.find_addressed_agent(conversation_context)
            if addressed_agent and addressed_agent in available_agents:
                print(f"📮 Direct address to: {addressed_agent['agent_name']}")
                return addressed_agent

        # 3. Fairness check - prioritize underutilized agents
        underutilized = self.get_underutilized_agents(available_agents)
        if len(underutilized) < len(available_agents):
            available_agents = underutilized

        # 4. Expertise matching within fair candidates
        if routing_config.get("expertise_routing", True):
            expert_agent = self.find_expertise_match(conversation_context, available_agents)
            if expert_agent:
                return expert_agent

        # 5. Round-robin fallback
        selected = self._round_robin_next(available_agents)
        print(f"🔄 Round-robin selection: {selected['agent_name']}")
        return selected

    def _round_robin_next(self, available_agents: List[Dict]) -> Dict:
        """Improved round-robin with available agents only"""
        if not available_agents:
            return self.kb_parser.active_team[0]

        agent = available_agents[self.round_robin_index % len(available_agents)]
        self.round_robin_index += 1
        return agent

# Initialize router
smart_router = SmartAgentRouter(kb_parser)

# ===========================================
# Enhanced Orchestrator with Wallet Economics and Manager-Led Flow
# ===========================================

class EnhancedOrchestrator:
    """Enhanced orchestrator with wallet economics, manager-led flow, proportional stages, stage cycling, and completion tracking"""

    def __init__(self):
        self.session_id = f"sess_{uuid.uuid4().hex[:8]}"
        self.conversation_history = []
        self.round_count = 0
        self.turn_count = 0
        self.team = kb_parser.active_team.copy()

        session_config = CONFIG.get("session", {})
        self.config = {
            "topic": kb_parser.get_project_topic(),
            "project_type": kb_parser.project_type,
            "timebox_minutes": session_config.get("timebox_minutes", 10),
            "max_rounds": session_config.get("max_rounds", 10),
            "domain_context": kb_parser.get_domain_context_summary(),
            "enable_introductions": session_config.get("enable_introductions", True),
            "adapters_enabled": CONFIG.get("adapters", {}).get("use_adapters", False),
            "routing_mode": CONFIG.get("routing", {}).get("routing_mode", "hybrid"),
            "stage_distribution": session_config.get("stage_distribution", {
                "design": 0.30,
                "validation": 0.30,
                "planning": 0.25,
                "summary": 0.15
            }),
            "enable_stage_cycling": session_config.get("enable_stage_cycling", True),
            "cycle_length": session_config.get("cycle_length", 8),
            "manager_leads": session_config.get("manager_leads", {
                "session_start": True,
                "round_start": True
            }),
            "wallet_enabled": wallet_manager.enabled
        }

        # Stage cycling tracking
        self.cycle_count = 0
        self.current_cycle_round = 0

        # Wallet economics tracking
        self.session_economics = {
            "total_earned": 0.0,
            "total_spent": 0.0,
            "completion_rewards": 0.0,
            "quality_bonuses": 0.0
        }

    def get_manager_agent(self) -> Dict:
        """Get the manager agent from the team"""
        for agent in self.team:
            if agent["role_type"] == "manager":
                return agent
        # Fallback to first agent if no manager defined
        return self.team[0]

    def get_current_stage_proportional(self, progress_pct: float) -> str:
        """Get current stage based on proportional distribution"""
        stage_dist = self.config["stage_distribution"]

        cumulative = 0
        for stage, percentage in stage_dist.items():
            cumulative += percentage
            if progress_pct <= cumulative:
                return stage

        # Fallback to summary if over 100%
        return "summary"

    def get_current_stage_cycling(self, round_num: int) -> str:
        """Get current stage with cycling for long sessions"""
        stages = ["design", "validation", "planning", "summary"]
        cycle_length = self.config["cycle_length"]

        # Determine position within cycle
        cycle_position = (round_num - 1) % cycle_length
        rounds_per_stage = cycle_length // len(stages)

        stage_index = min(cycle_position // rounds_per_stage, len(stages) - 1)
        return stages[stage_index]

    def get_current_stage(self, time_progress: float, round_progress: float) -> str:
        """Determine current stage using proportional distribution with optional cycling"""
        progress_pct = max(time_progress, round_progress)

        # For long sessions, use cycling
        if (self.config["enable_stage_cycling"] and
            self.config["max_rounds"] > 20):
            return self.get_current_stage_cycling(self.round_count)
        else:
            return self.get_current_stage_proportional(progress_pct)

    def display_system_dashboard(self):
        """Display comprehensive system dashboard with wallet economics"""
        print(f"\n{Colors.BOLD}📊 SYSTEM DASHBOARD{Colors.RESET}")
        print(f"🔄 Turn: {smart_router.global_turn_count} | Cooldown: {smart_router.cooldown_turns} turns")
        print(f"📈 {smart_router.get_speaker_summary()}")

        # Show recent speakers in cooldown
        if smart_router.recent_speakers:
            cooldown_names = []
            for agent_id in smart_router.recent_speakers:
                stats = smart_router.speaker_stats.get(agent_id, {})
                name = stats.get("agent_name", "Unknown")
                if stats.get("instance_number", 1) > 1:
                    name = f"{name}_{stats['instance_number']}"
                cooldown_names.append(name)
            print(f"❄️  Cooldown: {' → '.join(cooldown_names)}")

        # Wallet economics dashboard
        if wallet_manager.enabled:
            print(f"💰 {wallet_manager.get_wallet_summary()}")

        # Completion status
        completion_summary = kb_parser.completion_tracker.get_completion_summary()
        print(f"✅ {completion_summary}")

        print()

    def execute_introduction_phase(self):
        """Execute enhanced team introduction phase with wallet status"""
        if not self.config["enable_introductions"]:
            return

        print(f"\n{Colors.BOLD}{format_header('TEAM INTRODUCTIONS', '🎭', 70)}{Colors.RESET}\n")

        for agent_instance in self.team:
            color = kb_parser.get_agent_color(agent_instance)
            agent_name = agent_instance["agent_name"]
            role_type = agent_instance["role_type"]
            model_key = kb_parser.get_model_for_agent(agent_instance)

            if agent_instance["instance_number"] > 1:
                display_name = f"{agent_name}_{agent_instance['instance_number']}"
            else:
                display_name = agent_name

            introduction = kb_parser.get_introduction_text(agent_instance)

            # Enhancement and wallet indicators
            indicators = []

            # Check for adapter enhancement
            adapter_path = adapter_manager.get_adapter_path(model_key)
            if adapter_path:
                adapter_name = Path(adapter_path).name
                indicators.append(f"🚀{adapter_name}")

            # Wallet status
            if wallet_manager.enabled:
                wallet = wallet_manager.get_wallet(agent_instance["agent_id"])
                if wallet:
                    indicators.append(wallet.get_display_status())

            enhancement_text = f" [{', '.join(indicators)}]" if indicators else ""

            print(f"{color}👤 {display_name} - {role_type}{enhancement_text}{Colors.RESET}")
            wrapped_intro = wrap_text(introduction, CONFIG.get("display", {}).get("line_width", 80))
            print(f"{Colors.DIM}{wrapped_intro}{Colors.RESET}\n")

    def build_agent_prompt(self, agent_instance: Dict, stage: str, context: Dict) -> str:
        """Build enhanced contextual prompt for agent"""

        recent_turns = self.conversation_history[-6:] if self.conversation_history else []

        conversation_context = ""
        if recent_turns:
            conversation_context = "\n\n💬 Recent conversation:\n"
            for turn in recent_turns:
                speaker_name = turn.get("agent_name", "Unknown")
                role_type = turn.get("role_type", "")
                response = turn["response"][:150] + "..." if len(turn["response"]) > 150 else turn["response"]
                conversation_context += f"• {speaker_name} ({role_type}): {response}\n"

        domain_context = self.config["domain_context"]

        stage_instructions = {
            "design": "Focus on solution design, architecture, and technical approach.",
            "validation": "Validate feasibility, identify risks, and propose verification methods.",
            "planning": "Create actionable implementation plans, timelines, and resource requirements.",
            "summary": "Synthesize discussion into final recommendations and next steps."
        }

        team_context = "👥 Team members: " + ", ".join([
            f"{agent['agent_name']} ({agent['role_type']})" for agent in self.team
        ])

        time_remaining = max(0, context.get("deadline", time.time()) - time.time())
        time_min = int(time_remaining // 60)
        time_sec = int(time_remaining % 60)

        # Add cycling context for long sessions
        cycling_context = ""
        if (self.config["enable_stage_cycling"] and
            self.config["max_rounds"] > 20):
            cycling_context = f"\n🔄 Cycle: {self.cycle_count + 1} | Round {self.current_cycle_round + 1} in cycle"

        # Wallet economics context
        economics_context = ""
        if wallet_manager.enabled:
            wallet = wallet_manager.get_wallet(agent_instance["agent_id"])
            if wallet:
                status = wallet.get_status()
                economics_context = f"\n💰 Your resources: {status['balance']:.0f} credits"
                if status['balance'] < 30:
                    economics_context += " - Consider high-impact contributions"

        prompt = f"""
{format_header(f'ROUND {self.round_count} | STAGE: {stage.upper()}', '⚡', 70)}
⏰ Time: {time_min}m {time_sec}s{cycling_context}{economics_context}

🎯 Project: {self.config['topic']}
📁 Type: {self.config['project_type']}

{domain_context}

{team_context}

🎭 Your Role: You are {agent_instance['agent_name']}, working as {agent_instance['role_type']}.

📋 Stage Focus: {stage_instructions.get(stage, 'Continue the discussion')}

📜 Instructions:
1. If other team members asked you questions, answer them first
2. Provide substantive contribution for {stage} stage
3. Ask specific questions to other team members using their names (e.g., @{self.team[0]['agent_name']})
4. Reference previous contributions when building on them
5. Stay within project requirements and constraints
6. Be specific and actionable, avoid generic statements
7. Use available components from the database when relevant

{conversation_context}

🎯 Your response:
        """.strip()

        return wrap_text(prompt, CONFIG.get("display", {}).get("line_width", 80))

    def reward_quality_response(self, agent_id: str, response: str, metadata: Dict, validation: Dict):
        """Reward agents for quality responses with wallet credits"""
        if not wallet_manager.enabled:
            return

        wallet = wallet_manager.get_wallet(agent_id)
        if not wallet:
            return

        total_rewards = 0.0

        # Quality metrics for rewards
        quality_metrics = {
            "domain_terms_used": validation.get("domain_terms_used", 0),
            "components_mentioned": metadata.get("components_mentioned", 0),
            "questions_asked": metadata.get("questions_asked", 0),
            "references_made": len(metadata.get("references_made", [])),
            "addressed_to": len(metadata.get("addressed_to", []))
        }

        # Reward quality response attributes
        wallet_manager.reward_completion(agent_id, 0, quality_metrics)  # Use existing reward system

        # Track session economics
        reward_amount = sum([
            quality_metrics["domain_terms_used"] * wallet.reward_rates.get("domain_terms", 0),
            quality_metrics["components_mentioned"] * wallet.reward_rates.get("component_usage", 0)
        ])

        self.session_economics["quality_bonuses"] += reward_amount

    def update_session_economics(self):
        """Update session-wide economic tracking"""
        if not wallet_manager.enabled:
            return

        total_earned = 0.0
        total_spent = 0.0

        for agent_id, wallet in wallet_manager.wallets.items():
            status = wallet.get_status()
            total_earned += status["total_earned"]
            total_spent += status["total_spent"]

        self.session_economics["total_earned"] = total_earned
        self.session_economics["total_spent"] = total_spent

    def execute_session(self) -> str:
        """Execute complete enhanced session with wallet economics, manager-led flow, proportional stages, and completion tracking"""

        print(f"\n{Colors.BOLD}{format_header('SESSION START', '🚀', 70)}{Colors.RESET}")
        print(f"🆔 Session ID: {Colors.WHITE}{self.session_id}{Colors.RESET}")
        print(f"🎯 Project: {Colors.WHITE}{self.config['topic']}{Colors.RESET}")
        print(f"📁 Type: {Colors.WHITE}{self.config['project_type']}{Colors.RESET}")
        print(f"⏰ Timebox: {Colors.WHITE}{self.config['timebox_minutes']} minutes{Colors.RESET}")
        print(f"👥 Team Size: {Colors.WHITE}{len(self.team)} agents{Colors.RESET}")
        print(f"🧠 Routing: {Colors.WHITE}{self.config['routing_mode']}{Colors.RESET}")
        print(f"🔄 Cooldown: {Colors.WHITE}{smart_router.cooldown_turns} turns{Colors.RESET}")
        print(f"🚀 Adapters: {Colors.WHITE}{'Enabled' if self.config['adapters_enabled'] else 'Disabled'}{Colors.RESET}")
        print(f"👨‍💼 Manager-Led: {Colors.WHITE}{'Enabled' if self.config['manager_leads']['session_start'] else 'Disabled'}{Colors.RESET}")
        print(f"📊 Stage Distribution: {Colors.WHITE}{self.config['stage_distribution']}{Colors.RESET}")
        print(f"💰 Wallet Economics: {Colors.WHITE}{'Enabled' if self.config['wallet_enabled'] else 'Disabled'}{Colors.RESET}")

        # Session timing
        start_time = time.time()
        deadline = start_time + (self.config['timebox_minutes'] * 60)
        final_status = "completed"

        try:
            # Team introductions
            self.execute_introduction_phase()

            # Main collaboration phases
            while time.time() < deadline and self.round_count < self.config.get('max_rounds', 100):
                self.round_count += 1

                # Calculate progress
                time_progress = (time.time() - start_time) / (deadline - start_time)
                round_progress = self.round_count / self.config.get('max_rounds', 100)

                # Determine current stage
                stage = self.get_current_stage(time_progress, round_progress)

                # Update cycling tracking
                if self.config["enable_stage_cycling"]:
                    cycle_length = self.config["cycle_length"]
                    self.current_cycle_round = (self.round_count - 1) % cycle_length
                    self.cycle_count = (self.round_count - 1) // cycle_length

                # Display comprehensive dashboard
                self.display_system_dashboard()

                # Check if project is complete
                if kb_parser.completion_tracker.is_project_complete():
                    print(f"\n{Colors.GREEN}🎉 PROJECT COMPLETE - ENDING SESSION{Colors.RESET}")
                    break

                print(f"\n{Colors.BOLD}{format_header(f'ROUND {self.round_count} | STAGE: {stage.upper()}', '⚡', 70)}{Colors.RESET}\n")

                # Build conversation context for routing
                recent_context = " ".join([turn["response"] for turn in self.conversation_history[-3:]])

                # Manager-led flow
                if (self.config["manager_leads"]["session_start"] and self.round_count == 1) or \
                   (self.config["manager_leads"]["round_start"] and self.current_cycle_round == 0):
                    next_agent = self.get_manager_agent()
                    print(f"👨‍💼 Manager leading {'session' if self.round_count == 1 else 'round'}: {next_agent['agent_name']}")
                else:
                    next_agent = smart_router.get_next_agent(recent_context, self.team)

                # Smart agent selection for this round
                agents_spoken = set()

                # First agent is selected (manager or smart routing)
                agents_to_speak = min(3, len(self.team))
                for speak_turn in range(agents_to_speak):

                    if speak_turn > 0:  # After first agent, use normal routing
                        next_agent = smart_router.get_next_agent(recent_context, self.team)

                    if time.time() >= deadline:
                        break

                    # Build context
                    context = {
                        "deadline": deadline,
                        "stage": stage,
                        "round": self.round_count,
                        "previous_responses": [turn["response"] for turn in self.conversation_history[-3:]],
                        "routing_method": smart_router.routing_mode,
                        "speaker_count": smart_router.speaker_stats.get(next_agent["agent_id"], {}).get("count", 0) + 1
                    }

                    # Generate system prompt and user prompt
                    system_prompt = kb_parser.get_agent_system_prompt(next_agent)
                    user_prompt = self.build_agent_prompt(next_agent, stage, context)

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]

                    # Generate response with wallet constraints
                    turn_start = time.time()
                    response, generation_time, metadata = generate_streaming_response(next_agent, messages, context)
                    turn_end = time.time()

                    # Update speaker tracking AFTER response generation
                    smart_router.update_speaker_activity(next_agent)

                    # Check for completion items in response
                    completed_items = kb_parser.check_completion_in_response(
                        response, next_agent["agent_name"], next_agent["agent_id"]
                    )
                    if completed_items:
                        completion_reward = len(completed_items) * 50  # Reward amount
                        print(f"\n{Colors.GREEN}✅ COMPLETED: {len(completed_items)} criteria (+{completion_reward} credits){Colors.RESET}")
                        self.session_economics["completion_rewards"] += completion_reward

                    # Validation
                    validation = kb_parser.validate_against_requirements(response)

                    # Reward quality response
                    self.reward_quality_response(next_agent["agent_id"], response, metadata, validation)

                    # Enhanced validation display
                    req_indicators = ""
                    for key in validation.keys():
                        if key.endswith("_ok"):
                            req_indicators += Colors.CHECKMARK if validation.get(key, False) else Colors.CROSSMARK

                    # Wallet balance display
                    wallet_info = ""
                    if wallet_manager.enabled:
                        wallet = wallet_manager.get_wallet(next_agent["agent_id"])
                        if wallet:
                            status = wallet.get_status()
                            wallet_info = f"💰 {status['balance']:.0f} | "

                    print(f"{Colors.BOLD}📊 VALIDATION{Colors.RESET} {wallet_info}REQ: {req_indicators or 'N/A'} | "
                          f"🚫 Banned: {validation.get('banned_terms', 0)} | "
                          f"🎯 Domain: {validation.get('domain_terms_used', 0)} | "
                          f"🔧 Components: {metadata.get('components_mentioned', 0)} | "
                          f"✅ Completed: {len(completed_items)}")

                    # Log turn with wallet data
                    self.turn_count += 1
                    speaker_name = next_agent["agent_name"]
                    if next_agent["instance_number"] > 1:
                        speaker_name = f"{speaker_name}_{next_agent['instance_number']}"

                    # Get wallet data for logging
                    wallet_data = {}
                    if wallet_manager.enabled:
                        wallet = wallet_manager.get_wallet(next_agent["agent_id"])
                        if wallet:
                            wallet_status = wallet.get_status()
                            wallet_data = {
                                "wallet_balance": wallet_status["balance"],
                                "wallet_earned": wallet_status["total_earned"],
                                "wallet_spent": wallet_status["total_spent"],
                                "generation_cost": metadata.get("generation_cost", 0),
                                "wallet_charged": metadata.get("wallet_charged", False)
                            }

                    turn_data = {
                        "turn_id": f"turn_{uuid.uuid4().hex[:8]}",
                        "session_id": self.session_id,
                        "round_num": self.round_count,
                        "stage": stage,
                        "agent_id": next_agent["agent_id"],
                        "agent_name": speaker_name,
                        "role_type": next_agent["role_type"],
                        "model_id": kb_parser.get_model_for_agent(next_agent),
                        "device_used": str(DEVICE),
                        "turn_start_ts": datetime.fromtimestamp(turn_start, timezone.utc).isoformat(),
                        "turn_end_ts": datetime.fromtimestamp(turn_end, timezone.utc).isoformat(),
                        "duration_ms": int(generation_time * 1000),
                        "time_remaining_sec": int(max(0, deadline - turn_start)),
                        "response": response,
                        "banned_terms": validation.get("banned_terms", 0),
                        "questions_asked": metadata.get("questions_asked", 0),
                        "conversation_state": stage,
                        "enhanced_model": metadata.get("enhanced_model", False),
                        "adapter_used": metadata.get("adapter_used", ""),
                        "components_mentioned": metadata.get("components_mentioned", 0),
                        "routing_method": metadata.get("routing_method", "unknown"),
                        "addressed_to": ",".join(metadata.get("addressed_to", [])),
                        "speaker_count": metadata.get("speaker_count", 0),
                        "completion_items_done": len(completed_items),
                        "project_completion_pct": kb_parser.completion_tracker.get_completion_percentage(),
                        "cycle_count": self.cycle_count,
                        "cycle_round": self.current_cycle_round,
                        **wallet_data  # Include wallet data in turn log
                    }

                    # Add dynamic requirement validation fields
                    for key, value in validation.items():
                        if key.endswith("_ok"):
                            turn_data[key] = int(value)
                        elif key in ["domain_terms_used"]:
                            turn_data[key] = value

                    if CONFIG.get("display", {}).get("save_conversations", True):
                        log_to_csv("turns", turn_data)

                    # Store in conversation history
                    self.conversation_history.append({
                        "agent_id": next_agent["agent_id"],
                        "agent_name": speaker_name,
                        "role_type": next_agent["role_type"],
                        "response": response,
                        "metadata": metadata,
                        "turn_id": turn_data["turn_id"]
                    })

                    # Update context for next routing decision
                    recent_context = response
                    agents_spoken.add(next_agent["agent_id"])

                # Update session economics after each round
                self.update_session_economics()

        except Exception as e:
            print(f"{Colors.RED}💥 ERROR: Session failed: {e}{Colors.RESET}")
            final_status = f"error: {e}"

        # Final completion report
        print(f"\n{Colors.BOLD}📋 FINAL COMPLETION REPORT{Colors.RESET}")
        completion_report = kb_parser.completion_tracker.get_detailed_completion_report()
        print(completion_report)

        if kb_parser.completion_tracker.is_project_complete():
            print(f"\n{Colors.GREEN}🎉 ALL CRITERIA COMPLETED!{Colors.RESET}")
        else:
            incomplete_count = len(kb_parser.completion_tracker.get_incomplete_items())
            print(f"\n{Colors.YELLOW}⚠️  {incomplete_count} criteria remaining{Colors.RESET}")

        # Final wallet economics report
        if wallet_manager.enabled:
            print(f"\n{Colors.BOLD}💰 FINAL WALLET ECONOMICS{Colors.RESET}")
            print(wallet_manager.get_detailed_wallet_report())

            economics = self.session_economics
            print(f"📊 Session Economics:")
            print(f"  💰 Total Earned: {economics['total_earned']:.0f} credits")
            print(f"  💸 Total Spent: {economics['total_spent']:.0f} credits")
            print(f"  ✅ Completion Rewards: {economics['completion_rewards']:.0f} credits")
            print(f"  🎯 Quality Bonuses: {economics['quality_bonuses']:.0f} credits")

            net_change = economics['total_earned'] - economics['total_spent']
            print(f"  📈 Net Economic Change: {net_change:+.0f} credits")

        print(f"\n{Colors.BOLD}📊 FINAL SPEAKER SUMMARY{Colors.RESET}")
        print(f"📈 {smart_router.get_speaker_summary()}")

        # Session completion
        end_time = time.time()
        duration = (end_time - start_time) / 60
        completion_pct = kb_parser.completion_tracker.get_completion_percentage()

        print(f"\n{Colors.BOLD}{format_header('SESSION COMPLETE', '🎯', 70)}{Colors.RESET}")
        print(f"🆔 ID: {Colors.WHITE}{self.session_id}{Colors.RESET}")
        print(f"✅ Status: {Colors.GREEN if final_status == 'completed' else Colors.RED}{final_status}{Colors.RESET}")
        print(f"⏰ Duration: {Colors.WHITE}{duration:.1f}m / {self.config['timebox_minutes']}m{Colors.RESET}")
        print(f"🔄 Rounds: {Colors.WHITE}{self.round_count}{Colors.RESET}")
        print(f"💬 Turns: {Colors.WHITE}{self.turn_count}{Colors.RESET}")
        print(f"📊 Total Speaker Actions: {Colors.WHITE}{smart_router.global_turn_count}{Colors.RESET}")
        print(f"📋 Project Completion: {Colors.WHITE}{completion_pct:.1f}%{Colors.RESET}")

        if self.config["enable_stage_cycling"]:
            print(f"🔄 Cycles Completed: {Colors.WHITE}{self.cycle_count + 1}{Colors.RESET}")

        if wallet_manager.enabled:
            economics = self.session_economics
            print(f"💰 Economic Efficiency: {Colors.WHITE}{economics['total_earned']:.0f}E / {economics['total_spent']:.0f}S{Colors.RESET}")

        return self.session_id

# ===========================================
# System Initialization and Execution
# ===========================================

print(f"\n{Colors.BOLD}{format_header('ENHANCED MULTI-AGENT SYSTEM v10.0 READY', '🚀', 70)}{Colors.RESET}")
print(f"📁 Project Type: {Colors.WHITE}{kb_parser.project_type}{Colors.RESET}")
print(f"🎯 Topic: {Colors.WHITE}{kb_parser.get_project_topic()}{Colors.RESET}")
print(f"📋 Requirements: {Colors.WHITE}{list(kb_parser.requirements.keys())}{Colors.RESET}")
print(f"👥 Agent Roles: {Colors.WHITE}{list(kb_parser.agent_roles.keys())}{Colors.RESET}")
print(f"🔧 Components Available: {Colors.WHITE}{len(kb_parser.components_db)}{Colors.RESET}")
print(f"📊 Completion Criteria: {Colors.WHITE}{len(kb_parser.completion_tracker.completion_items)}{Colors.RESET}")
print(f"🧠 Routing Mode: {Colors.WHITE}{smart_router.routing_mode}{Colors.RESET}")
print(f"💰 Metabolic Constraints: {Colors.WHITE}{'Active' if wallet_manager.enabled else 'Disabled'}{Colors.RESET}")

# Main execution function
def run_enhanced_session():
    """Main function to execute an enhanced multi-agent session"""
    try:
        orchestrator = EnhancedOrchestrator()
        session_id = orchestrator.execute_session()
        print(f"\n{Colors.GREEN}🎉 Session completed successfully!{Colors.RESET}")
        return session_id
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️  Session interrupted by user{Colors.RESET}")
        return None
    except Exception as e:
        print(f"\n{Colors.RED}❌ Session failed: {e}{Colors.RESET}")
        return None

# ===========================================
# CLI Interface and Main Execution
# ===========================================

def main():
    """Main entry point for the enhanced multi-agent system"""
    try:
        print(f"\n{Colors.BOLD}🚀 STARTING ENHANCED MULTI-AGENT SESSION{Colors.RESET}")

        # Validate prerequisites
        if not MODEL_CACHE:
            print(f"{Colors.RED}❌ ERROR: No models loaded. Run GPU loader first.{Colors.RESET}")
            return

        if not kb_parser.active_team:
            print(f"{Colors.RED}❌ ERROR: No active team configured.{Colors.RESET}")
            return

        # Execute session
        session_id = run_enhanced_session()

        if session_id:
            print(f"\n{Colors.GREEN}✅ Session {session_id} completed successfully{Colors.RESET}")
        else:
            print(f"\n{Colors.RED}❌ Session failed or was interrupted{Colors.RESET}")

    except Exception as e:
        print(f"\n{Colors.RED}💥 FATAL ERROR: {e}{Colors.RESET}")
    finally:
        print(f"\n{Colors.DIM}Thank you for using Enhanced Multi-Agent System v10.0{Colors.RESET}")

# ===========================================
# Interactive Mode Functions with Wallet Features
# ===========================================

def show_system_status():
    """Display current system status and configuration with wallet status"""
    print(f"\n{Colors.BOLD}📊 SYSTEM STATUS{Colors.RESET}")
    print(f"📦 Models loaded: {len(MODEL_CACHE)}")
    print(f"👥 Active agents: {len(kb_parser.active_team)}")
    print(f"🔧 Components: {len(kb_parser.components_db)}")
    print(f"📋 Completion items: {len(kb_parser.completion_tracker.completion_items)}")
    print(f"🧠 Embeddings: {'✅' if kb_parser.component_embeddings is not None else '❌'}")
    print(f"🚀 Adapters: {'✅' if CONFIG.get('adapters', {}).get('use_adapters', False) else '❌'}")
    print(f"💾 Data logging: {'✅' if CONFIG.get('display', {}).get('save_conversations', True) else '❌'}")
    print(f"💰 Wallet system: {'✅' if wallet_manager.enabled else '❌'}")

    if wallet_manager.enabled:
        print(f"💳 Wallet summary: {wallet_manager.get_wallet_summary()}")

def run_test_session():
    """Run a short test session for validation"""
    print(f"\n{Colors.BOLD}🧪 RUNNING TEST SESSION{Colors.RESET}")

    # Override config for test
    original_timebox = CONFIG.get("session", {}).get("timebox_minutes", 10)
    original_max_rounds = CONFIG.get("session", {}).get("max_rounds", 10)

    CONFIG["session"]["timebox_minutes"] = 2
    CONFIG["session"]["max_rounds"] = 3

    try:
        session_id = run_enhanced_session()
        print(f"\n{Colors.GREEN}✅ Test session completed: {session_id}{Colors.RESET}")
    finally:
        # Restore original config
        CONFIG["session"]["timebox_minutes"] = original_timebox
        CONFIG["session"]["max_rounds"] = original_max_rounds

def get_completion_report():
    """Get current completion status report"""
    print(f"\n{Colors.BOLD}📋 COMPLETION STATUS REPORT{Colors.RESET}")
    report = kb_parser.completion_tracker.get_detailed_completion_report()
    print(report)

    summary = kb_parser.completion_tracker.get_completion_summary()
    print(f"\n📊 Summary: {summary}")

def get_wallet_report():
    """Get current wallet economics report"""
    if not wallet_manager.enabled:
        print(f"\n{Colors.YELLOW}💸 Wallet system is disabled{Colors.RESET}")
        return

    print(f"\n{Colors.BOLD}💰 WALLET ECONOMICS REPORT{Colors.RESET}")
    detailed_report = wallet_manager.get_detailed_wallet_report()
    print(detailed_report)

def reset_wallets():
    """Reset all wallet balances to starting amounts"""
    if not wallet_manager.enabled:
        print(f"\n{Colors.YELLOW}💸 Wallet system is disabled{Colors.RESET}")
        return

    wallet_config = CONFIG.get("wallet", {})
    starting_balance = wallet_config.get("starting_balance", 100.0)

    for agent_id, wallet in wallet_manager.wallets.items():
        wallet.balance = starting_balance
        wallet.total_earned = 0.0
        wallet.total_spent = 0.0
        wallet.transaction_log.clear()

    print(f"\n{Colors.GREEN}💳 All wallets reset to {starting_balance} credits{Colors.RESET}")

def list_available_commands():
    """List all available interactive commands"""
    commands = {
        "run_enhanced_session()": "Execute a full multi-agent session with wallet economics",
        "show_system_status()": "Display current system configuration and wallet status",
        "run_test_session()": "Run a short validation session",
        "get_completion_report()": "Show project completion status",
        "get_wallet_report()": "Show detailed wallet economics (if enabled)",
        "reset_wallets()": "Reset all agent wallets to starting balance",
        "list_available_commands()": "Show this help message",
        "main()": "Main entry point with full validation"
    }

    print(f"\n{Colors.BOLD}📚 AVAILABLE COMMANDS{Colors.RESET}")
    for cmd, desc in commands.items():
        print(f"{Colors.CYAN}{cmd:<25}{Colors.RESET} - {desc}")

# ===========================================
# Execute if run as main script
# ===========================================

if __name__ == "__main__":
    main()

# ===========================================
# Filename: src/multi_agent_orchestration.py
# Tree Location: /project_root/src/multi_agent_orchestration.py
# Creation Date: 2024-12-19 16:45:00 UTC
# ===========================================
