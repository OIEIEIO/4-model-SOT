# 4-Model-SOT: Source of Truth

Multi-Agent System with Metabolic Wallet Constraints - Tarinn mouthpiece

Enhanced multi-agent cargo drone design system using 4 specialized models with metabolic resource constraints.

## System Architecture

### 4-Model Configuration
- **Technical Pool** (Qwen/Qwen2.5-1.5B-Instruct): Electrical, Mechanical, Flight Controls Engineers
- **Practical Pool** (meta-llama/Llama-3.2-1B-Instruct): Test Engineer
- **Strategic Pool** (google/gemma-2-2b-it): Systems Engineer, Manager
- **Reasoning Pool** (h2oai/h2o-danube2-1.8b-base): Reasoning Specialist

### 7-Agent Team
1. **Eel** - Electrical Engineer (technical pool)
2. **Bear** - Mechanical Engineer (technical pool)
3. **Falcon** - Flight Controls Engineer (technical pool)
4. **Beaver** - Systems Engineer (strategic pool)
5. **Mongoose** - Test Engineer (practical pool)
6. **Lion** - Manager (strategic pool)
7. **Crow** - Reasoning Specialist (reasoning pool)

## Wallet Economics System

### Metabolic Constraints
- Starting balance: 100 credits per agent
- Exponential decay: 10% per minute (metabolic consumption)
- Operation costs: Basic response (10), Enhanced model (25)
- Rewards: Completion items (50), Quality bonuses (variable)

### Economic Pressure
- Broke agents excluded from participation
- Enhanced models require higher resource investment
- Quality work rewarded through pattern matching
- Strategic resource allocation required for success

```git clone https://github.com/OIEIEIO/4-model-SOT```

```cd 4-model-SOT```

```python bootstrap_project.py``` - creates all files and folders for the project

### Below all components created by - bootstrap_project.py

📊 All components integrated with metabolic wallet constraints

```
(base) jorge@jorge-X99:~/4-model-SOT$ tree
```

```
├── adapters
│   ├── checkpoints
│   └── lora
├── bootstrap_project.py
├── config.json
├── data
│   ├── completion_tracking.csv
│   ├── golden
│   ├── preferences
│   ├── preferences.csv
│   ├── quality_metrics.csv
│   ├── sessions
│   ├── sessions.csv
│   ├── turns.csv
│   ├── wallet_logs
│   └── wallet_transactions.csv
├── download_models.py
├── evals
│   ├── golden
│   │   └── golden_suite_v10.0.md
│   └── regression
├── kb
│   ├── acceptance.md
│   ├── agents.md
│   ├── components.csv
│   ├── design_notes.md
│   ├── SOT.md
│   └── tests_template.md
├── models
│   ├── base
│   └── cache
├── README.md
├── requirements.txt
├── run10.py
├── runs
│   ├── artifacts
│   └── logs
├── src
│   └── core
└── VERSION.json

21 directories, 20 files
```
(base) jorge@jorge-X99:~/4-model-SOT$ 

```python run10.py``` - downloads all the models and does a 10 minute run
