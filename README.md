# 4-Model-SOT: Collaborative AI with Economic Constraints

**Tarinn Neural Net - Multi-Agent Intelligence Platform**

A framework demonstrating how multiple AI models can collaborate under resource constraints to solve complex problems, moving beyond single-model limitations toward team-based artificial intelligence.

## Core Innovation: Economic Collaboration

Traditional AI systems use one model per task. 4-Model-SOT creates AI teams where different models play specialized roles, compete for limited resources, and must collaborate to succeed. Economic pressure forces strategic thinking and quality work - just like human teams.

## What Makes This Different

- **Role Specialization**: Models optimized for technical, practical, strategic, and reasoning tasks
- **Resource Scarcity**: Metabolic wallet system creates realistic constraints and incentives  
- **Iterative Improvement**: Models learn from collaboration and adapt over time
- **Measurable Outcomes**: Track collaboration effectiveness, not just individual performance

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

### Wallet Economics System

#### Metabolic Constraints
- Starting balance: 100 credits per agent
- Exponential decay: 10% per minute (metabolic consumption)
- Operation costs: Basic response (10), Enhanced model (25)
- Rewards: Completion items (50), Quality bonuses (variable)

#### Economic Pressure
- Broke agents excluded from participation
- Enhanced models require higher resource investment
- Quality work rewarded through pattern matching
- Strategic resource allocation required for success

## Real-World Applications

- **Engineering Design**: Complex systems requiring multiple disciplines
- **Research Projects**: Teams analyzing data from different perspectives  
- **Business Strategy**: Multi-stakeholder decision making with resource constraints
- **Creative Collaboration**: AI teams working on artistic or innovative projects
- **Educational Simulation**: Teaching collaboration and resource management

## The Collaboration Advantage

Human expertise emerges from teams, not individuals. 4-Model-SOT applies this principle to AI - creating systems that can tackle problems requiring diverse thinking patterns, cross-disciplinary knowledge, and strategic resource allocation.

## Beyond Cargo Drones

While demonstrated through drone design, the framework generalizes to any collaborative problem. The economic constraints and role-based architecture create emergent behaviors: quality over quantity, strategic thinking, and genuine cooperation between AI agents.

## Growth Trajectory

This represents early steps toward AI systems that collaborate, compete, and improve together - moving from isolated models toward artificial intelligence that works more like human teams, with the mathematical precision of Tarinn's neural architectures.

## Quick Start
```bash
git clone https://github.com/OIEIEIO/4-model-SOT
cd 4-model-SOT
python bootstrap_project.py  # Creates all files and folders for the project
python run10.py             # Downloads models and runs 10-minute demo
```
Project Structure
Below shows all components created by bootstrap_project.py:
```
├── adapters
│   ├── checkpoints
│   └── lora
├── bootstrap_project.py
├── config.json
├── data
│   ├── completion_tracking.csv
│   ├── golden
│   ├── preferences
│   ├── preferences.csv
│   ├── quality_metrics.csv
│   ├── sessions
│   ├── sessions.csv
│   ├── turns.csv
│   ├── wallet_logs
│   └── wallet_transactions.csv
├── download_models.py
├── evals
│   ├── golden
│   │   └── golden_suite_v10.0.md
│   └── regression
├── kb
│   ├── acceptance.md
│   ├── agents.md
│   ├── components.csv
│   ├── design_notes.md
│   ├── SOT.md
│   └── tests_template.md
├── models
│   ├── base
│   └── cache
├── README.md
├── requirements.txt
├── runs
│   ├── artifacts
│   └── logs
├── src
│   └── core
└── VERSION.json
```
All components integrated with metabolic wallet constraints.

Built by Tarinn Neural Net - Advancing collaborative artificial intelligence through economic constraints and specialized model architectures.
