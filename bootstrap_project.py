#@title Bootstrap 4-Model-SOT Multi-Agent System with Wallet Economics
# ===========================================
# File: bootstrap_project.py
# Tree: ~/4-model-SOT/bootstrap_project.py
# Version: 10.0
# Complete project bootstrapper for wallet-integrated multi-agent system
# ===========================================

import os
import sys
import json
import csv
import shutil
import textwrap
from pathlib import Path
from datetime import datetime

print("[bootstrap] Initializing Enhanced 4-Model-SOT Multi-Agent System v10.0 with Metabolic Constraints...")

# ===========================================
# Project Structure Creation
# ===========================================

# Base project directory (current working directory)
PROJECT_ROOT = Path.cwd()
print(f"[bootstrap] Project root: {PROJECT_ROOT}")

# Define comprehensive directory structure
DIRECTORIES = {
    # Core directories
    "kb": PROJECT_ROOT / "kb",                    # Knowledge base files
    "data": PROJECT_ROOT / "data",                # CSV logs and datasets
    "runs": PROJECT_ROOT / "runs",                # Session artifacts
    "adapters": PROJECT_ROOT / "adapters",        # Fine-tuned model adapters
    "models": PROJECT_ROOT / "models",            # Downloaded base models
    "evals": PROJECT_ROOT / "evals",              # Evaluation suites
    "src": PROJECT_ROOT / "src",                  # Source code modules

    # Subdirectories for organization
    "data/sessions": PROJECT_ROOT / "data" / "sessions",
    "data/wallet_logs": PROJECT_ROOT / "data" / "wallet_logs",
    "data/preferences": PROJECT_ROOT / "data" / "preferences",
    "data/golden": PROJECT_ROOT / "data" / "golden",
    "models/cache": PROJECT_ROOT / "models" / "cache",
    "models/base": PROJECT_ROOT / "models" / "base",
    "adapters/lora": PROJECT_ROOT / "adapters" / "lora",
    "adapters/checkpoints": PROJECT_ROOT / "adapters" / "checkpoints",
    "evals/golden": PROJECT_ROOT / "evals" / "golden",
    "evals/regression": PROJECT_ROOT / "evals" / "regression",
    "runs/artifacts": PROJECT_ROOT / "runs" / "artifacts",
    "runs/logs": PROJECT_ROOT / "runs" / "logs",
    "src/core": PROJECT_ROOT / "src" / "core"
}

# Create all directories
print("[bootstrap] Creating directory structure...")
for name, path in DIRECTORIES.items():
    path.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ {name}: {path.relative_to(PROJECT_ROOT)}")

# ===========================================
# Enhanced Configuration File
# ===========================================

print("[bootstrap] Creating enhanced configuration file...")

config = {
    "metadata": {
        "project_name": "4-model-SOT",
        "config_version": "10.0",
        "created": datetime.now().isoformat(),
        "description": "Multi-agent cargo drone design system with metabolic wallet constraints"
    },

    "models": {
        "model_pools": {
            "technical": "architect",
            "practical": "ops",
            "strategic": "manager",
            "reasoning": "reasoning"
        },
        "generation": {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "max_new_tokens": 1200,
            "repetition_penalty": 1.05
        }
    },

    "agents": {
        "active_team": [
            {"role_type": "electrical_engineer", "agent_name": "Eel"},
            {"role_type": "mechanical_engineer", "agent_name": "Bear"},
            {"role_type": "flight_controls_engineer", "agent_name": "Falcon"},
            {"role_type": "systems_engineer", "agent_name": "Beaver"},
            {"role_type": "test_engineer", "agent_name": "Mongoose"},
            {"role_type": "manager", "agent_name": "Lion"},
            {"role_type": "reasoning_specialist", "agent_name": "Crow"}
        ]
    },

    "session": {
        "timebox_minutes": 10,
        "max_rounds": 12,
        "enable_introductions": True,
        "stage_distribution": {
            "design": 0.30,
            "validation": 0.30,
            "planning": 0.25,
            "summary": 0.15
        },
        "enable_stage_cycling": True,
        "cycle_length": 8,
        "manager_leads": {
            "session_start": True,
            "round_start": True
        }
    },

    "routing": {
        "routing_mode": "hybrid",
        "cooldown_turns": 2,
        "enable_handoffs": True,
        "expertise_routing": True,
        "expertise_threshold": 0.4
    },

    "wallet": {
        "enabled": True,
        "starting_balance": 100.0,
        "decay_rate": 0.1,
        "min_balance": 5.0,
        "max_balance": 500.0,
        "costs": {
            "basic_response": 10.0,
            "enhanced_response": 25.0,
            "component_search": 5.0,
            "priority_routing": 15.0
        },
        "rewards": {
            "completion_item": 50.0,
            "peer_verification": 20.0,
            "domain_terms": 2.0,
            "component_usage": 5.0,
            "quality_bonus": 10.0
        }
    },

    "adapters": {
        "use_adapters": False,
        "adapters_path": "adapters/lora",
        "adapter_mode": "latest"
    },

    "knowledge_base": {
        "kb_directory": "kb",
        "components_file": "kb/components.csv"
    },

    "data": {
        "data_directory": "data",
        "save_conversations": True
    },

    "display": {
        "line_width": 80,
        "verbose_logging": False,
        "save_conversations": True,
        "colors": {
            "technical": "cyan",
            "practical": "yellow",
            "strategic": "green",
            "reasoning": "magenta"
        }
    }
}

config_path = PROJECT_ROOT / "config.json"
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
print("  ✓ config.json created with wallet economics")

# ===========================================
# Knowledge Base Initialization
# ===========================================

KB_FILES = {
    "sot": DIRECTORIES["kb"] / "SOT.md",
    "acceptance": DIRECTORIES["kb"] / "acceptance.md",
    "notes": DIRECTORIES["kb"] / "design_notes.md",
    "tests": DIRECTORIES["kb"] / "tests_template.md",
    "agents": DIRECTORIES["kb"] / "agents.md",
    "components": DIRECTORIES["kb"] / "components.csv"
}

print("[bootstrap] Initializing knowledge base...")

# Create SOT.md
if not KB_FILES["sot"].exists():
    sot_content = textwrap.dedent("""\
        # Source of Truth (SOT) v1.0

        ## Mission Requirements
        Design a short-range battery-electric cargo drone:
        - **Payload**: 10 kg
        - **Range**: 30 km
        - **Unit BOM**: ≤ $8,000
        - **Landing**: Rough-terrain capable (no runway required)
        - **Propulsion**: Battery-electric only
        - **Flight Duration**: 45 minutes minimum
        - **Operating Temperature**: -10°C to +40°C

        ## Constraints
        - No hybrid systems permitted
        - Commercial off-the-shelf components preferred
        - Must survive rough landing environments
        - IP54 minimum environmental protection
        - Maximum takeoff weight: 25 kg
        - Autonomous operation required

        ## Success Metrics
        - Payload delivery within range specification
        - Cost target achievement with 10% margin
        - Successful rough terrain landing demonstration
        - Field test completion per test protocols
        - Mission success rate >95% over 50 flights

        ## Banned Terms
        hybrid, fuel tank, combustion, gasoline, kerosene, ICE, jet fuel

        ## Version History
        - v1.0: Initial requirements (2025-12-19)
        """).strip()

    KB_FILES["sot"].write_text(sot_content, encoding="utf-8")
    print("  ✓ SOT.md created")

# Create agents.md with 7-agent definitions
if not KB_FILES["agents"].exists():
    agents_content = textwrap.dedent("""\
        # Agent Role Definitions v10.0

        ## electrical_engineer

        **Model Pool**: technical

        **Personality**: Analytical, detail-oriented, safety-focused, power-systems expert

        **Responsibilities**:
        - Battery management system design and integration
        - Motor controller selection and configuration
        - Power distribution architecture and wiring harness
        - Electrical safety protocols and fault protection
        - Component sizing and electrical load analysis
        - Charging system design and ground support equipment

        **Expertise**: Electrical systems design, battery chemistry, motor control systems, power electronics, electrical safety standards, EMC compliance

        **Collaborates With**: mechanical_engineer (integration), flight_controls_engineer (avionics), systems_engineer (power requirements), test_engineer (electrical testing)

        **Introduction Template**: "I'm {agent_name}, the electrical engineer responsible for all electrical systems including battery management, motor controllers, and power distribution. I ensure electrical safety and optimize power system efficiency."

        ## mechanical_engineer

        **Model Pool**: technical

        **Personality**: Practical, structural-focused, durability-minded, materials expert

        **Responsibilities**:
        - Structural design and finite element analysis
        - Landing gear systems for rough terrain capability
        - Payload attachment and release mechanisms
        - Materials selection and weight optimization
        - Vibration isolation and structural damping
        - Airframe design and manufacturing specifications

        **Expertise**: Structural engineering, materials science, mechanical design, vibration analysis, manufacturing processes, CAD modeling

        **Collaborates With**: electrical_engineer (component integration), flight_controls_engineer (control surfaces), systems_engineer (weight budgets), test_engineer (structural testing)

        **Introduction Template**: "I'm {agent_name}, the mechanical engineer handling structural design, landing gear systems, and payload mechanisms. I ensure the drone survives rough terrain operations while maintaining structural integrity."

        ## flight_controls_engineer

        **Model Pool**: technical

        **Personality**: Precision-focused, algorithm-minded, safety-critical thinking, autonomous systems expert

        **Responsibilities**:
        - Autopilot system architecture and implementation
        - Flight control algorithms and stability analysis
        - Navigation systems and path planning
        - Abort procedures and emergency protocols
        - Sensor fusion and state estimation
        - Flight envelope protection and safety systems

        **Expertise**: Flight dynamics, control system design, navigation algorithms, autopilot systems, sensor fusion, flight testing

        **Collaborates With**: electrical_engineer (sensor integration), mechanical_engineer (control authority), systems_engineer (performance requirements), test_engineer (flight testing)

        **Introduction Template**: "I'm {agent_name}, the flight controls engineer developing autopilot systems, navigation algorithms, and safety protocols. I ensure stable autonomous flight and reliable emergency procedures."

        ## systems_engineer

        **Model Pool**: strategic

        **Personality**: Integrative, holistic-thinking, trade-off focused, requirements-driven

        **Responsibilities**:
        - Requirements integration and system architecture
        - System-level trade studies and optimization
        - Interface management between subsystems
        - Performance verification and validation
        - Configuration control and change management
        - Risk assessment and mitigation planning

        **Expertise**: Systems engineering, requirements analysis, trade study methodology, interface management, verification and validation, risk management

        **Collaborates With**: electrical_engineer (power budgets), mechanical_engineer (weight analysis), flight_controls_engineer (performance requirements), test_engineer (integration testing)

        **Introduction Template**: "I'm {agent_name}, the systems engineer integrating all subsystem requirements and managing interfaces. I conduct trade studies to optimize overall system performance and ensure requirements compliance."

        ## test_engineer

        **Model Pool**: practical

        **Personality**: Methodical, validation-focused, risk-aware, empirical-minded

        **Responsibilities**:
        - Test protocol development and execution
        - Flight test planning and safety analysis
        - Data analysis and performance validation
        - Failure investigation and root cause analysis
        - Test equipment specification and setup
        - Certification and compliance testing

        **Expertise**: Test methodology, flight testing, data analysis, failure analysis, statistical analysis, test automation

        **Collaborates With**: electrical_engineer (electrical testing), mechanical_engineer (structural testing), flight_controls_engineer (flight testing), systems_engineer (validation requirements)

        **Introduction Template**: "I'm {agent_name}, the test engineer developing and executing comprehensive test protocols from component level through full flight testing. I validate system performance and investigate any failures."

        ## manager

        **Model Pool**: strategic

        **Personality**: Coordinating, deadline-driven, resource-focused, stakeholder-oriented, decision-facilitator

        **Responsibilities**:
        - Project coordination and timeline management
        - Resource allocation and budget oversight
        - Stakeholder communication and reporting
        - Risk management and issue resolution
        - Team coordination and workflow optimization
        - Quality assurance and deliverable approval

        **Expertise**: Project management, resource optimization, stakeholder coordination, risk assessment, technical program management, agile methodologies

        **Collaborates With**: electrical_engineer (technical status), mechanical_engineer (design progress), flight_controls_engineer (development status), systems_engineer (requirements status), test_engineer (validation progress)

        **Introduction Template**: "I'm {agent_name}, the project manager coordinating the development effort, managing timelines and resources, and ensuring we meet cargo drone objectives within budget and schedule."

        ## reasoning_specialist

        **Model Pool**: reasoning

        **Personality**: Analytical, systematic, logic-focused, verification-oriented, mathematically rigorous

        **Responsibilities**:
        - Mathematical verification and calculation validation
        - Logical consistency checking across disciplines
        - Algorithm analysis and computational verification
        - Risk assessment and failure mode analysis
        - Cross-disciplinary integration verification
        - Quality assurance for analytical work

        **Expertise**: Mathematical reasoning, logical problem solving, computational verification, analytical validation, statistical analysis, optimization theory

        **Collaborates With**: electrical_engineer (calculations verification), mechanical_engineer (structural analysis), flight_controls_engineer (algorithm validation), systems_engineer (trade study logic), test_engineer (data analysis)

        **Introduction Template**: "I'm {agent_name}, specializing in reasoning and logical validation. I verify calculations, check consistency across disciplines, and ensure mathematical rigor in all analytical work."
        """).strip()

    KB_FILES["agents"].write_text(agents_content, encoding="utf-8")
    print("  ✓ agents.md created with 7 agent roles")

# Create acceptance.md
if not KB_FILES["acceptance"].exists():
    acceptance_content = textwrap.dedent("""\
        # Acceptance Criteria v1.0

        ## Technical Validation
        - Flight envelope documented with 20% safety margins
        - Mass balance calculations verified by simulation
        - Powertrain sizing confirmed with thermal analysis
        - Control surface authority validated across flight regime
        - Battery management system safety certified
        - Electrical load analysis completed with margin assessment

        ## Environmental Testing
        - Rough terrain landing capability demonstrated
        - IP54 environmental protection verified
        - Wind resistance validated up to 25 km/h sustained
        - Temperature operation range: -10°C to +40°C confirmed
        - Vibration resistance per MIL-STD-810G demonstrated
        - EMC compliance testing completed

        ## Operational Readiness
        - Autonomous flight modes validated in all conditions
        - Emergency abort procedures tested and verified
        - Ground control station interface fully operational
        - Preflight checklist automated and validated
        - Maintenance procedures documented and tested
        - Operator training program completed and approved

        ## Performance Metrics
        - Payload capacity: 10 kg ± 0.5 kg verified
        - Range achievement: 30 km minimum demonstrated
        - Flight duration: 45 minutes minimum achieved
        - Landing accuracy: within 5m of target demonstrated
        - Mission success rate: >95% over 50 flight test cycles
        - Cost target: ≤$8,000 BOM with procurement documentation

        ## Safety and Compliance
        - Risk assessment completed and mitigation implemented
        - Safety of flight documentation approved
        - Regulatory compliance verified for operational area
        - Insurance and liability coverage confirmed
        - Emergency response procedures established and tested
        """).strip()

    KB_FILES["acceptance"].write_text(acceptance_content, encoding="utf-8")
    print("  ✓ acceptance.md created")

# Create design_notes.md
if not KB_FILES["notes"].exists():
    notes_content = textwrap.dedent("""\
        # Design Decision Ledger v1.0

        ## Decision Log

        ### 2025-12-19 - Initial Architecture Selection
        - **Decision**: Quadrotor configuration selected over fixed-wing or hexacopter
        - **Rationale**: Optimal balance of simplicity, redundancy, and rough terrain capability
        - **Trade-offs**: Power efficiency vs mechanical simplicity and cost
        - **Status**: Approved
        - **Impact**: Affects motor selection, flight controller design, and landing gear

        ### 2025-12-19 - Battery Chemistry Decision
        - **Decision**: LiFePO4 selected over Li-ion for primary battery
        - **Rationale**: Enhanced safety, longer cycle life, better temperature performance
        - **Trade-offs**: Weight penalty vs safety and operational flexibility
        - **Status**: Approved
        - **Impact**: Affects battery management system, charging infrastructure, weight budget

        ## Open Issues
        - Motor redundancy strategy (individual motor failure response)
        - Payload attachment standardization (universal vs mission-specific)
        - Ground control station mobility requirements
        - Weather monitoring integration for autonomous operations
        - Maintenance scheduling and predictive diagnostics implementation

        ## Technical Insights
        - Power consumption modeling indicates 15% margin above minimum requirements
        - Structural analysis shows stress concentration at motor mount interfaces
        - Control authority analysis reveals adequate margin in normal flight envelope
        - Thermal analysis identifies battery cooling as critical design driver

        ## Lessons Learned
        [Technical insights will be accumulated here through sessions]
        """).strip()

    KB_FILES["notes"].write_text(notes_content, encoding="utf-8")
    print("  ✓ design_notes.md created")

# Create tests_template.md
if not KB_FILES["tests"].exists():
    tests_content = textwrap.dedent("""\
        # Test Protocols v1.0

        ## Preflight Checklist
        - Battery state of charge ≥ 95% with cell balance verified
        - Motor response test: all motors 0-100% sweep with current monitoring
        - Control surface deflection check and actuator health verification
        - Communication link quality ≥ 90% RSSI with redundancy confirmed
        - GPS/GNSS lock acquired with ≥6 satellites and HDOP <2.0
        - Payload attachment verification with release mechanism test
        - Weather conditions within operational limits (wind, visibility, precipitation)
        - Emergency abort systems armed and verified

        ## Abort Criteria
        - Wind speeds > 25 km/h sustained or gusts > 35 km/h
        - GPS/GNSS lock lost > 3 seconds consecutive
        - Battery voltage drop > 5% unexpected or thermal limits exceeded
        - Communication loss > 10 seconds or link quality <70%
        - Motor failure indication or asymmetric thrust detected
        - Payload detachment warning or center of gravity shift
        - Ground control station malfunction or operator incapacitation
        - Airspace violation imminent or obstacle avoidance failure

        ## Progressive Test Plan

        ### Phase 1: Component and Subsystem Validation
        - Duration: 2 weeks, ground testing only
        - Configuration: All subsystems bench-tested individually
        - Success Criteria: All components meet specification, integration interfaces verified
        - Test Count: 25 individual component tests, 10 integration tests

        ### Phase 2: Integrated Ground Testing
        - Duration: 1 week, ground testing with brief hover tests
        - Configuration: Complete vehicle, 0kg payload, 2m altitude maximum
        - Success Criteria: Stable hover, control response, emergency shutdown
        - Test Count: 15 ground tests, 5 low hover tests

        ### Phase 3: Basic Flight Testing
        - Duration: 2 weeks, progressive flight envelope expansion
        - Configuration: 0-5kg payload, 5-15km range, flat terrain only
        - Success Criteria: Stable flight, navigation accuracy, safe landing
        - Test Count: 20 flight tests with increasing complexity

        ### Phase 4: Operational Validation
        - Duration: 3 weeks, full mission profile testing
        - Configuration: 10kg payload, 30km range, rough terrain operations
        - Success Criteria: Complete mission success, rough landing capability
        - Test Count: 50 full mission flights with statistical analysis

        ## Data Collection and Analysis
        - Flight data logging at 100Hz minimum for all sensors
        - Video documentation of all flight tests from multiple angles
        - Performance metrics calculation and trending analysis
        - Failure investigation protocol with 24-hour response requirement
        - Statistical analysis of mission success rate and performance margins

        ## Safety Protocols
        - Test site establishment with 500m safety radius
        - Emergency response team on standby during all flight tests
        - Weather monitoring with automatic test suspension
        - Flight termination system independent of primary control
        - Post-flight inspection checklist before subsequent operations
        """).strip()

    KB_FILES["tests"].write_text(tests_content, encoding="utf-8")
    print("  ✓ tests_template.md created")

# Create components.csv database
if not KB_FILES["components"].exists():
    components_data = [
        {
            "part_name": "T-Motor U8 Pro",
            "component_type": "Motor",
            "description": "High efficiency brushless motor for multirotor applications with integrated cooling",
            "manufacturer": "T-Motor",
            "price_usd": "189.99",
            "specs": "400W, 900KV, 48V max, IP54 rated",
            "weight_g": "198",
            "datasheet_url": "https://store.tmotor.com/goods.php?id=438",
            "category": "propulsion"
        },
        {
            "part_name": "Pixhawk 6C",
            "component_type": "Flight Controller",
            "description": "Advanced autopilot system with triple redundant IMU and integrated vibration isolation",
            "manufacturer": "Holybro",
            "price_usd": "299.99",
            "specs": "STM32H7, Triple IMU, Barometer, Magnetometer, GPS",
            "weight_g": "42",
            "datasheet_url": "https://docs.px4.io/main/en/flight_controller/pixhawk6c.html",
            "category": "avionics"
        },
        {
            "part_name": "Tattu 22000mAh 6S",
            "component_type": "Battery",
            "description": "High capacity LiPo battery pack optimized for long endurance applications",
            "manufacturer": "Tattu",
            "price_usd": "449.99",
            "specs": "22000mAh, 22.2V nominal, 25C discharge, XT90 connector",
            "weight_g": "2650",
            "datasheet_url": "https://www.gensace.de/tattu-22000mah-22-2v-25c-6s1p-lipo-battery-pack.html",
            "category": "power"
        },
        {
            "part_name": "Tarot 25mm Carbon Tube",
            "component_type": "Structural",
            "description": "High strength carbon fiber tube for multirotor arms and structural elements",
            "manufacturer": "Tarot",
            "price_usd": "12.99",
            "specs": "25mm OD, 23mm ID, 500mm length, 3K carbon weave",
            "weight_g": "85",
            "datasheet_url": "https://www.tarotrc.com/product/carbon-fiber-tube-tl2749.html",
            "category": "structure"
        },
        {
            "part_name": "CF 1755 Propeller",
            "component_type": "Propeller",
            "description": "Large diameter carbon fiber propeller optimized for efficiency and payload carrying",
            "manufacturer": "APC",
            "price_usd": "45.99",
            "specs": "17 inch diameter, 5.5 inch pitch, carbon fiber construction",
            "weight_g": "28",
            "datasheet_url": "https://www.apcprop.com/product/17x5-5cf/",
            "category": "propulsion"
        },
        {
            "part_name": "Cube Orange+",
            "component_type": "Flight Controller",
            "description": "Industrial grade autopilot with enhanced processing power and I/O capabilities",
            "manufacturer": "CubePilot",
            "price_usd": "525.00",
            "specs": "STM32H7, Dual IMU, Temperature controlled, Ethernet, CAN FD",
            "weight_g": "68",
            "datasheet_url": "https://docs.cubepilot.org/user-guides/autopilot/the-cube-orange-plus",
            "category": "avionics"
        },
        {
            "part_name": "Here3+ GNSS",
            "component_type": "GPS",
            "description": "High precision GNSS receiver with RTK capability and integrated compass",
            "manufacturer": "CubePilot",
            "price_usd": "285.00",
            "specs": "GPS/GLONASS/Galileo/BeiDou, RTK ready, Triple redundant compass",
            "weight_g": "35",
            "datasheet_url": "https://docs.cubepilot.org/user-guides/here-3/here-3-manual",
            "category": "navigation"
        },
        {
            "part_name": "RFD900x Telemetry",
            "component_type": "Radio",
            "description": "Long range telemetry radio with high data rate and mesh networking capability",
            "manufacturer": "RFDesign",
            "price_usd": "385.00",
            "specs": "900MHz ISM band, 40km range, 250kbps data rate, mesh network",
            "weight_g": "18",
            "datasheet_url": "https://files.rfdesign.com.au/Files/documents/RFD900x%20DataSheet.pdf",
            "category": "communication"
        }
    ]

    with open(KB_FILES["components"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=components_data[0].keys(), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(components_data)
    print("  ✓ components.csv created with 8 drone components")

# ===========================================
# Enhanced CSV Logging Schemas
# ===========================================

print("[bootstrap] Setting up enhanced CSV schemas with wallet tracking...")

# Define comprehensive CSV schemas including wallet fields
CSV_SCHEMAS = {
    "sessions": [
        "session_id", "start_ts", "end_ts", "timebox_min", "actual_duration_min",
        "topic", "project_type", "final_status", "total_rounds", "total_turns",
        "total_speaker_actions", "project_completion_pct", "cycles_completed",
        "total_earned", "total_spent", "completion_rewards", "quality_bonuses",
        "net_economic_change", "wallet_enabled", "config_version", "notes"
    ],

    "turns": [
        "turn_id", "session_id", "round_num", "stage", "agent_id", "agent_name",
        "role_type", "model_id", "device_used", "turn_start_ts", "turn_end_ts",
        "duration_ms", "time_remaining_sec", "response", "banned_terms",
        "questions_asked", "conversation_state", "enhanced_model", "adapter_used",
        "components_mentioned", "routing_method", "addressed_to", "speaker_count",
        "completion_items_done", "project_completion_pct", "cycle_count", "cycle_round",
        "wallet_balance", "wallet_earned", "wallet_spent", "generation_cost",
        "wallet_charged", "domain_terms_used"
    ],

    "wallet_transactions": [
        "transaction_id", "session_id", "turn_id", "agent_id", "agent_name",
        "timestamp", "transaction_type", "amount", "balance_before", "balance_after",
        "operation", "description", "multiplier"
    ],

    "completion_tracking": [
        "completion_id", "session_id", "turn_id", "item_id", "category",
        "description", "completed", "evidence", "completed_by", "completed_at",
        "agent_id", "reward_amount"
    ],

    "preferences": [
        "pair_id", "session_id", "turn_id_a", "turn_id_b", "preference",
        "confidence", "reason", "preference_type", "created_ts", "human_validated",
        "used_for_training"
    ],

    "quality_metrics": [
        "turn_id", "session_id", "agent_id", "domain_terms_used", "components_mentioned",
        "questions_asked", "references_made", "addressed_to_count", "req_compliance_score",
        "banned_terms_count", "quality_bonus_earned", "timestamp"
    ]
}

# Create CSV files with headers
for schema_name, fields in CSV_SCHEMAS.items():
    csv_path = DIRECTORIES["data"] / f"{schema_name}.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
            writer.writeheader()
        print(f"  ✓ {schema_name}.csv schema created")

# ===========================================
# Model Download Script for 4-Model System
# ===========================================

print("[bootstrap] Creating 4-model download script...")

model_download_script = textwrap.dedent("""\
    #!/usr/bin/env python3
    # 4-Model-SOT Model Download Script
    # Downloads and caches all 4 models used in the multi-agent system

    import torch
    import time
    from pathlib import Path
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sentence_transformers import SentenceTransformer

    MODEL_CACHE_DIR = Path("models/cache")
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("[download] Starting 4-Model-SOT model download...")

    # 4-Model Configuration
    AGENT_MODELS = {
        "architect": {
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "role": "technical",
            "color": "cyan"
        },
        "ops": {
            "model_id": "meta-llama/Llama-3.2-1B-Instruct",
            "role": "practical",
            "color": "yellow"
        },
        "manager": {
            "model_id": "google/gemma-2-2b-it",
            "role": "strategic",
            "color": "green"
        },
        "reasoning": {
            "model_id": "h2oai/h2o-danube2-1.8b-base",
            "role": "reasoning_specialist",
            "color": "magenta"
        }
    }

    EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

    # Check CUDA availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[download] GPU Available: {gpu_name} ({total_memory:.1f} GB)")
    else:
        device = torch.device("cpu")
        print("[download] Using CPU (GPU not available)")

    # Download language models
    print("[download] Downloading 4 language models...")
    for agent_name, config in AGENT_MODELS.items():
        model_id = config["model_id"]
        print(f"[download] {agent_name}: {model_id}")

        try:
            start_time = time.time()

            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=MODEL_CACHE_DIR,
                trust_remote_code=True
            )

            # Download model
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=MODEL_CACHE_DIR,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )

            download_time = time.time() - start_time
            print(f"  ✓ {agent_name} downloaded in {download_time:.1f}s")

        except Exception as e:
            print(f"  ✗ Failed to download {agent_name}: {e}")

    # Download embedding model
    print("[download] Downloading embedding model...")
    try:
        start_time = time.time()
        embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_ID,
            cache_folder=MODEL_CACHE_DIR
        )
        download_time = time.time() - start_time
        print(f"  ✓ Embedding model downloaded in {download_time:.1f}s")
    except Exception as e:
        print(f"  ✗ Failed to download embedding model: {e}")

    print("[download] All models cached successfully!")
    print(f"[download] Cache location: {MODEL_CACHE_DIR.absolute()}")
    print("[download] Ready to run multi-agent system!")
    """)

download_script_path = PROJECT_ROOT / "download_models.py"
with open(download_script_path, "w", encoding="utf-8") as f:
    f.write(model_download_script)
print("  ✓ download_models.py created for 4-model system")

# ===========================================
# Governance and Evaluation Framework
# ===========================================

print("[bootstrap] Setting up governance framework...")

# Enhanced golden evaluation template
golden_eval_template = textwrap.dedent("""\
    # Golden Evaluation Suite v10.0

    ## Test Cases for 4-Model-SOT System

    ### TC001: SOT Compliance Validation
    - **Input**: "Design a hybrid drone with 15 kg payload and jet propulsion"
    - **Expected**: System rejects hybrid/jet, enforces 10kg payload and battery-electric
    - **Pass Criteria**: Response mentions 10kg limit, battery-electric only, rejects banned terms
    - **Wallet Impact**: Quality bonus for requirement compliance

    ### TC002: Multi-Agent Collaboration
    - **Input**: Standard session with all 7 agents active
    - **Expected**: Agents address each other using @AgentName format, build on contributions
    - **Pass Criteria**: >80% responses include agent addressing, logical conversation flow
    - **Wallet Impact**: Rewards for collaboration and peer references

    ### TC003: Wallet Economic Constraints
    - **Input**: Session with normal wallet starting balances
    - **Expected**: Resource scarcity affects agent selection, quality work rewarded
    - **Pass Criteria**: Economic stratification visible, broke agents filtered, positive rewards
    - **Wallet Impact**: System demonstrates metabolic constraint functionality

    ### TC004: Technical Depth and Component Integration
    - **Input**: "What motor should we use for the cargo drone?"
    - **Expected**: Specific component recommendations with calculations
    - **Pass Criteria**: Components from database mentioned, technical specifications provided
    - **Wallet Impact**: Component usage rewards, domain expertise bonuses

    ### TC005: Completion Tracking Integration
    - **Input**: Session with acceptance criteria loaded
    - **Expected**: Agents complete specific criteria and mark accomplishments
    - **Pass Criteria**: Completion percentage increases, specific criteria marked done
    - **Wallet Impact**: Completion rewards distributed, quality bonuses earned

    ## Performance Benchmarks
    - SOT compliance rate: >95%
    - Banned term elimination: 100%
    - Inter-agent addressing: >80%
    - Technical depth score: >0.7
    - Conversation coherence: >0.8
    - Wallet system functionality: >90%
    - Component database utilization: >70%
    - Completion tracking accuracy: >85%

    ## Regression Test Suite
    - Session completion rate: >95%
    - Model loading success: 100%
    - CSV logging integrity: 100%
    - Wallet balance persistence: 100%
    - Agent role assignment accuracy: 100%
    """).strip()

golden_eval_path = DIRECTORIES["evals"] / "golden" / "golden_suite_v10.0.md"
with open(golden_eval_path, "w", encoding="utf-8") as f:
    f.write(golden_eval_template)
print("  ✓ Golden evaluation suite v10.0 created")

# ===========================================
# README and Documentation
# ===========================================

print("[bootstrap] Creating project documentation...")

readme_content = textwrap.dedent("""\
    # 4-Model-SOT: Multi-Agent System with Metabolic Wallet Constraints

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

    ## Quick Start

    1. **Install Dependencies**:
       ```bash
       pip install transformers torch sentence-transformers accelerate peft datasets
       ```

    2. **Download Models**:
       ```bash
       python download_models.py
       ```

    3. **Run System**:
       ```python
       # Execute the model loader block
       # Then execute Part 1: Core Infrastructure
       # Then execute Part 2: Orchestration
       # Finally run: run_enhanced_session()
       ```

    ## Configuration

    - **config.json**: System configuration with wallet economics
    - **kb/**: Knowledge base with SOT, acceptance criteria, agent definitions
    - **data/**: CSV logging with wallet transaction tracking
    - **models/cache/**: Downloaded model cache

    ## Features

    - ✅ 4-model architecture with role-based assignment
    - ✅ 7-agent team with specialized expertise
    - ✅ Metabolic wallet constraints with resource decay
    - ✅ Component database with semantic search
    - ✅ Completion tracking with automatic rewards
    - ✅ Advanced routing with cooldown + wallet filtering
    - ✅ Real-time economic dashboard
    - ✅ Comprehensive CSV logging with wallet data

    ## Project Structure

    ```
    4-model-SOT/
    ├── config.json                 # System configuration
    ├── download_models.py           # Model download script
    ├── kb/                         # Knowledge base
    │   ├── SOT.md                  # Source of truth requirements
    │   ├── agents.md               # 7-agent role definitions
    │   ├── acceptance.md           # Acceptance criteria
    │   ├── components.csv          # Component database
    │   └── tests_template.md       # Test protocols
    ├── data/                       # CSV logs and datasets
    │   ├── turns.csv               # Turn-by-turn logs with wallet data
    │   ├── wallet_transactions.csv # Detailed wallet transactions
    │   └── completion_tracking.csv # Project completion tracking
    ├── models/cache/               # Downloaded model cache
    ├── adapters/lora/              # LoRA fine-tuned adapters
    └── evals/golden/               # Golden evaluation suite
    ```

    ## Version History
    - v10.0: 4-model system with wallet economics integration
    - v9.0: Enhanced routing and completion tracking
    - v1.0: Initial 3-model prototype
    """).strip()

readme_path = PROJECT_ROOT / "README.md"
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_content)
print("  ✓ README.md created")

# Version tracking with wallet system
version_log = {
    "project_version": "10.0.0",
    "project_name": "4-model-SOT",
    "sot_version": "1.0",
    "agents_version": "10.0",
    "config_version": "10.0",
    "created": datetime.now().isoformat(),
    "features": {
        "wallet_economics": True,
        "4_model_architecture": True,
        "7_agent_team": True,
        "metabolic_constraints": True,
        "component_database": True,
        "completion_tracking": True,
        "advanced_routing": True,
        "csv_logging_enhanced": True
    },
    "model_configuration": {
        "technical": "Qwen/Qwen2.5-1.5B-Instruct",
        "practical": "meta-llama/Llama-3.2-1B-Instruct",
        "strategic": "google/gemma-2-2b-it",
        "reasoning": "h2oai/h2o-danube2-1.8b-base",
        "embedding": "sentence-transformers/all-MiniLM-L6-v2"
    }
}

version_path = PROJECT_ROOT / "VERSION.json"
with open(version_path, "w", encoding="utf-8") as f:
    json.dump(version_log, f, indent=2)
print("  ✓ VERSION.json created")

# ===========================================
# Package Requirements
# ===========================================

requirements_content = textwrap.dedent("""\
    # 4-Model-SOT Multi-Agent System Requirements
    torch>=2.0.0
    transformers>=4.55.0
    sentence-transformers>=2.2.2
    accelerate>=0.25.0
    peft>=0.7.0
    datasets>=2.14.0
    scikit-learn>=1.3.0
    numpy>=1.24.0
    pandas>=2.0.0
    pathlib
    textwrap
    uuid
    threading
    csv
    json
    re
    time
    datetime
    typing
    math
    """).strip()

requirements_path = PROJECT_ROOT / "requirements.txt"
with open(requirements_path, "w", encoding="utf-8") as f:
    f.write(requirements_content)
print("  ✓ requirements.txt created")

# ===========================================
# Final Summary and Instructions
# ===========================================

print("\n" + "="*70)
print("4-MODEL-SOT PROJECT BOOTSTRAP COMPLETE")
print("="*70)

print(f"\n🚀 Project: 4-Model-SOT Multi-Agent System v10.0")
print(f"📁 Root: {PROJECT_ROOT}")
print(f"🧠 Architecture: 4 models, 7 agents, wallet economics")

print(f"\n📋 Created Components:")
print(f"  ✓ Directory structure (12 directories)")
print(f"  ✓ Enhanced config.json with wallet system")
print(f"  ✓ Knowledge base: SOT, agents, acceptance, tests (4 files)")
print(f"  ✓ Component database: 8 drone components")
print(f"  ✓ CSV schemas: 6 schemas with wallet tracking")
print(f"  ✓ Model download script for 4-model system")
print(f"  ✓ Golden evaluation suite v10.0")
print(f"  ✓ Documentation: README.md, VERSION.json, requirements.txt")

print(f"\n💰 Wallet Economics:")
print(f"  ✓ Metabolic constraints with 10% decay rate")
print(f"  ✓ Operation costs: Basic (10), Enhanced (25)")
print(f"  ✓ Reward system: Completion (50), Quality (variable)")
print(f"  ✓ Economic tracking in all CSV schemas")

print(f"\n🎯 7-Agent Team Configuration:")
print(f"  ✓ Eel (Electrical Engineer) - Technical Pool")
print(f"  ✓ Bear (Mechanical Engineer) - Technical Pool")
print(f"  ✓ Falcon (Flight Controls Engineer) - Technical Pool")
print(f"  ✓ Beaver (Systems Engineer) - Strategic Pool")
print(f"  ✓ Mongoose (Test Engineer) - Practical Pool")
print(f"  ✓ Lion (Manager) - Strategic Pool")
print(f"  ✓ Crow (Reasoning Specialist) - Reasoning Pool")

print(f"\n🔄 Next Steps:")
print(f"  1. Install requirements: pip install -r requirements.txt")
print(f"  2. Download models: python download_models.py")
print(f"  3. Load the GPU model loader block")
print(f"  4. Execute Part 1: Core Infrastructure")
print(f"  5. Execute Part 2: Agent Coordination")
print(f"  6. Run session: run_enhanced_session()")

print(f"\n🧪 Testing:")
print(f"  • Golden evaluation suite ready")
print(f"  • Regression testing configured")
print(f"  • Wallet economics validation included")

print(f"\n✅ Ready for team deployment!")
print(f"📊 All components integrated with metabolic wallet constraints")

# ===========================================
# File: bootstrap_project.py
# Tree: ~/4-model-SOT/bootstrap_project.py
# Created: 2025-12-19 17:30:00 UTC
# ===========================================
