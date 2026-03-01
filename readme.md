##### Multi-Agent AI System for Medical Device Design
This system uses 4 specialized AI agents working together in latent space to design complete medical devices (pulse oximeters, defibrillators, dialysis machines, etc.). Each agent is an expert in its domain, and they collaborate without losing information by sharing internal "thoughts" (latent representations) rather than text.


C:\Users\Dell\LatentMAS_Medical\
│
├── 📄 run.py                          # MAIN ENTRY POINT
│     Loads your 4 trained agents and runs the design process
│     Usage: python run.py --task "Design pulse oximeter" --latent_steps 50
│
├── 📄 models.py                        # CORE LATENT SPACE ENGINE
│     Wraps HuggingFace models to enable latent space operations
│     Contains: ModelWrapper, LatentSpaceAligner
│     This is what allows agents to share "thoughts" without text!
│
├── 📄 prompts.py                       # TASK TEMPLATES
│     Contains prompt templates for each agent type
│     Helps structure inputs consistently
│
├── 📄 utils.py                          # HELPER FUNCTIONS
│     File I/O, JSON parsing, visualization helpers
│
├── 📄 data_preparation.py                # DATA CLEANING SCRIPT
│     Converts your books/PDFs/code into training format
│     Run this first to prepare your datasets
│
├── 📄 train_agents.py                    # TRAINING SCRIPT
│     Fine-tunes all 4 agents on your cleaned data
│     Saves weights to checkpoints/ folder
│
├── 📄 requirements.txt                    # DEPENDENCIES
│     All Python packages needed: torch, transformers, etc.
│     Run: pip install -r requirements.txt
│
├── 📁 agents/                             # SPECIALIZED AGENTS
│   ├── 📄 __init__.py                      # Makes it a package
│   ├── 📄 mechanical_agent.py              # CAD/Mechanical expert
│   │     Generates: STEP files, OpenSCAD code, material specs
│   │
│   ├── 📄 electronics_agent.py             # VHDL/Circuit expert
│   │     Generates: VHDL/Verilog, KiCad schematics, BOM
│   │
│   ├── 📄 firmware_agent.py                # Embedded C++ expert
│   │     Generates: C++ code, RTOS configs, algorithms
│   │
│   └── 📄 bio_agent.py                     # Biomedical expert
│         Generates: Safety specs, ISO standards, biocompatibility
│
├── 📁 methods/                             # THREE APPROACHES (for comparison)
│   ├── 📄 __init__.py                      # Package init
│   ├── 📄 baseline.py                       # SINGLE MODEL BASELINE
│   │     One model does everything (not specialized, for comparison only)
│   │
│   ├── 📄 text_mas.py                       # TEXT-BASED MULTI-AGENT (OLD WAY)
│   │     Agents talk via text - LOSES INFORMATION!
│   │     This proves why latent space is better
│   │
│   └── 📄 latent_mas.py                     # LATENT SPACE MULTI-AGENT (OUR METHOD)
│         Agents share internal thoughts - NO INFORMATION LOSS!
│         4x faster, 80% fewer tokens than text_mas.py
│
├── 📁 data/                                 # YOUR CLEANED TRAINING DATA
│   ├── 📁 mechanical/
│   │   ├── 📄 cad_instructions.jsonl        # CAD generation examples
│   │   ├── 📄 material_specs.jsonl          # Material properties
│   │   └── 📄 mechanical_qa.jsonl           # Mechanical Q&A
│   │
│   ├── 📁 electronics/
│   │   ├── 📄 vhdl_examples.jsonl           # VHDL code examples
│   │   ├── 📄 verilog_examples.jsonl        # Verilog examples
│   │   ├── 📄 circuit_designs.jsonl         # Circuit descriptions
│   │   └── 📄 spice_netlists.jsonl          # SPICE simulations
│   │
│   ├── 📁 firmware/
│   │   ├── 📄 embedded_cpp.jsonl            # C++ firmware examples
│   │   ├── 📄 rtos_configs.jsonl            # RTOS examples
│   │   └── 📄 algorithm_descriptions.jsonl  # Signal processing algorithms
│   │
│   └── 📁 bio/
│       ├── 📄 fda_guidelines.jsonl          # FDA standards
│       ├── 📄 iso_standards.jsonl           # ISO 10993, 60601
│       ├── 📄 biocompatibility.jsonl        # Material compatibility
│       └── 📄 safety_requirements.jsonl     # Safety specs
│
├── 📁 checkpoints/                          # YOUR TRAINED MODEL WEIGHTS
│   ├── 📁 mechanical_expert/                 # Fine-tuned CAD expert
│   │   └── 📁 final_model/                    # Actual weights
# Multi-Agent AI System for Medical Device Design

This repository implements a latent-space multi-agent system for designing medical devices (for example: pulse oximeters, defibrillators, dialysis machines). Four specialized agents (mechanical, electronics, firmware, and biomedical) collaborate by sharing internal latent representations rather than text to reduce information loss.

## Quickstart

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main demo (example):

```bash
python run.py --task "Design pulse oximeter" --latent_steps 50 --output_dir ./outputs/pulse_ox_design_output
```

## Project Overview

- `run.py` — Main entry point that loads trained agents and runs design tasks.
- `models.py` — Latent space engine and model wrappers.
- `prompts.py` — Prompt and task templates.
- `utils.py` — Helper functions (I/O, JSON, visualization).
- `data_preparation.py` — Data cleaning / preprocessing script.
- `train_agents.py` — Fine-tuning script for the agent models.
- `requirements.txt` — Python dependencies (torch, transformers, etc.).

### Agents (package: `agents/`)

- `mechanical_agent.py` — CAD/mechanical expert (generates STEP, OpenSCAD, material specs).
- `electronics_agent.py` — VHDL/circuit expert (generates VHDL/Verilog, BOMs, schematics).
- `firmware_agent.py` — Embedded C++ expert (generates firmware, RTOS configs).
- `bio_agent.py` — Biomedical expert (safety specs, standards, biocompatibility).

### Methods (package: `methods/`)

- `baseline.py` — Single-model baseline for comparison.
- `text_mas.py` — Text-based multi-agent approach (agents communicate via text).
- `latent_mas.py` — Latent-space multi-agent implementation (agents share internal representations).

### Data and Checkpoints

- `data/` — Prepared training data (JSONL examples per domain).
- `checkpoints/` — Trained model weights (one directory per agent).
- `configs/agent_config.yaml` — Agent hyperparameters and settings.

## Example output structure

When you run the example above the system will create an output folder such as `outputs/pulse_ox_design_output/` containing:

- `full_design.json` — Complete design JSON with metadata and generated files.
- `firmware/` — Generated C++ firmware files (e.g., `pulse_ox_main.cpp`, sensor driver headers).
- `hardware/` — HDL and schematics (VHDL/Verilog, netlists).
- `mechanical/` — CAD output (OpenSCAD, STEP, material specs).
- `bio/` — Safety and compliance documents (IEC/ISO references, clinical workflow).
- `test/` — Test benches and simulation outputs.
- `manufacturing/` — BOMs and manufacturing notes.

Example `full_design.json` (abbreviated):

```json
{
  "design_id": "POX-20250302-001",
  "timestamp": "2025-03-02T10:30:00",
  "specifications": { /* ... */ },
  "agents_used": ["mechanical", "electronics", "firmware", "bio"],
  "generated_files": { /* firmware, hardware, mechanical, bio */ }
}
```

## Notes

- Use `data_preparation.py` to convert raw sources into training JSONL files before fine-tuning.
- Store checkpoints under `checkpoints/<agent>_expert/final_model/`.

---

If you want, I can also:

- Add a short CONTRIBUTING section and license.
- Create a minimal README badge / table of contents.
assembly.step - STEP file (CAD)



