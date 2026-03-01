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
│   ├── 📁 electronics_expert/                # Fine-tuned VHDL expert
│   │   └── 📁 final_model/
│   ├── 📁 firmware_expert/                   # Fine-tuned C++ expert
│   │   └── 📁 final_model/
│   └── 📁 bio_expert/                        # Fine-tuned biomedical expert
│       └── 📁 final_model/
│
├── 📁 configs/                               # CONFIGURATION FILES
│   └── 📄 agent_config.yaml                   # Agent parameters
│
├── 📁 outputs/                                # GENERATED DESIGNS
│   └── 📁 pulse_ox_design_output/             # EXAMPLE OUTPUT FOLDER
│       ├── 📄 full_design.json                 # Complete design in JSON
│       ├── 📁 firmware/                         # Generated firmware files
│       │   ├── 📄 pulse_ox_main.cpp
│       │   ├── 📄 sensor_driver.h
│       │   └── 📄 rtos_config.h
│       ├── 📁 hardware/                         # Generated HDL files
│       │   ├── 📄 pulse_ox_frontend.vhd
│       │   ├── 📄 pulse_ox_top.v
│       │   └── 📄 schematic.net
│       ├── 📁 mechanical/                        # Generated CAD files
│       │   ├── 📄 enclosure.scad
│       │   ├── 📄 material_spec.json
│       │   └── 📄 assembly.step
│       ├── 📁 bio/                               # Generated safety specs
│       │   ├── 📄 safety_spec.json
│       │   ├── 📄 biocompatibility.txt
│       │   └── 📄 clinical_workflow.md
│       ├── 📁 test/                              # Test benches
│       │   ├── 📄 test_pulse_ox.py
│       │   └── 📄 simulation_results.txt
│       └── 📁 manufacturing/                      # Manufacturing files
│           ├── 📄 bom.csv
│           └── 📄 pcb_requirements.txt
│
└── 📄 README.md                              # THIS FILE


📂 Output Folder: pulse_ox_design_output
When you run:

bash
python run.py --task "Design pulse oximeter" --output_dir ./outputs/pulse_ox_design_output
The system creates outputs/pulse_ox_design_output/ containing:

📄 1. full_design.json - Complete design in one file
json
{
  "design_id": "POX-20250302-001",
  "timestamp": "2025-03-02T10:30:00",
  "specifications": {...},
  "agents_used": ["mechanical", "electronics", "firmware", "bio"],
  "generated_files": {
    "firmware": {...},
    "hardware": {...},
    "mechanical": {...},
    "bio": {...}
  }
}
📁 2. firmware/ - Actual C++ code files
pulse_ox_main.cpp - Main controller code

sensor_driver.h - MAX30102 sensor driver

rtos_config.h - FreeRTOS configuration

algorithm.cpp - Signal processing algorithms

📁 3. hardware/ - VHDL/Verilog and schematics
pulse_ox_frontend.vhd - VHDL implementation

pulse_ox_top.v - Verilog alternative

schematic.net - KiCad netlist

bom_hardware.csv - Component list

📁 4. mechanical/ - CAD and material specs
enclosure.scad - OpenSCAD 3D model

material_spec.json - Material properties

assembly.step - STEP file (CAD)

pcb_mount.scad - PCB holder design

📁 5. bio/ - Safety and compliance
safety_spec.json - IEC 60601-1 requirements

biocompatibility.txt - ISO 10993 certifications

clinical_workflow.md - Usage guidelines

sterilization.txt - Sterilization methods

📁 6. test/ - Verification files
test_pulse_ox.py - Python test bench

simulation_results.txt - Simulation outputs

waveform_data.csv - Test waveforms

📁 7. manufacturing/ - Production files
bom.csv - Complete bill of materials

pcb_requirements.txt - PCB fabrication notes

assembly_instructions.md - Assembly guide


