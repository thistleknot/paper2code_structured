# 📄 Enhanced PaperCoder: Structured Code Generation from Scientific Papers

📄 **Original work:** [Paper2Code on arXiv](https://arxiv.org/abs/2504.17192) | 🙏 **Credit:** [Original PaperCoder Repository](https://github.com/going-doer/Paper2Code)

**Enhanced PaperCoder** is a structured multi-phase pipeline that transforms scientific papers into comprehensive code repositories. Building on the excellent foundation of the original PaperCoder work, this enhanced version introduces a systematic approach with structured outputs, dependency analysis, and file organization.

## 🌟 Key Enhancements

- **📋 Structured Planning Phase**: Strategic analysis with Six Thinking Hats methodology and dependency ranking
- **🔍 Comprehensive Analysis Phase**: Individual file analysis with focused requirements
- **📁 Smart File Organization**: Dependency-aware development ordering (utilities → classes → main)
- **💻 Enhanced Coding Phase**: Structured code generation with diff outputs and parallel processing
- **🛠️ Flexible API Support**: Compatible with OpenAI, Ollama, and other OpenAI-compatible endpoints
- **📊 Basic Resume Capability**: Skip planning phase and resume from analysis if data exists
- **📄 PDF Support**: Use docling to convert PDFs to markdown before processing

---

## 🗺️ Table of Contents

- [⚡ Quick Start](#-quick-start)
- [📚 Pipeline Architecture](#-pipeline-architecture)
- [🛠️ Installation & Setup](#-installation--setup)
- [🚀 Usage Examples](#-usage-examples)
- [📊 Output Structure](#-output-structure)
- [🔧 Advanced Configuration](#-advanced-configuration)
- [🙏 Acknowledgments](#-acknowledgments)

---

## ⚡ Quick Start

### Prerequisites
```bash
# Install dependencies
pip install requests tqdm json-repair

# For PDF to markdown conversion
pip install docling
```

### Basic Usage
```bash
# Convert PDF to markdown first (if needed)
docling paper.pdf --output paper.md

# Run pipeline with OpenAI-compatible API (OpenRouter, Ollama, etc.)
export OPENROUTER_API_KEY="your-api-key"

python main.py \
    --paper_name kumo \
    --paper_markdown_path kumo_relational_foundation_model.md \
    --api_base_url "https://openrouter.ai/api/v1" \
    --api_key "$OPENROUTER_API_KEY" \
    --reasoning_model "deepseek/deepseek-chat-v3-0324" \
    --coding_model "qwen/qwen-2.5-coder-32b-instruct" \
    --output_dir outputs/kumo \
    --output_repo_dir repos/kumo
```

---

## 📚 Pipeline Architecture

The pipeline consists of **four main phases** with **seven internal planning stages**:

### 🎯 **Phase 1: Strategic Planning** (7 Internal Stages)

#### Stage 1: **Core Planning** 
**Predicates-first methodology** - actions before entities:
- **Predicates**: Core interactions and transformations (what happens)
- **Entities**: Components and actors (what interacts)
- **Intent & Requirements**: Paper methodology extraction with necessary conditions
- **Methods & Classes**: Function headers and abstract class definitions
- **Datasets & Metrics**: Experimental setup and evaluation framework

#### Stage 2: **Six Thinking Hats Analysis**
Strategic evaluation across multiple perspectives:
- **White Hat**: Facts and data analysis
- **Red Hat**: Intuitive concerns and emotional responses  
- **Black Hat**: Risk identification and mitigation strategies
- **Yellow Hat**: Benefits and opportunities assessment
- **Green Hat**: Creative alternatives and solutions
- **Blue Hat**: Process control and meta-analysis

#### Stage 3: **Dependency Analysis**
Priority-driven feature ranking:
- **Critical Path Detection**: High-priority features with no dependencies
- **Utility vs Effort Matrix**: Strategic resource allocation
- **File Impact Mapping**: Which files each feature affects
- **Development Sequencing**: Optimal implementation order

#### Stage 4: **Code Structure Design**
YAML-style code organization (avoids LLM repetition issues):
- **Utility Functions**: Standalone function headers and descriptions
- **Class Headers**: Class names with initialization parameters
- **Class Member Functions**: Method definitions and interactions
- **Main Processing**: Execution flow and orchestration steps
- **File Structure**: Recommended file organization and purposes

#### Stage 5: **System Architecture**
Technical implementation framework:
- **Data Structures & Interfaces**: API contracts and schemas
- **Program Call Flow**: Execution sequence and dependencies
- **Implementation Approach**: Technical strategy and patterns

#### Stage 6: **Task Breakdown**
Development-ready file specifications:
- **Logic Analysis**: Per-file implementation requirements
- **Metadata Tagging**: Priority, utility, effort, and critical path flags
- **Package Dependencies**: Third-party requirements identification

#### Stage 7: **Configuration Generation**
Deployment-ready settings:
- **YAML Configuration**: Paper-specific hyperparameters
- **Environment Setup**: Model and training configurations
- **Validation Parameters**: Testing and evaluation settings

### 🔍 **Phase 2: Individual File Analysis**
Detailed analysis for each file identified in planning:
- **Core Functionality**: What each file implements from the paper
- **Implementation Strategy**: Technical approach and design patterns
- **Dependencies & Data Flow**: Integration with other components
- **Focused Requirements**: Methods, classes, and interaction specifications

### 📁 **Phase 3: File Organization**
Dependency-aware development ordering:
- **File Type Classification**: Utilities, classes, and main orchestration
- **Priority Assessment**: Critical path and dependency mapping
- **Development Order**: Optimal sequence (utilities → classes → main)

### 💻 **Phase 4: Code Generation**
Parallel structured code generation:
- **Structured Implementation**: Complete code with deliberation reasoning
- **Diff Output**: Version control ready format
- **Parallel Processing**: Concurrent file generation with context sharing
- **Quality Validation**: Import checks and basic functionality testing

---

## 🛠️ Installation & Setup

### Clone and Install
```bash
git clone <your-repo-url>
cd enhanced-papercoder
pip install -r requirements.txt
```

The pipeline supports any OpenAI-compatible API endpoint including OpenRouter, Ollama, and direct OpenAI access.

---

## 🚀 Usage Examples

### Basic Paper Processing
```bash
# Convert PDF and process
docling paper.pdf --output paper.md

python main.py \
    --paper_name kumo \
    --paper_markdown_path kumo_relational_foundation_model.md \
    --api_base_url "https://openrouter.ai/api/v1" \
    --api_key "$OPENROUTER_API_KEY" \
    --reasoning_model "deepseek/deepseek-chat-v3-0324" \
    --coding_model "qwen/qwen-2.5-coder-32b-instruct" \
    --output_dir outputs/kumo \
    --output_repo_dir repos/kumo
```

### Resume from Analysis Phase
```bash
# Skip planning if already completed
python main.py \
    --paper_name kumo \
    --paper_markdown_path kumo_relational_foundation_model.md \
    --api_base_url "https://openrouter.ai/api/v1" \
    --api_key "$OPENROUTER_API_KEY" \
    --reasoning_model "deepseek/deepseek-chat-v3-0324" \
    --coding_model "qwen/qwen-2.5-coder-32b-instruct" \
    --output_dir outputs/kumo \
    --output_repo_dir repos/kumo \
    --resume_from_analysis
```

**Resume Logic:**
- If planning phase is complete but analysis isn't → Resume from analysis
- If both planning and analysis are complete → Skip directly to coding
- Otherwise → Start from the beginning

---

## 📊 Output Structure

### Complete Pipeline Outputs
```
outputs/kumo/
├── 📋 Planning Artifacts (7 stages)
│   ├── planning_response.json                 # Raw planning API response
│   ├── planning_structured.json              # Core planning with predicates-first
│   ├── six_hats_response.json                # Raw six hats API response
│   ├── six_hats_structured.json              # Strategic analysis & risk assessment
│   ├── dependency_response.json              # Raw dependency API response
│   ├── dependency_structured.json            # Priority ranking with utility/effort
│   ├── code_structure_response.json          # Raw code structure API response
│   ├── code_structure_structured.json        # YAML-style code organization
│   ├── architecture_response.json            # Raw architecture API response
│   ├── architecture_structured.json          # System design & interfaces
│   ├── task_list_response.json               # Raw task list API response
│   ├── task_list_structured.json             # File breakdown with metadata
│   ├── config_response.json                  # Raw config API response
│   ├── config_structured.json                # Configuration generation
│   ├── planning_config.yaml                  # Generated config file
│   ├── planning_trajectories.json            # Complete conversation history
│   ├── model_config.json                     # Model assignments per stage
│   └── all_structured_responses.json         # Combined planning data
│
├── 🔍 Analysis & Organization Artifacts
│   ├── {filename}_simple_analysis_response.json      # Raw per-file analysis
│   ├── {filename}_simple_analysis_structured.json    # Structured per-file analysis
│   ├── file_organization_response.json               # Raw file organization
│   └── file_organization_structured.json             # Development order
│
├── 💻 Coding Artifacts
│   ├── structured_code_responses/            # Structured code with deliberation
│   │   └── {filename}_structured.json       # Deliberation + utility + code
│   ├── coding_artifacts/                     # Implementation reasoning
│   │   ├── {filename}_coding.txt            # Full coding response
│   │   └── {filename}_deliberation.txt      # Reasoning + utility
│   ├── diffs/                               # Version control diffs
│   │   └── {filename}.diff                  # Git-ready diff format
│   └── coding_results.json                  # Success/failure summary
│
└── 📄 Final Repository
repos/kumo_repo/
├── config.yaml                # Runtime configuration (copied from planning)
├── main.py                    # Entry point (generated last in development order)
├── utils/                     # Utility functions (generated first)
│   ├── data_processing.py
│   └── evaluation_metrics.py
├── models/                    # Core classes (generated second)
│   ├── transformer.py
│   └── attention.py
└── evaluation/               # Other components
    └── benchmark.py
```

### Key Structured Outputs

#### Planning Phase Results (7 JSON files)
- **Strategic Analysis**: Six Thinking Hats evaluation with risk mitigation
- **Dependency Ranking**: High/medium/low priority with utility/effort scores
- **Code Structure**: Function headers, class definitions, file organization
- **Architecture**: System design with data flow and interfaces
- **Task Breakdown**: File-by-file implementation requirements
- **Configuration**: Complete YAML with paper-specific hyperparameters

#### Analysis Phase Results
- **Individual File Analysis**: Detailed implementation strategy per file
- **Core Functionality**: What each file implements from the paper
- **Focused Requirements**: Methods, classes, and interaction specifications
- **Technical Considerations**: Dependencies, performance, error handling

#### File Organization Results
- **Development Order**: Utilities → Classes → Main processing
- **File Type Classification**: Priority and dependency analysis
- **Implementation Rationale**: Why files are ordered this way

#### Coding Phase Results
- **Structured Code**: Complete implementations with deliberation reasoning
- **Diff Files**: Clean version control integration ready for git
- **Parallel Results**: Success/failure tracking across all files
- **Context Sharing**: Previously implemented files available to subsequent generation

---

## 🔧 Advanced Configuration

### Command Line Options
```bash
# Core Configuration
--paper_name           # Project identifier
--paper_markdown_path  # Input paper in markdown format
--output_dir          # Artifacts storage directory  
--output_repo_dir     # Final repository location

# Model Selection
--reasoning_model     # Model for planning & analysis (default: granite3.3:2b)
--coding_model        # Model for code generation (default: DeepSeek-R1)
--api_base_url        # API endpoint (default: http://localhost:11434)
--api_key             # API authentication key

# Performance Tuning
--max_parallel        # Parallel coding tasks (default: 1)
--timeout             # Request timeout seconds (default: 600)
--seed                # Deterministic generation seed (default: 42)

# Workflow Control
--resume_from_analysis  # Skip planning if data exists
```

### Model Recommendations

#### 🌐 **OpenAI-Compatible APIs**
```bash
# Current example models
--reasoning_model "deepseek/deepseek-chat-v3-0324"     # Strategic planning
--coding_model "qwen/qwen-2.5-coder-32b-instruct"     # Code generation
```

### API Client Features
- **Timeout Handling**: Configurable request timeouts with retry logic
- **Generation Settings Rotation**: Balanced → Precise → Creative on retries
- **Seed Management**: Deterministic generation with automatic seed incrementation
- **Streaming Support**: Real-time response monitoring with repetition detection

---

## 🤖 AutoGen Integration

AutoGen multi-agent collaboration is implemented but commented out for future use:
- Complete implementation in `functions.py` 
- Multi-agent approach with Engineer, Critic, CodeExecutor, Manager
- To activate: uncomment `run_autogen_coding_phase()` in main.py

---

## 📈 Performance Notes

- **Planning Phase**: ~5-12 minutes (7 strategic stages)
- **Analysis Phase**: ~1-2 minutes per file
- **File Organization**: ~30 seconds
- **Coding Phase**: ~3-8 minutes per file
- **Total Time**: ~25-45 minutes for typical papers (5-15 files)

**Tips**: Use `--resume_from_analysis` to skip planning when experimenting with different models.

---

## 🙏 Acknowledgments

This enhanced version builds upon the excellent foundation provided by the original **PaperCoder** research:

> 📄 **Original Paper**: [Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning](https://arxiv.org/abs/2504.17192)  
> 🛠️ **Original Repository**: [https://github.com/going-doer/Paper2Code](https://github.com/going-doer/Paper2Code)  
> 👥 **Original Authors**: Outstanding work by the Paper2Code research team

### What We Enhanced
- **Predicates-First Methodology**: FOL logic approach - actions before entities P(S,[O])
- **Seven-Stage Strategic Planning**: Planning → Six Hats → Dependencies → Code Structure → Architecture → Tasks → Config
- **Smart File Organization**: Priority-driven development ordering
- **Structured Code Generation**: Deliberation + utility + diff format for complete traceability
- **Basic Resume Capability**: Skip planning phase when data exists
- **AutoGen Integration**: Multi-agent collaboration framework (future direction)

---

## 🐛 Known Issues

- **Resume Granularity**: Can only resume from analysis phase, not mid-coding
- **Parallel Dependencies**: Files generated in parallel may have inter-dependencies
- **Error Recovery**: Limited handling for malformed API responses

---

*Convert PDFs with docling, then transform papers into code with enhanced structured planning.*