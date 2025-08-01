# 📄 Enhanced PaperCoder: Structured Code Generation from Scientific Papers

📄 **Original work:** [Paper2Code on arXiv](https://arxiv.org/abs/2504.17192) | 🙏 **Credit:** [Original PaperCoder Repository](https://github.com/going-doer/Paper2Code)

**Enhanced PaperCoder** is a structured multi-phase pipeline that transforms scientific papers into comprehensive code repositories. Building on the excellent foundation of the original PaperCoder work, this enhanced version introduces a systematic approach with structured outputs, dependency analysis, and **global coherence through accumulating context**.

## 🌟 Key Enhancements

- **📋 Configuration-Driven Pipeline**: Automated stage dependency management with Russian nested dolls architecture
- **🧠 Accumulating Context Pattern**: Each iteration is informed by all previous outputs for global coherence
- **🔍 Comprehensive Analysis Phase**: Individual file analysis with focused requirements and cross-file consistency
- **📁 Smart File Organization**: Dependency-aware development ordering (utilities → classes → main)
- **💻 Enhanced Coding Phase**: File-level resume capability with structured code generation and diff outputs
- **🛠️ Flexible API Support**: Compatible with OpenAI, Ollama, and other OpenAI-compatible endpoints
- **🔄 Advanced Resume Capability**: Resume from any phase and skip completed files automatically
- **📄 PDF Support**: Use docling to convert PDFs to markdown before processing
- **🎯 Uncertainty Boundary Management**: Systematic propagation of "unclear" items across stages

---

## 🗺️ Table of Contents

- [⚡ Quick Start](#-quick-start)
- [📚 Pipeline Architecture](#-pipeline-architecture)
- [🧠 Accumulating Context Innovation](#-accumulating-context-innovation)
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

*Convert PDFs with docling, then transform papers into code with enhanced structured planning.*

### Basic Usage
```bash
# Convert PDF to markdown first (if needed)
docling paper.pdf --output paper.md

# Run pipeline with OpenAI-compatible API (OpenRouter, Ollama, etc.)
export OPENROUTER_API_KEY="your-api-key"

python main.py \
    --paper_name paper \
    --paper_markdown_path paper.md \
    --api_base_url "https://openrouter.ai/api/v1" \
    --api_key "$OPENROUTER_API_KEY" \
    --reasoning_model "deepseek/deepseek-chat-v3-0324" \
    --coding_model "qwen/qwen-2.5-coder-32b-instruct" \
    --output_dir outputs/paper \
    --output_repo_dir repos/paper
```

---

## 🧠 Accumulating Context Innovation

### **Global Coherence Through Progressive Context Building**

Our breakthrough **Accumulating Context Pattern** ensures every iteration benefits from all previous outputs, achieving global coherence while maintaining individual focus:

```
File 1: Context = [] → Analysis 1
File 2: Context = [Analysis 1] → Analysis 2  
File 3: Context = [Analysis 1, Analysis 2] → Analysis 3
...
```

### **How It Works**

#### **Phase 2: File Analysis with Context Accumulation**
- **Iteration 1**: Analyzes first file with base context only
- **Iteration 2**: Sees Analysis 1 + analyzes second file  
- **Iteration 3**: Sees Analysis 1 + Analysis 2 + analyzes third file
- **Result**: Each file analysis is informed by ALL previous decisions

#### **Phase 4: Code Generation with Context Accumulation**  
- **Enhanced Context Sharing**: Each new file sees all previously generated code
- **Smart Context Management**: Full code for recent files, signatures for older files
- **Integration Guidance**: Explicit consistency checks with established patterns

### **Benefits Achieved**

- ✅ **Global Coherence**: Consistent interpretation of uncertain requirements across all files
- ✅ **Uncertainty Boundary Consistency**: Same approach to handling "unclear" items throughout
- ✅ **Design Pattern Consistency**: Architectural decisions propagate across all components
- ✅ **Interface Compatibility**: Data structures and APIs remain consistent across files
- ✅ **Individual Focus**: Still generates detailed analysis/code one file at a time
- ✅ **Error Isolation**: Single file failure doesn't corrupt the entire process

### **Uncertainty Boundary Management**

Each stage declares:
- **"Here's what I determined with confidence"** (main outputs)
- **"Here's where my reasoning hits limits"** (`anything_unclear`)  
- **"Downstream: design around these uncertainties"**

This prevents **assumption cascade failures** and enables **uncertainty-aware reasoning**.

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
    --paper_name paper \
    --paper_markdown_path paper.md \
    --api_base_url "https://openrouter.ai/api/v1" \
    --api_key "$OPENROUTER_API_KEY" \
    --reasoning_model "deepseek/deepseek-chat-v3-0324" \
    --coding_model "qwen/qwen-2.5-coder-32b-instruct" \
    --output_dir outputs/paper \
    --output_repo_dir repos/paper
```

---

## 📚 Pipeline Architecture

The pipeline consists of **four main phases** with **seven internal planning stages** and **accumulating context throughout**:

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

### 🔍 **Phase 2: Individual File Analysis** (🧠 **With Accumulating Context**)
**Progressive context building for global coherence:**
- **Iteration Pattern**: Each file analysis includes ALL previous analyses in context
- **Core Functionality**: What each file implements from the paper
- **Implementation Strategy**: Technical approach consistent with previous files
- **Dependencies & Data Flow**: Integration with other components
- **Focused Requirements**: Methods, classes, and interaction specifications
- **Coherence Guidance**: Explicit consistency checks with established patterns

### 📁 **Phase 3: File Organization**
Dependency-aware development ordering:
- **File Type Classification**: Utilities, classes, and main orchestration
- **Priority Assessment**: Critical path and dependency mapping
- **Development Order**: Optimal sequence (utilities → classes → main)

### 💻 **Phase 4: Code Generation** (🧠 **With Enhanced Context Accumulation**)
**Structured code generation with progressive context:**
- **Context Management**: Full code for recent files, signatures for older files
- **Integration Guidance**: Explicit consistency with previously generated components
- **Structured Implementation**: Complete code with deliberation reasoning
- **Diff Output**: Version control ready format
- **Parallel Processing**: Concurrent file generation with intelligent context sharing
- **Quality Validation**: Import checks and basic functionality testing

## 📊 Output Structure

### Complete Pipeline Outputs
```
outputs/paper/
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
├── 🔍 Analysis & Organization Artifacts (🧠 Context-Aware)
│   ├── {filename}_simple_analysis_response.json      # Raw per-file analysis with context
│   ├── {filename}_simple_analysis_structured.json    # Structured per-file analysis
│   ├── file_organization_response.json               # Raw file organization
│   └── file_organization_structured.json             # Development order
│
├── 💻 Coding Artifacts (🧠 Context-Enhanced)
│   ├── structured_code_responses/            # Structured code with deliberation
│   │   └── {filename}_structured.json       # Deliberation + utility + code
│   ├── coding_artifacts/                     # Implementation reasoning
│   │   ├── {filename}_coding.txt            # Full coding response with context
│   │   └── {filename}_deliberation.txt      # Reasoning + utility
│   ├── diffs/                               # Version control diffs
│   │   └── {filename}.diff                  # Git-ready diff format
│   └── coding_results.json                  # Success/failure summary with context metrics
│
└── 📄 Final Repository (Globally Coherent)
repos/paper_repo/
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

#### Analysis Phase Results (🧠 **Context-Aware**)
- **Individual File Analysis**: Detailed implementation strategy per file with global coherence
- **Core Functionality**: What each file implements from the paper
- **Focused Requirements**: Methods, classes, and interaction specifications
- **Technical Considerations**: Dependencies, performance, error handling
- **Coherence Tracking**: How each file relates to previously analyzed components

#### File Organization Results
- **Development Order**: Utilities → Classes → Main processing
- **File Type Classification**: Priority and dependency analysis
- **Implementation Rationale**: Why files are ordered this way

#### Coding Phase Results (🧠 **Context-Enhanced**)
- **Structured Code**: Complete implementations with deliberation reasoning
- **Diff Files**: Clean version control integration ready for git
- **Parallel Results**: Success/failure tracking across all files
- **Context Sharing**: Previously implemented files available to subsequent generation
- **Integration Validation**: Consistency checks with established patterns

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
--resume_from_analysis  # Skip planning and resume from analysis phase if data exists
--resume_from_coding    # Skip to coding phase and resume from existing generated files  
--force_regenerate      # Force regeneration of all files (ignore existing generated files)
```

**Resume Logic:**
- **Auto-detection**: System detects completed phases and resumes automatically
- **Phase-level resume**: Skip planning/analysis phases if already completed  
- **File-level resume**: Completed files are detected and skipped during code generation
- **Context Restoration**: Accumulating context is rebuilt from existing analyses/code
- **Ctrl-C safe**: Interrupt at any point and resume exactly where you left off

### Model Recommendations

#### 🌐 **OpenAI-Compatible APIs**
```bash
# Current example models (256K+ context recommended for accumulating context)
--reasoning_model "deepseek/deepseek-chat-v3-0324"     # Strategic planning
--coding_model "qwen/qwen-2.5-coder-32b-instruct"     # Code generation
```

**Context Requirements**: The accumulating context pattern works best with models supporting 256K+ tokens for optimal global coherence.

### API Client Features
- **Timeout Handling**: Configurable request timeouts with retry logic
- **Generation Settings Rotation**: Balanced → Precise → Creative on retries
- **Seed Management**: Deterministic generation with automatic seed incrementation
- **Streaming Support**: Real-time response monitoring with repetition detection
- **Context Management**: Intelligent handling of large accumulating contexts

---

## 🤖 AutoGen Integration

AutoGen multi-agent collaboration is implemented but commented out for future use:
- Complete implementation in `functions.py` 
- Multi-agent approach with Engineer, Critic, CodeExecutor, Manager
- To activate: uncomment `run_autogen_coding_phase()` in main.py

---

## 📈 Performance Notes

- **Planning Phase**: ~5-12 minutes (7 strategic stages)
- **Analysis Phase**: ~1-2 minutes per file (with accumulating context)
- **File Organization**: ~30 seconds
- **Coding Phase**: ~3-8 minutes per file (with enhanced context)
- **Total Time**: ~25-45 minutes for typical papers (5-15 files)
- **Context Overhead**: Minimal with 256K+ context models

**Tips**: 
- Use `--resume_from_analysis` to skip planning when experimenting with different models
- Accumulating context works best with high-context models (256K+ tokens)
- Context accumulation provides significant quality improvements for multi-file projects

---

## 🙏 Acknowledgments

This enhanced version builds upon the excellent foundation provided by the original **PaperCoder** research:

> 📄 **Original Paper**: [Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning](https://arxiv.org/abs/2504.17192)  
> 🛠️ **Original Repository**: [https://github.com/going-doer/Paper2Code](https://github.com/going-doer/Paper2Code)  
> 👥 **Original Authors**: Outstanding work by the Paper2Code research team

### What We Enhanced
- **🧠 Accumulating Context Pattern**: Global coherence through progressive context building (breakthrough innovation)
- **Predicates-First Methodology**: FOL logic approach - actions before entities P(S,[O])
- **Configuration-Driven Pipeline**: Russian nested dolls architecture eliminates hardcoded stage management
- **Seven-Stage Strategic Planning**: Planning → Six Hats → Dependencies → Code Structure → Architecture → Tasks → Config
- **🎯 Uncertainty Boundary Management**: Systematic propagation and handling of "unclear" items
- **File-Level Resume Capability**: Skip completed files automatically on restart with context restoration
- **Smart File Organization**: Priority-driven development ordering
- **Structured Code Generation**: Deliberation + utility + diff format for complete traceability
- **Robust Error Recovery**: Generation setting rotation with automatic retry logic
- **AutoGen Integration**: Multi-agent collaboration framework (future direction)

### **🧠 Accumulating Context: Our Key Innovation**

The **Accumulating Context Pattern** represents a significant advancement in multi-step LLM reasoning:
- **Global Coherence**: Every iteration benefits from all previous decisions
- **Uncertainty Consistency**: Systematic handling of ambiguous requirements
- **Design Pattern Propagation**: Architectural decisions cascade properly
- **Interface Compatibility**: APIs and data structures remain consistent
- **Zero Redundancy**: No true overlaps, only refinement and specialization

This pattern can be applied to any multi-iteration LLM workflow requiring global coherence.

---

## 🐛 Known Issues

- **AutoGen Integration**: Multi-agent collaboration framework is implemented but commented out (ready for activation)

---