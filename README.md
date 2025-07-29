# 📄 paper2code_structured: Structured Code Generation from Scientific Papers

![PaperCoder Overview](./assets/papercoder_overview.png)

📄 **Original work:** [Paper2Code on arXiv](https://arxiv.org/abs/2504.17192) | 🙏 **Credit:** [Original PaperCoder Repository](https://github.com/going-doer/Paper2Code)

**paper2code_structured** is a structured multi-phase pipeline that transforms scientific papers into comprehensive code repositories. Building on the excellent foundation of the original PaperCoder work, this enhanced version introduces a systematic three-phase approach with structured outputs, dependency analysis, and file organization.

## 🌟 Key Enhancements

- **📋 Structured Planning Phase**: Strategic analysis with Six Thinking Hats methodology and dependency ranking
- **🔍 Comprehensive Analysis Phase**: Individual file analysis with focused requirements
- **📁 Smart File Organization**: Dependency-aware development ordering (utilities → classes → main)
- **💻 Enhanced Coding Phase**: Structured code generation with diff outputs and validation
- **🛠️ Flexible API Support**: Compatible with OpenAI, Ollama, and other OpenAI-compatible endpoints
- **📊 Resume Capability**: Skip completed phases and resume from analysis or coding
- **🤖 Future AutoGen Integration**: Multi-agent collaboration framework (commented, ready for activation)

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
pip install requests tqdm

# For AutoGen support (future)
pip install pyautogen
```

### Basic Usage
```bash
python main.py \
    --paper_name kumo \
    --paper_markdown_path kumo_relational_foundation_model.md \
    --output_dir output \
    --output_repo_dir repo
```

---

## 📚 Pipeline Architecture

### Three-Phase Structured Pipeline

#### 🎯 **Phase 1: Strategic Planning**
Comprehensive project analysis and strategic planning:

1. **Core Planning**: Extract entities, predicates, features, and requirements
2. **Six Thinking Hats Analysis**: Strategic evaluation from multiple perspectives
3. **Dependency Analysis**: Priority ranking with utility/effort assessment
4. **Code Structure Design**: Function headers, class definitions, file organization
5. **Architecture Design**: System interfaces and data flow
6. **Task Breakdown**: Ordered file list with metadata
7. **Configuration Generation**: YAML config with hyperparameters

#### 🔍 **Phase 2: Individual Analysis**
Detailed analysis for each implementation file:

- **Core Functionality**: What the file implements from the paper
- **Implementation Strategy**: How to structure the component
- **Technical Considerations**: Performance, error handling, dependencies
- **Paper-Specific Requirements**: Methodology preservation details
- **Focused Requirements**: Methods, classes, and predicate interactions

#### 📁 **Phase 3: File Organization & Coding**
Smart development workflow with structured outputs:

1. **File Organization**: Dependency-aware ordering (utilities → classes → main)
2. **Structured Code Generation**: Complete implementations with deliberation
3. **Diff Output**: Clean diff files for version control
4. **Validation**: Import checking and execution testing
5. **Repository Creation**: Complete working codebase

---

## 🛠️ Installation & Setup

### Clone and Install
```bash
git clone <your-repo-url>
cd enhanced-papercoder
pip install -r requirements.txt
```

### API Configuration

#### 🥇 **Primary: Local Ollama (Recommended)**
The pipeline is **optimized for Ollama** - leveraging its OpenAI-compatible API for seamless local inference:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull recommended models
ollama pull hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:DeepSeek-R1-0528-Qwen3-8B-Q6_K.gguf
ollama pull kirito1/qwen3-coder:latest

# Start Ollama (runs on http://localhost:11434 by default)
ollama serve

# Run with recommended models
python main.py \
    --paper_name kumo \
    --paper_markdown_path kumo_model.md \
    --reasoning_model "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:DeepSeek-R1-0528-Qwen3-8B-Q6_K.gguf" \
    --coding_model "kirito1/qwen3-coder:latest" \
    --output_dir output \
    --output_repo_dir repo
```

**Why Ollama?**
- ✅ **Free & Private**: No API costs, data stays local
- ✅ **Fast Setup**: OpenAI-compatible API out of the box
- ✅ **No Rate Limits**: Generate as much as needed
- ✅ **Optimal for Development**: Perfect for iterative paper-to-code workflows

#### 🌐 **Cloud: OpenRouter (Multi-Model Access)**
Access top-tier models through OpenRouter's unified API:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"

# Using Kimi K2 for both reasoning and coding
python main.py \
    --paper_name attention_mechanism \
    --paper_markdown_path transformer_paper.md \
    --api_base_url "https://openrouter.ai/api/v1" \
    --api_key "$OPENROUTER_API_KEY" \
    --reasoning_model "moonshotai/kimi-k2" \
    --coding_model "moonshotai/kimi-k2" \
    --output_dir outputs/transformer \
    --output_repo_dir repos/transformer_repo
```

**Why OpenRouter?**
- 🎯 **Unified API**: Access multiple providers through one interface
- 💰 **Cost Effective**: Competitive pricing
- 📊 **Usage Analytics**: Track costs and performance

#### 🏢 **Enterprise: Direct OpenAI API**
For organizations with direct OpenAI access:

```bash
export OPENAI_API_KEY="your-openai-key"

python main.py \
    --paper_name enterprise_project \
    --paper_markdown_path technical_paper.md \
    --api_base_url "https://api.openai.com/v1" \
    --api_key "$OPENAI_API_KEY" \
    --reasoning_model "gpt-4" \
    --coding_model "gpt-4" \
    --output_dir outputs/enterprise \
    --output_repo_dir repos/enterprise_repo
```

---

## 🚀 Usage Examples

### Basic Paper Processing
```bash
# Default Ollama setup with recommended models
python main.py \
    --paper_name kumo \
    --paper_markdown_path examples/kumo_relational_foundation_model.md \
    --reasoning_model "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:DeepSeek-R1-0528-Qwen3-8B-Q6_K.gguf" \
    --coding_model "kirito1/qwen3-coder:latest" \
    --output_dir outputs/kumo \
    --output_repo_dir repos/kumo_repo
```

### With OpenRouter Cloud
```bash
# High-quality cloud generation via OpenRouter
export OPENROUTER_API_KEY="your-openrouter-key"

python main.py \
    --paper_name attention_transformer \
    --paper_markdown_path papers/attention_is_all_you_need.md \
    --api_base_url "https://openrouter.ai/api/v1" \
    --api_key "$OPENROUTER_API_KEY" \
    --reasoning_model "moonshotai/kimi-k2" \
    --coding_model "moonshotai/kimi-k2" \
    --output_dir outputs/transformer \
    --output_repo_dir repos/transformer_implementation
```

### Advanced Configuration
```bash
# Full configuration with performance tuning
python main.py \
    --paper_name research_implementation \
    --paper_markdown_path papers/research_paper.md \
    --reasoning_model "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:DeepSeek-R1-0528-Qwen3-8B-Q6_K.gguf" \
    --coding_model "kirito1/qwen3-coder:latest" \
    --output_dir outputs/research \
    --output_repo_dir repos/research_repo \
    --max_parallel 2 \
    --timeout 600 \
    --seed 42
```

### Resume from Analysis Phase
```bash
# Skip planning if already completed
python main.py \
    --paper_name kumo \
    --paper_markdown_path kumo_relational_foundation_model.md \
    --output_dir outputs/kumo \
    --output_repo_dir repos/kumo_repo \
    --resume_from_analysis
```

### High-Quality Generation (Longer Processing)
```bash
# Use stronger models for better results
python main.py \
    --paper_name research_paper \
    --paper_markdown_path paper.md \
    --reasoning_model "gpt-4" \
    --coding_model "gpt-4" \
    --api_base_url "https://api.openai.com" \
    --api_key "$OPENAI_API_KEY" \
    --timeout 900 \
    --output_dir outputs/high_quality \
    --output_repo_dir repos/research_implementation
```

---

## 📊 Output Structure

### Complete Pipeline Outputs
```
outputs/kumo/
├── 📋 Planning Artifacts
│   ├── planning_structured.json         # Core planning data
│   ├── six_hats_structured.json        # Strategic analysis
│   ├── dependency_structured.json      # Priority ranking
│   ├── code_structure_structured.json  # Code organization
│   ├── architecture_structured.json    # System design
│   ├── task_list_structured.json      # File breakdown
│   ├── config_structured.json         # Configuration
│   ├── planning_config.yaml           # Generated config
│   └── all_structured_responses.json  # Combined data
│
├── 🔍 Analysis Artifacts
│   ├── {filename}_simple_analysis_structured.json  # Per-file analysis
│   └── file_organization_structured.json           # Development order
│
├── 💻 Coding Artifacts
│   ├── structured_code_responses/      # Structured code outputs
│   ├── coding_artifacts/              # Deliberation & utilities
│   ├── diffs/                         # Version control diffs
│   └── coding_results.json           # Summary results
│
└── 📄 Repository Output
repos/kumo_repo/
├── main.py                 # Entry point
├── config.yaml            # Configuration
├── utils/                  # Utility functions
├── models/                 # Model implementations
├── data/                   # Data processing
└── evaluation/            # Evaluation scripts
```

### Key Structured Outputs

#### Planning Phase Results
- **Strategic Analysis**: Six Thinking Hats evaluation with risk mitigation
- **Dependency Ranking**: High/medium/low priority with utility/effort scores
- **File Organization**: Development order optimized for dependencies
- **Configuration**: Complete YAML with paper-specific hyperparameters

#### Analysis Phase Results
- **Individual File Analysis**: Detailed implementation strategy per file
- **Focused Requirements**: Methods, classes, and interaction specifications
- **Technical Considerations**: Dependencies, performance, error handling

#### Coding Phase Results
- **Structured Code**: Complete implementations with deliberation reasoning
- **Diff Files**: Clean version control integration
- **Validation Results**: Import checks and execution testing
- **Development Summary**: Success/failure rates with detailed metrics

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

#### 🚀 **Ollama Local (Recommended)**
```bash
# Tested and recommended combination
--reasoning_model "hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:DeepSeek-R1-0528-Qwen3-8B-Q6_K.gguf"
--coding_model "kirito1/qwen3-coder:latest"
```

#### 🌐 **OpenRouter Cloud**
```bash
# High-quality unified model
--reasoning_model "moonshotai/kimi-k2"
--coding_model "moonshotai/kimi-k2"
--api_base_url "https://openrouter.ai/api/v1"
```

---

## 🤖 Future AutoGen Integration

The pipeline includes ready-to-activate AutoGen multi-agent collaboration:

```python
# In main.py, uncomment this section for multi-agent coding:
results = run_autogen_coding_phase(
    paper_content=paper_content,
    output_dir=args.output_dir,
    output_repo_dir=args.output_repo_dir,
    api_client=api_client,
    coding_model=args.coding_model,
    development_order=file_org_data.get('development_order', []),
    cache_seed=args.seed
)
```

**AutoGen Features:**
- **Engineer Agent**: Code implementation specialist
- **Critic Agent**: Quality assurance and review
- **Executor Agent**: Code validation and testing
- **Manager Agent**: Workflow coordination
- **Real-time Collaboration**: Iterative improvement through agent feedback

---

## 🙏 Acknowledgments

This enhanced version builds upon the excellent foundation provided by the original **PaperCoder** research:

> 📄 **Original Paper**: [Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning](https://arxiv.org/abs/2504.17192)  
> 🛠️ **Original Repository**: [https://github.com/going-doer/Paper2Code](https://github.com/going-doer/Paper2Code)  
> 👥 **Original Authors**: Outstanding work by the Paper2Code research team

### What We Enhanced
- **Structured Pipeline**: Systematic three-phase approach with dependency analysis
- **API Flexibility**: Support for multiple LLM providers and local models
- **File Organization**: Smart dependency-aware development ordering
- **Resume Capability**: Efficient workflow with checkpoint recovery
- **Code Quality**: Structured outputs with validation and diff generation
- **Future-Ready**: AutoGen integration framework for multi-agent collaboration

### Core Methodology Credit
The strategic planning methodology, Six Thinking Hats analysis, and paper-to-code transformation concepts remain faithful to the original research. Our enhancements focus on engineering efficiency, structured outputs, and deployment flexibility while preserving the innovative approach of the original work.

---

## 📈 Performance Notes

- **Planning Phase**: ~2-5 minutes depending on paper complexity
- **Analysis Phase**: ~1-3 minutes per file
- **Coding Phase**: ~2-10 minutes per file depending on model and complexity
- **Total Time**: ~15-45 minutes for typical research papers (5-15 files)