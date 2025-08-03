# prompts.py
"""
JSON Schema definitions and prompt templates for the whiteboard-based planning pipeline
"""

from imports import *

def format_config_as_tokens(config_state):
    if not config_state.strip():
        return "No previous configuration state available."
    
    # Wrap each top-level section in XML-like tokens
    lines = config_state.split('\n')
    result = []
    current_section = None
    
    for line in lines:
        if line and not line.startswith(' ') and ':' in line:
            if current_section:
                result.append(f"</{current_section}>")
            current_section = line.split(':')[0].strip()
            result.append(f"<{current_section}>")
            result.append(line)
        else:
            result.append(line)
    
    if current_section:
        result.append(f"</{current_section}>")
    
    return '\n'.join(result)

# JSON Schema Definitions with Whiteboard Updates
PLANNING_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "intent": {
            "type": "string",
            "description": "Primary objective of the implementation plan"
        },
        "entities": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 3,
            "description": "Key actors, components or concepts from the plan"
        },
        "predicates": {
            "type": "array", 
            "items": {"type": "string"},
            "maxItems": 5,
            "description": "Core interactions and relationships from the plan"
        },
        "outline": {"type": "string", "description": "Freeform implementation thoughts"},
        "implied_requirements": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Necessary conditions for implementation"
        },
        "features": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 13,
            "description": "Necessary features to achieve the implementation goals"
        },
        "datasets": {
            "type": "array",
            "items": {"type": "string"}
        },
        "methods": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Function definition headers only"
        },
        "abstract_classes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Class names and init variables"
        },
        "experimental_setup": {"type": "string"},
        "model_architecture": {"type": "string"},
        "hyper_params": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": "string"}
                },
                "required": ["name", "value"]
            }
        },
        "evaluation_metrics": {
            "type": "array",
            "items": {"type": "string"}
        },
        "remaining_questions": {
            "type": "array",
            "items": {"type": "string"}
        },
        "summary": {"type": "string"},
        "updates": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Whiteboard updates in key.value format, use '' to delete keys"
        }
    },
    "required": ["title", "intent", "entities", "predicates", "features", 
                "methods", "datasets", "experimental_setup", "model_architecture", 
                "hyper_params", "evaluation_metrics", "remaining_questions", 
                "outline", "summary", "implied_requirements", "abstract_classes", "updates"],
    "additionalProperties": False
}

SIX_HATS_SCHEMA = {
    "type": "object",
    "properties": {
        "white_hat": {
            "type": "string",
            "description": "White Hat: Facts, data and known information from the plan"
        },
        "blue_hat": {
            "type": "string", 
            "description": "Blue Hat: Process control and thinking about thinking"
        },
        "black_hat": {
            "type": "string",
            "description": "Black Hat: Potential risks and limitations in the plan"
        },
        "red_hat": {
            "type": "string",
            "description": "Red Hat: Intuitive concerns and emotional responses"
        },
        "yellow_hat": {
            "type": "string",
            "description": "Yellow Hat: Benefits and opportunities in the approach"
        },
        "green_hat": {
            "type": "string",
            "description": "Green Hat: Alternative approaches and creative solutions"
        },
        "critical_path": {
            "type": "string",
            "description": "Priority sequence for implementation based on strategic analysis"
        },
        "risk_mitigation": {
            "type": "string",
            "description": "Strategies to address identified risks from black hat analysis"
        },
        "updates": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Whiteboard updates in key.value format, use '' to delete keys"
        }
    },
    "required": ["white_hat", "blue_hat", "black_hat", "red_hat", 
                "yellow_hat", "green_hat", "critical_path", "risk_mitigation", "updates"],
    "additionalProperties": False
}
#not used
UML_SCHEMA = {
    "type": "object",
    "properties": {
        "class_diagram": {
            "type": "string",
            "description": "Mermaid classDiagram syntax showing classes, attributes, methods, and relationships"
        },
        "component_diagram": {
            "type": "string", 
            "description": "Mermaid graph syntax showing system components and their interactions"
        },
        "key_classes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "class_name": {"type": "string"},
                    "purpose": {"type": "string"},
                    "key_methods": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "key_attributes": {
                        "type": "array", 
                        "items": {"type": "string"}
                    }
                },
                "required": ["class_name", "purpose", "key_methods", "key_attributes"]
            }
        },
        "component_interactions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from_component": {"type": "string"},
                    "to_component": {"type": "string"},
                    "interaction_type": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["from_component", "to_component", "interaction_type", "description"]
            }
        },
        "design_rationale": {"type": "string"}
    },
    "required": ["class_diagram", "component_diagram", "key_classes", 
                "component_interactions", "design_rationale"],
    "additionalProperties": False
}

DEPENDENCY_SCHEMA = {
    "type": "object",
    "properties": {
        "deliberation": {
            "type": "string",
            "description": "Reasoning for the feature ranking and dependency order"
        },
        "ranked": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rank": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Priority rank based on dependencies and critical path"
                    },
                    "title": {
                        "type": "string",
                        "description": "Feature name from planning phase"
                    },
                    "utility": {
                        "type": "string", 
                        "enum": ["high", "medium", "low"],
                        "description": "Value delivered by this feature"
                    },
                    "effort": {
                        "type": "string",
                        "enum": ["high", "medium", "low"], 
                        "description": "Implementation complexity"
                    },
                    "affected_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of files that need to be created or modified for this feature"
                    }
                },
                "required": ["rank", "title", "utility", "effort", "affected_files"]
            },
            "description": "Ranked features with dependency-aware prioritization and file mapping"
        },
        "updates": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Whiteboard updates in key.value format, use '' to delete keys"
        }
    },
    "required": ["deliberation", "ranked", "updates"],
    "additionalProperties": False
}

CODE_STRUCTURE_SCHEMA = {
    "type": "object",
    "properties": {
        "utility_functions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of utility function names and brief descriptions (one line each)"
        },
        "class_headers": {
            "type": "array", 
            "items": {"type": "string"},
            "description": "List of class names with their initialization parameters (one line each)"
        },
        "class_member_functions": {
            "type": "array",
            "items": {"type": "string"}, 
            "description": "List of class member function headers with brief descriptions (one line each)"
        },
        "main_processing": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of main processing steps in execution order (one line each)"
        },
        "file_structure": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of recommended file names and their primary purpose (one line each)"
        },
        "updates": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Whiteboard updates in key.value format, use '' to delete keys"
        }
    },
    "required": ["utility_functions", "class_headers", "class_member_functions", "main_processing", "file_structure", "updates"],
    "additionalProperties": False
}

ARCHITECTURE_SCHEMA = {
    "type": "object",
    "properties": {
        "implementation_approach": {"type": "string"},
        "file_list": {
            "type": "array",
            "items": {"type": "string"}
        },
        "data_structures_and_interfaces": {"type": "string"},
        "program_call_flow": {"type": "string"},
        "anything_unclear": {"type": "string"},
        "updates": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Whiteboard updates in key.value format, use '' to delete keys"
        }
    },
    "required": ["implementation_approach", "file_list", 
                "data_structures_and_interfaces", "program_call_flow", 
                "anything_unclear", "updates"],
    "additionalProperties": False
}

TASK_LIST_SCHEMA = {
    "type": "object",
    "properties": {
        "required_packages": {
            "type": "array",
            "items": {"type": "string"}
        },
        "required_other_language_third_party_packages": {
            "type": "array",
            "items": {"type": "string"}
        },
        "logic_analysis": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 2
            }
        },
        "task_list": {
            "type": "array",
            "items": {"type": "string"}
        },
        "task_metadata": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "critical_path": {"type": "boolean"},
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                    "utility": {"type": "string", "enum": ["high", "medium", "low"]},
                    "effort": {"type": "string", "enum": ["high", "medium", "low"]}
                },
                "required": ["filename", "critical_path", "priority", "utility", "effort"]
            }
        },
        "full_api_spec": {"type": "string"},
        "shared_knowledge": {"type": "string"},
        "anything_unclear": {"type": "string"},
        "updates": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Whiteboard updates in key.value format, use '' to delete keys"
        }
    },
    "required": ["required_packages", "required_other_language_third_party_packages",
                "logic_analysis", "task_list", "task_metadata", "full_api_spec", 
                "shared_knowledge", "anything_unclear", "updates"],
    "additionalProperties": False
}

CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "config_yaml": {"type": "string"},
        "parameter_sources": {"type": "string"},
        "missing_parameters": {"type": "string"},
        "updates": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Whiteboard updates in key.value format, use '' to delete keys"
        }
    },
    "required": ["config_yaml", "parameter_sources", "missing_parameters", "updates"],
    "additionalProperties": False
}

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "core_functionality": {"type": "string"},
        "implementation_strategy": {"type": "string"},
        "technical_considerations": {"type": "string"},
        "dependencies_and_data_flow": {"type": "string"},
        "testing_and_validation": {"type": "string"},
        "paper_specific_requirements": {"type": "string"},
        "focused_requirements": {
            "type": "object",
            "properties": {
                "methods_focus": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Method signatures and interactions this file implements"
                },
                "classes_focus": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "Class definitions and member variables this file contains"
                },
                "predicate_interactions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "How this file's methods interact with other components"
                }
            },
            "required": ["methods_focus", "classes_focus", "predicate_interactions"]
        },
        "updates": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Whiteboard updates in key.value format, use '' to delete keys"
        }
    },
    "required": ["core_functionality", "implementation_strategy", "technical_considerations",
                "dependencies_and_data_flow", "testing_and_validation", 
                "paper_specific_requirements", "focused_requirements", "updates"],
    "additionalProperties": False
}

FILE_ORGANIZATION_SCHEMA = {
    "type": "object",
    "properties": {
        "deliberation": {
            "type": "string",
            "description": "Reasoning about file categorization and ordering decisions"
        },
        "file_analysis": {
            "type": "array",
            "items": {
                "type": "object", 
                "properties": {
                    "filename": {"type": "string"},
                    "file_type": {"type": "string", "enum": ["utility", "class", "main"]},
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                    "requirements_mapped": {
                        "type": "array", 
                        "items": {"type": "string"},
                        "description": "List of requirements/features this file implements"
                    },
                    "justification": {
                        "type": "string",
                        "description": "Why this file was categorized this way"
                    }
                },
                "required": ["filename", "file_type", "priority", "requirements_mapped", "justification"]
            }
        },
        "development_order": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Files ordered by: high utilities, high classes, medium utilities, medium classes, low utilities, low classes, main"
        },
        "updates": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Whiteboard updates in key.value format, use '' to delete keys"
        }
    },
    "required": ["deliberation", "file_analysis", "development_order", "updates"],
    "additionalProperties": False
}

CODE_SCHEMA = {
    "type": "object", 
    "properties": {
        "deliberation": {
            "type": "string",
            "description": "Reasoning and thought process before providing the implementation"
        },
        "utility": {
            "type": "string", 
            "description": "Intended purpose and value proposition of the function/module (like a docstring summary)"
        },
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "Name of the file being implemented"
                    },
                    "diff_file": {
                        "type": "string",
                        "description": "Complete Python code implementation formatted as diff content"
                    }
                },
                "required": ["file_name", "diff_file"]
            },
            "description": "Array of files with their implementations as diff format"
        },
        "updates": {
            "type": "array", 
            "items": {"type": "string"},
            "description": "Whiteboard updates in key.value format"
        }
    },
    "required": ["deliberation", "utility", "files", "updates"],
    "additionalProperties": False
}

GAP_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "deliberation": {"type": "string"},
        "findings": {"type": "string"},
        "paper_fidelity_score": {"type": "number", "minimum": 0, "maximum": 1},
        "undefined_items": {
            "type": "object",
            "properties": {
                "corrected_functions": {"type": "array", "items": {"type": "string"}},
                "corrected_classes": {"type": "array", "items": {"type": "string"}},
                "corrected_constants": {"type": "array", "items": {"type": "string"}},
                "corrected_imports": {"type": "array", "items": {"type": "string"}},
                "corrected_main": {"type": "array", "items": {"type": "string"}},
                "corrected_config": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["corrected_functions", "corrected_classes", "corrected_constants", 
                        "corrected_imports", "corrected_main", "corrected_config"]
        },
        "updates": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Whiteboard updates in key.value format"
        }
    },
    "required": ["deliberation", "findings", "paper_fidelity_score", "undefined_items", "updates"],
    "additionalProperties": False
}

CATEGORY_IMPLEMENTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "implementation": {"type": "string"},
        "items_completed": {"type": "array", "items": {"type": "string"}},
        "context_summary": {"type": "string"},
        "updates": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Whiteboard updates in key.value format"
        }
    },
    "required": ["implementation", "items_completed", "context_summary", "updates"],
    "additionalProperties": False
}

FILE_CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "imports_content": {
            "type": "string",
            "description": "What should go in imports.py - library imports and dependencies"
        },
        "constants_content": {
            "type": "string", 
            "description": "What should go in constants.py - configuration values, hyperparameters"
        },
        "functions_content": {
            "type": "string",
            "description": "What should go in functions.py - UTILITY FUNCTIONS ONLY (no classes)"
        },
        "classes_content": {
            "type": "string",
            "description": "What should go in classes.py - class definitions and their methods"
        },
        "main_content": {
            "type": "string",
            "description": "What should go in main.py - high-level execution flow, orchestration"
        },
        "updates": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Whiteboard updates in key.value format"
        }
    },
    "required": ["imports_content", "constants_content", "functions_content", "classes_content", "main_content", "updates"],
    "additionalProperties": False
}

# Prompt Templates with Config-Driven Context + Original Stage Rules

def get_planning_prompt(paper_content, config_state="", whiteboard_yaml=""):
    return [
        {
            "role": "system", 
            "content": """You are an expert researcher and strategic planner with a deep understanding of experimental design and reproducibility in scientific research. 

CRITICAL: You will receive a research paper in markdown format below. You must read and analyze the ACTUAL paper content provided, not make assumptions about what paper it might be.

Your task is to create a detailed implementation plan that identifies the core components from the paper. Strategic analysis (critical path, risk mitigation) will be handled in a subsequent stage.

**CONFIG-DRIVEN CONTEXT**: You have access to current configuration state containing accumulated outputs from completed stages (if any). This enables dynamic context without hardcoded parameters.

**WHITEBOARD CONTEXT**: You have access to persistent global memory that accumulates knowledge across iterations and sessions.

**WHITEBOARD UPDATES**: At the end of your response, provide updates for the shared whiteboard in key.value format. Use dot notation for nested keys (e.g., "planning.title", "planning.entities.0"). Use empty string '' to delete a key.

Focus on:
1. Identifying PREDICATES first (actions, processes, interactions, transformations)
2. Then ENTITIES (components, agents, objects that interact via predicates)
3. FEATURES needed to implement the methodology
4. Methods are function signatures that implement predicates
5. Classes are containers for entities with member variables"""
        },
        {
            "role": "user",
            "content": f"""## Current Whiteboard State (Global Memory)
```yaml
{whiteboard_yaml}
```

## Current Configuration State (Previous Stages)
```yaml
{format_config_as_tokens(config_state)}
```

## Paper Content
{paper_content}

## Task
Analyze the paper and create a comprehensive implementation plan focusing on WHAT needs to be built (strategic HOW comes later).

## Response Requirements
Extract and structure the following information in this exact order:

**Core Elements:**
- title: Paper title
- intent: Primary objective of the implementation plan
- entities: Key actors, components or concepts (max 3)
- predicates: Core interactions and relationships (max 5)
- outline: Freeform thoughts on implementation approach
- implied_requirements: Necessary conditions for successful implementation
- features: Necessary features to achieve implementation goals (max 13)

**Technical Framework:**
- datasets: Required data sources
- methods: Function definition headers that implement the predicates
- abstract_classes: Class names with their initialization variables
- experimental_setup: How experiments should be configured
- model_architecture: Specific architecture details from paper
- hyper_params: Parameters mentioned in paper (name-value pairs)
- evaluation_metrics: Metrics used for validation
- remaining_questions: Unclear aspects needing clarification
- summary: Concise overview of the methodology

**Whiteboard Updates:**
- updates: Array of "key.value" strings to update the whiteboard (use '' to delete keys)

Focus on WHAT to implement based on the paper's actual content. Strategic priorities and risk analysis will be handled in the next phase."""
        }
    ]

def get_six_hats_prompt(paper_content, config_state="", whiteboard_yaml=""):
    """Six Hats analysis with config-driven context + original stage rules"""
    return [
        {
            "role": "system", 
            "content": """You are an expert strategic analyst using Six Thinking Hats methodology for research paper implementations.

**Data Format**: You will receive structured data in YAML format wrapped in XML-like tokens (e.g., <planning>...</planning>). Inside these tokens, data is formatted as indented key-value pairs:
- Keys are on their own lines  
- Values are indented under their keys
- Nested structures use additional indentation
- Arrays show each item indented under the key

Your task is to provide Six Thinking Hats analysis AND determine the strategic critical path and risk mitigation based on that analysis.

**Data Format**: You will receive structured data in YAML format. Inside the configuration state, data is formatted as indented key-value pairs:
- Keys are on their own lines
- Values are indented under their keys
- Nested structures use additional indentation
- Arrays show each item indented under the key

**CONFIG-DRIVEN CONTEXT**: You have access to current configuration state containing accumulated outputs from completed stages. This provides dynamic context for informed strategic analysis.

**WHITEBOARD CONTEXT**: Global memory maintains strategic insights across iterations, helping build upon previous strategic decisions.

**STRATEGIC FOCUS**: This is where strategic decisions about HOW to implement are made based on the WHAT identified in planning. Your analysis will directly influence dependency ranking, architecture decisions, and risk mitigation strategies.

**WHITEBOARD UPDATES**: Provide strategic insights in key.value format using dot notation."""
        },
        {
            "role": "user",
            "content": f"""## Current Whiteboard State (Global Memory)
```yaml
{whiteboard_yaml}
```

## Current Configuration State (Planning Stage Output)
```yaml
{format_config_as_tokens(config_state)}
```

## Original Paper
{paper_content}

## Task
Apply Six Thinking Hats strategic analysis to the accumulated planning information and determine strategic implementation approach.

Based on the planning outputs in the config state, provide strategic evaluation:

**Six Hats Analysis:**

**White Hat (Facts)**: 
- What concrete, factual information do we have from the planning stage?
- What technical requirements are clearly defined?
- What paper methodology details are explicit vs. ambiguous?
- What data and resources are available?

**Red Hat (Feelings)**: 
- What are your intuitive reactions to the planned approach?
- What feels right or concerning about the proposed implementation?
- What emotional responses does the complexity level evoke?
- What gut feelings about feasibility and risks?

**Black Hat (Caution)**: 
- What could go wrong with the planned approach?
- What are the technical risks and potential failures?
- What implementation challenges are most concerning?
- What dependencies could become blockers?
- What paper ambiguities could cause problems?

**Yellow Hat (Benefits)**: 
- What are the advantages of the planned approach?
- What opportunities and positive outcomes could result?
- What strategic value does this implementation provide?
- What competitive advantages or research contributions?

**Green Hat (Creativity)**: 
- What alternative approaches could work better?
- What creative solutions emerge for identified challenges?
- What innovative implementation strategies are possible?
- What new ideas could improve the approach?

**Blue Hat (Process)**: 
- How should we manage and control the implementation process?
- What meta-level insights about the approach emerge?
- How should we sequence the work for maximum success?
- What process controls and checkpoints are needed?

**Strategic Conclusions (Based on Hats Analysis):**

**Critical Path**: Based on the analysis above (especially Blue Hat process insights and Yellow Hat opportunities), what should be the priority sequence for implementation? Consider the features and methods identified in planning.

**Risk Mitigation**: Based on the Black Hat analysis, what specific strategies should be used to address the identified risks? How can Red Hat concerns be addressed?

**Strategic Priorities**: What should be the focus areas that maximize Yellow Hat benefits while minimizing Black Hat risks?

**Implementation Approach**: How do Green Hat creative solutions inform the overall implementation strategy?

## Response Format
- white_hat: Facts, data and known information from planning and paper
- red_hat: Intuitive concerns and emotional responses to planned approach  
- black_hat: Potential risks and limitations in the planned implementation
- yellow_hat: Benefits and opportunities in the planned approach
- green_hat: Alternative approaches and creative solutions for challenges
- blue_hat: Process control and meta-level insights about implementation management
- critical_path: Priority sequence for implementation based on strategic analysis
- risk_mitigation: Specific strategies to address identified risks from black hat analysis
- strategic_priorities: Focus areas that maximize benefits while minimizing risks
- implementation_approach: How creative solutions inform overall strategy
- updates: Array of "key.value" strings to update shared strategic knowledge

Let the six hats analysis inform your strategic conclusions about critical path, risk mitigation, and implementation approach."""
        }
    ]

def get_dependency_prompt(paper_content, config_state="", whiteboard_yaml=""):
    """Dependency analysis with config-driven context + original stage rules"""
    return [
        {
            "role": "system",
            "content": """You are an expert project manager and dependency analyst specializing in research implementations.

Your task is to analyze the features identified in planning and create a dependency-aware ranking that considers critical path, utility, effort, and file impact mapping.

**Data Format**: You will receive structured data in YAML format wrapped in XML-like tokens (e.g., <planning>...</planning>). Inside these tokens, data is formatted as indented key-value pairs:
- Keys are on their own lines  
- Values are indented under their keys
- Nested structures use additional indentation
- Arrays show each item indented under the key

**CONFIG-DRIVEN CONTEXT**: You have access to current configuration state containing planning outputs and strategic analysis. This enables informed dependency analysis without hardcoded stage parameters.

**WHITEBOARD CONTEXT**: Global memory provides accumulated insights about dependencies and strategic priorities across iterations.

Your task is to analyze the features identified in planning and strategic analysis to create a dependency-aware ranking that considers critical path, utility, effort, and file impact mapping.

**Ranking Logic**:
- Critical path features from strategic analysis that can be done immediately = rank "high"
- Critical path features blocked by dependencies = rank "medium" or "low" until dependencies resolved
- Non-critical path features ranked by utility vs effort ratio
- Root dependencies in critical chains get highest priority
- Strategic priorities from Six Hats analysis influence ranking

**File Mapping**:
- Identify which specific files need to be created/modified for each feature
- Consider file dependencies when ranking (features affecting shared files may have dependencies)
- Use file overlap to identify potential conflicts or sequential dependencies
- Reference architecture and code structure from config state

**WHITEBOARD UPDATES**: Provide dependency insights in key.value format using dot notation."""
        },
        {
            "role": "user",
            "content": f"""## Current Whiteboard State (Global Memory)
```yaml
{whiteboard_yaml}
```

## Current Configuration State (Planning + Strategic Analysis)
```yaml
{format_config_as_tokens(config_state)}
```

## Original Paper
{paper_content}

## Task
Based on the features identified in planning and insights from strategic analysis (both in config state), create a dependency-aware ranking of features with file impact mapping.

## Analysis Requirements

**From Configuration State**: Extract and analyze:
- Planning data: features, methods, abstract_classes, requirements
- Strategic insights: critical_path, risk_mitigation, hats analysis priorities
- Any other accumulated knowledge that affects dependencies

**Strategic Integration**: 
- How do strategic priorities from Six Hats analysis affect feature ranking?
- What risk mitigation strategies influence dependency ordering?
- How does the critical path from strategic analysis guide sequencing?

## Ranking Criteria

**Rank Assignment** (considering strategic analysis):
- **high**: Critical path features with no blocking dependencies (can start immediately)
  * Features identified as high priority in strategic critical path
  * Root dependencies that unblock other critical features
  * Risk mitigation features that address Black Hat concerns
- **medium**: Important features with some dependencies or moderate complexity
  * Features supporting critical path but not immediately actionable
  * Medium strategic priority items from analysis
  * Features with manageable dependency chains
- **low**: Non-critical features or those heavily dependent on others
  * Nice-to-have features not on critical path
  * Features with complex dependency chains
  * Items that can be deferred without strategic impact

**Utility Assessment** (value delivered considering strategic benefits):
- **high**: Critical path components, high strategic value, addresses key Yellow Hat benefits
- **medium**: Important supporting features, moderate strategic impact
- **low**: Nice-to-have features, minimal immediate strategic value

**Effort Assessment** (implementation complexity considering strategic risks):
- **high**: Complex algorithms, significant integration work, high technical risk from Black Hat analysis
- **medium**: Moderate complexity, some technical challenges identified in strategic analysis
- **low**: Straightforward implementation, minimal complexity, low strategic risk

**File Impact Mapping** (considering architecture from config state):
- **affected_files**: List specific Python files that need to be created or modified
- Consider file dependencies (if Feature A modifies file X and Feature B also modifies file X, there may be a dependency)
- Think about logical file groupings (related functionality should be in same files)
- Reference code structure and architecture from config state
- Include main.py, config files, utility files as appropriate
- Consider strategic priorities when grouping functionality

## Strategic Dependencies

Analyze how strategic insights affect dependencies:
- What features address critical risks and should be prioritized?
- How do Green Hat creative solutions create new dependency relationships?
- What Blue Hat process insights affect implementation sequencing?
- How do Yellow Hat benefits guide utility assessment?

## Response Format
- deliberation: Detailed reasoning for ranking decisions, incorporating strategic analysis, dependencies, critical path, file impacts, and risk considerations
- ranked: Array of objects with rank, title, utility, effort, and affected_files for each feature, ordered by strategic priority and dependencies
- strategic_integration: How strategic analysis influenced the dependency ranking
- risk_considerations: How identified risks affect the dependency ordering
- critical_path_analysis: How the strategic critical path guides the ranking
- updates: Array of "key.value" strings to update shared dependency knowledge

Focus on creating a practical implementation roadmap that respects dependencies, incorporates strategic priorities, maps file impacts, and maximizes early value delivery while addressing strategic risks."""
        }
    ]

def get_uml_prompt(paper_content, planning_response, six_hats_response, dependency_response):
    return [
        {
            "role": "system",
            "content": """You are an expert software architect and UML designer specializing in research paper implementations.

Your task is to create comprehensive UML diagrams that will guide the architecture design and implementation team.

**Token Format**: You will receive structured data wrapped in XML-like tokens (e.g., <planning>...</planning>). Inside these tokens, data is formatted as indented key-value pairs without colons or braces:
- Keys are on their own lines
- Values are indented under their keys
- Nested structures use additional indentation
- Arrays show each item indented under the key

Focus on:
1. Translating the research methodology into clear class structures
2. Defining component relationships and data flow
3. Creating a foundation for architectural decisions
4. Providing practical implementation guidance
5. Considering the dependency ranking for design priorities

These diagrams will inform the subsequent architecture design and file structure."""
        },
        {
            "role": "user",
            "content": f"""## Original Paper
{paper_content}

## Implementation Plan
{planning_response}

## Strategic Analysis (Six Hats)
{six_hats_response}

## Dependency Analysis
{dependency_response}

## Task
Based on the paper methodology, strategic analysis, and dependency ranking above, create detailed UML diagrams that will serve as the foundation for architecture design.

## Requirements
- Create a **Class Diagram** showing all major classes derived from the paper's methodology
- Create a **Component Diagram** showing high-level system components and interactions
- Focus on the ACTUAL paper methodology - translate research concepts into software design
- Use proper UML notation in Mermaid syntax
- Design should support the strategic priorities and dependency ranking identified
- Prioritize high-ranked features in the class and component design

## Response Format
- class_diagram: Complete mermaid classDiagram with classes, attributes, methods, relationships
- component_diagram: Mermaid graph showing components and their connections
- key_classes: Array of objects with class_name, purpose, key_methods, key_attributes
- component_interactions: Array of objects describing how components interact
- design_rationale: Explanation of how the design supports the paper's methodology and dependency priorities

**Foundation for Architecture**: These diagrams will inform file structure, interfaces, and system architecture."""
        }
    ]


def get_code_structure_prompt(paper_content, config_state="", whiteboard_yaml=""):
    """Code structure design with config-driven context + original stage rules"""
    return [
        {
            "role": "system", 
            "content": """You are a software architect creating a high-level code structure outline for research implementations.

**Data Format**: You will receive structured data in YAML format wrapped in XML-like tokens (e.g., <planning>...</planning>). Inside these tokens, data is formatted as indented key-value pairs:
- Keys are on their own lines  
- Values are indented under their keys
- Nested structures use additional indentation
- Arrays show each item indented under the key

**CONFIG-DRIVEN CONTEXT**: You have access to current configuration state containing planning outputs, strategic analysis, and dependency ranking. This enables informed code structure design.

**WHITEBOARD CONTEXT**: Global memory provides accumulated architectural insights and design patterns across iterations.

IMPORTANT: Do NOT write actual code. Only provide one-line descriptions and headers.

Based on the accumulated config data, create structured sections for:
- utility_functions: Function names with brief purpose (standalone functions only)
- class_headers: Class names with __init__ parameters (entity containers)
- class_member_functions: Method names with brief descriptions (predicate implementations)
- main_processing: Execution flow steps (orchestration sequence)
- file_structure: Recommended file organization (dependency-aware grouping)

Keep each item to ONE LINE maximum. Focus on structure, not implementation. Informed by strategic priorities and dependency analysis.

**WHITEBOARD UPDATES**: Provide architectural insights in key.value format."""
        },
        {
            "role": "user",
            "content": f"""## Current Whiteboard State (Global Memory)
```yaml
{whiteboard_yaml}
```

## Current Configuration State (Planning + Strategic + Dependencies)
```yaml
{format_config_as_tokens(config_state)}
```

## Original Paper
{paper_content}

## Task
Create a code structure outline based on the accumulated configuration knowledge from planning, strategic analysis, and dependency ranking.

## Design Requirements

**From Configuration State**: Leverage accumulated knowledge:
- Planning: features, methods, abstract_classes, predicates, entities
- Strategic: critical_path priorities, risk mitigation requirements
- Dependencies: high/medium/low ranking, utility vs effort analysis
- Any architectural insights from previous iterations

**Strategic Alignment**:
- Prioritize code structure for high-ranked features from dependency analysis
- Address risk mitigation requirements from strategic analysis in the structure
- Ensure critical path features have clear structural representation
- Consider utility vs effort when designing interfaces

**Code Structure Sections**:

**utility_functions** (standalone functions, predicates implementation):
- Focus on high-utility, low-effort functions first
- Include risk mitigation utilities identified in strategic analysis
- One line per function: "function_name - brief purpose"

**class_headers** (entity containers with initialization):
- Design classes that represent entities from planning
- Prioritize classes needed for critical path features
- One line per class: "ClassName(init_param1, init_param2) - purpose"

**class_member_functions** (predicate implementations as methods):
- Focus on methods that implement predicates from planning
- Prioritize methods for high-ranked features
- Include methods that address strategic risk mitigation
- One line per method: "ClassName.method_name - brief purpose"

**main_processing** (execution orchestration):
- Sequence based on dependency ranking and critical path
- Include strategic checkpoints and risk mitigation steps
- One line per step: "step description - purpose"

**file_structure** (dependency-aware organization):
- Group functionality based on dependency analysis
- Separate high-priority critical path components
- Consider file impact mapping from dependency analysis
- One line per file: "filename.py - primary purpose and key components"

## Response Format
- utility_functions: Array of one-line function descriptions prioritized by strategic importance
- class_headers: Array of one-line class definitions aligned with critical path needs
- class_member_functions: Array of one-line method descriptions supporting high-ranked features
- main_processing: Array of one-line execution steps following strategic sequence
- file_structure: Array of one-line file descriptions organized by dependencies and strategic priorities
- strategic_alignment: How the structure supports strategic priorities and addresses risks
- dependency_integration: How the structure reflects dependency ranking and critical path
- updates: Array of "key.value" strings to update shared architectural knowledge

Remember: ONE LINE per element, NO actual code implementation. Focus on structure that supports the strategic implementation roadmap and dependency-driven development approach."""
        }
    ]
    
def get_architecture_prompt(paper_content, config_state="", whiteboard_yaml=""):
    """System architecture design with config-driven context + original stage rules"""
    return [
        {
            "role": "system",
            "content": """You are an expert software architect specializing in research paper implementations.

**Data Format**: You will receive structured data in YAML format wrapped in XML-like tokens (e.g., <planning>...</planning>). Inside these tokens, data is formatted as indented key-value pairs:
- Keys are on their own lines  
- Values are indented under their keys
- Nested structures use additional indentation
- Arrays show each item indented under the key

**CONFIG-DRIVEN CONTEXT**: You have access to accumulated configuration state containing planning outputs, strategic analysis, dependency ranking, and code structure design. This enables comprehensive architecture design.

**WHITEBOARD CONTEXT**: Global memory provides accumulated architectural insights and design patterns across iterations.

Your task is to create a software system architecture that implements the research methodology while incorporating all accumulated strategic and structural insights.

**WHITEBOARD UPDATES**: Provide architectural insights in key.value format using dot notation."""
        },
        {
            "role": "user",
            "content": f"""## Current Whiteboard State (Global Memory)
```yaml
{whiteboard_yaml}
```

## Current Configuration State (All Previous Stages)
```yaml
{format_config_as_tokens(config_state)}
```

## Original Paper
{paper_content}

## Task
Based on all accumulated configuration knowledge, create a comprehensive software system architecture that implements the research methodology.

## Architecture Requirements

**From Configuration State**: Integrate accumulated knowledge:
- Planning: features, methods, classes, predicates, entities, requirements
- Strategic: critical_path, risk_mitigation, strategic priorities, hats analysis
- Dependencies: feature ranking, utility/effort analysis, file impact mapping
- Code Structure: utility functions, class design, processing flow, file organization

**Strategic Integration**:
- Architecture must support the critical path identified in strategic analysis
- Address risk mitigation strategies through architectural decisions
- Prioritize high-ranked features in the architectural design
- Consider Green Hat creative solutions in architectural approaches
- Include Blue Hat process controls in the architecture

**Technical Integration**:
- Align with code structure design from previous stage
- Support dependency ranking through modular design
- Enable utility vs effort optimization through component separation
- Address file impact mapping through clear interface design

## Architecture Components

**Implementation Approach**:
- How will the system implement the paper's methodology?
- How do strategic insights and dependency priorities influence the technical approach?
- What architectural patterns best support the critical path?
- How does the approach address risks identified in strategic analysis?
- How do creative solutions from Green Hat analysis inform the architecture?

**File Organization** (based on dependency analysis and code structure):
- List of Python files needed reflecting dependency ranking priorities
- Always include main.py and config.yaml
- Organize files to support critical path and minimize dependencies
- Group related functionality based on file impact mapping

**Data Structures and Interfaces**:
- Detailed interfaces based on code structure design and dependency analysis
- API contracts that support strategic priorities
- Data flow patterns that enable risk mitigation
- Interface design that supports utility maximization

**Program Call Flow**:
- Use mermaid sequenceDiagram syntax showing execution flow
- Sequence should follow critical path from strategic analysis
- Include strategic checkpoints and risk mitigation points
- Show how high-priority features interact with supporting components
- Demonstrate how the architecture supports dependency ranking

**Integration Strategy**:
- How do components integrate to support the research methodology?
- How does integration address strategic priorities and risks?
- What configuration points support flexible deployment?
- How does the architecture enable incremental implementation following dependency ranking?

## Response Format
- implementation_approach: Technical approach incorporating all accumulated insights (strategic, dependencies, code structure)
- file_list: List of Python files organized by dependency priorities (always include main.py, config.yaml)
- data_structures_and_interfaces: Detailed interfaces based on accumulated design knowledge
- program_call_flow: Mermaid sequenceDiagram showing execution flow aligned with critical path
- integration_strategy: How components work together to support research methodology and strategic goals
- strategic_alignment: How architecture supports strategic priorities and addresses identified risks
- dependency_optimization: How architecture enables dependency-driven development
- anything_unclear: List any ambiguities from paper needing clarification, informed by previous ambiguity resolution patterns
- updates: Array of "key.value" strings to update shared architectural knowledge

**Dependency-Driven Architecture**: File structure and component design should directly reflect dependency ranking priorities, strategic critical path, and accumulated design insights."""
        }
    ]

def get_task_list_prompt(paper_content, config_state="", whiteboard_yaml=""):
    """Task breakdown with config-driven context + original stage rules"""
    return [
        {
            "role": "system",
            "content": """You are an expert software project manager and architect specializing in research paper implementations.

**Data Format**: You will receive structured data in YAML format wrapped in XML-like tokens (e.g., <planning>...</planning>). Inside these tokens, data is formatted as indented key-value pairs:
- Keys are on their own lines  
- Values are indented under their keys
- Nested structures use additional indentation
- Arrays show each item indented under the key

**CONFIG-DRIVEN CONTEXT**: You have access to accumulated configuration state containing outputs from all completed stages: planning, strategic analysis, dependency ranking, code structure, and architecture design.

**WHITEBOARD CONTEXT**: Global memory provides accumulated insights about tasks, priorities, and implementation strategies across iterations.

Your task is to create a detailed task breakdown that synthesizes ALL accumulated knowledge into actionable implementation tasks.

**WHITEBOARD UPDATES**: Provide task insights in key.value format using dot notation."""
        },
        {
            "role": "user",
            "content": f"""## Current Whiteboard State (Global Memory)
```yaml
{whiteboard_yaml}
```

## Current Configuration State (All Previous Stages)
```yaml
{format_config_as_tokens(config_state)}
```

## Original Paper
{paper_content}

## Task
Based on ALL accumulated configuration knowledge, create a comprehensive task breakdown that synthesizes planning, strategic priorities, dependency ranking, code structure, and architectural design.

## Task Breakdown Requirements

**From Configuration State**: Synthesize accumulated knowledge:
- Planning: features, methods, classes, requirements, paper methodology
- Strategic: critical_path, risk_mitigation, strategic priorities, benefits/risks analysis
- Dependencies: feature ranking, utility/effort assessment, file impact mapping
- Code Structure: utility functions, class design, processing flow, file organization
- Architecture: implementation approach, interfaces, integration strategy

**Strategic Integration**:
- Order tasks according to dependency ranking (high-ranked features first)
- Address risks identified in strategic analysis through task design
- Base breakdown on SPECIFIC paper requirements and methodology
- Ensure tasks align with code structure and architecture design
- Incorporate risk mitigation strategies as explicit tasks
- Include strategic priorities and critical path considerations
- Use dependency analysis utility and effort assessments

**Task Metadata Assignment**:
- **critical_path**: Boolean indicating if task is on strategic critical path
- **priority**: high/medium/low based on dependency ranking and strategic analysis
- **utility**: high/medium/low from dependency analysis considering strategic benefits
- **effort**: high/medium/low from dependency analysis considering strategic risks

## Implementation Planning

**Package Dependencies**:
- Required Python packages based on accumulated analysis (e.g., 'torch>=1.9.0', 'transformers>=4.0.0')
- Consider strategic risk mitigation in dependency selection
- Include packages needed for architecture implementation

**Logic Analysis** (per file):
- Detailed description for each file combining:
  * Code structure insights about what belongs in each file
  * Architectural requirements for interfaces and integration
  * Strategic priorities affecting implementation approach
  * Risk mitigation requirements for each component

**Task Organization**:
- Files ordered by dependency ranking and critical path priorities
- Sequence supports strategic implementation roadmap
- Enables incremental delivery of high-utility features
- Minimizes development risks through proper sequencing

**API and Integration Specifications**:
- API documentation reflecting architectural interface design
- Integration points that support strategic priorities
- Configuration interfaces that enable risk mitigation

## Response Format
- required_packages: Python packages needed incorporating all accumulated requirements
- required_other_language_third_party_packages: Non-Python dependencies or ["No third-party dependencies required"]
- logic_analysis: Array of [filename, detailed_description] pairs synthesizing code structure, architecture, and strategic insights
- task_list: Ordered list of files implementing dependency ranking and critical path priorities
- task_metadata: Array of objects with filename, critical_path (bool), priority, utility, effort from comprehensive analysis
- full_api_spec: API documentation reflecting architectural design (can be empty string if not needed)
- shared_knowledge: Common utilities, data structures, or patterns from accumulated design knowledge
- strategic_integration: How tasks incorporate strategic priorities and risk mitigation
- dependency_roadmap: How task sequence follows dependency analysis and critical path
- anything_unclear: Missing details from paper affecting implementation, informed by accumulated ambiguity patterns
- updates: Array of "key.value" strings to update shared task knowledge

**Comprehensive Task Design**: Task breakdown should directly synthesize dependency ranking, strategic critical path, code structure design, architectural requirements, and accumulated insights into actionable implementation tasks."""
        }
    ]

def get_config_prompt(paper_content, config_state, whiteboard_yaml=""):
    """
    Config generation prompt that references current accumulated state
    
    Args:
        paper_content: Original paper content
        config_state: Current config containing all stage outputs
        whiteboard_yaml: Global persistent memory between iterations
    """
    return [
        {
            "role": "system",
            "content": """You are an expert configuration specialist for research paper implementations.

**Data Format**: You will receive structured data in YAML format wrapped in XML-like tokens (e.g., <planning>...</planning>). Inside these tokens, data is formatted as indented key-value pairs:
- Keys are on their own lines  
- Values are indented under their keys
- Nested structures use additional indentation
- Arrays show each item indented under the key

**CONFIG-DRIVEN CONTEXT**: You have access to the current accumulated configuration state containing outputs from all completed stages. This enables dynamic context injection without hardcoded parameters.

**WHITEBOARD CONTEXT**: You also have access to persistent global memory that accumulates knowledge across multiple iterations and sessions.

**WHITEBOARD UPDATES**: At the end of your response, provide updates for the shared whiteboard in key.value format. Use dot notation for nested keys. Use empty string '' to delete a key.

Your task is to extract hyperparameters, training settings, and model configurations ONLY from the paper content and accumulated stage outputs.
DO NOT fabricate values - only use what is explicitly mentioned in the paper or derived from stage analysis."""
        },
        {
            "role": "user", 
            "content": f"""## Current Whiteboard State (Global Memory)
```yaml
{whiteboard_yaml}
```

## Current Configuration State (All Stage Outputs)
```yaml
{format_config_as_tokens(config_state)}
```

## Original Paper
{paper_content}

## Task
Based on all the accumulated configuration state and global knowledge, generate a deployment-ready configuration file.

## Requirements
- Extract hyperparameters from the paper content (learning rates, batch sizes, epochs, etc.)
- Include architecture parameters from the accumulated stage outputs
- Add configuration for identified risks and mitigation strategies from strategic analysis
- Structure config to support the architecture design from accumulated outputs
- Consider dependency ranking for feature toggles or phased implementation
- Include strategic considerations from Six Hats analysis
- Add configuration sections for high-priority features identified in dependency analysis
- Include comments explaining parameter sources (paper vs. stage analysis vs. strategic decisions)

## Configuration Categories

**Model Parameters** (from paper):
- Architecture specifications mentioned in paper
- Hyperparameters explicitly stated
- Training configurations described

**Strategic Configuration** (from accumulated stages):
- Risk mitigation settings from Six Hats analysis
- Priority-based feature flags from dependency analysis
- Architecture-driven configuration sections

**Implementation Configuration** (from stage outputs):
- File paths and data locations
- Logging and monitoring settings
- Development vs. production settings
- Dependency-aware module loading

## Response Format
- config_yaml: Complete YAML configuration as a string with sections for:
  * model (architecture and hyperparameters from paper)
  * training (learning configurations from paper)
  * data (dataset paths and preprocessing)
  * strategic (risk mitigation and priority flags)
  * implementation (file paths, logging, etc.)
  * features (dependency-ranked feature toggles)
- parameter_sources: Detailed explanation of where each parameter came from (paper citations, stage analysis references)
- missing_parameters: List of parameters not specified in paper that need tuning, with suggested ranges
- strategic_considerations: How strategic analysis influenced configuration choices
- updates: Array of "key.value" strings to update shared knowledge

Focus on creating a comprehensive configuration that supports both the research methodology AND the strategic implementation approach."""
        }
    ]

def get_analysis_prompt(paper_content, config_state="", whiteboard_yaml="", todo_file_name="", todo_file_desc=""):
    """Analysis prompt with config-driven context + original stage rules + coherence guidance"""
    return [
        {
            "role": "system",
            "content": """You are an expert software architect and researcher specializing in implementing academic research.

**Data Format**: You will receive structured data in YAML format wrapped in XML-like tokens (e.g., <planning>...</planning>). Inside these tokens, data is formatted as indented key-value pairs:
- Keys are on their own lines  
- Values are indented under their keys
- Nested structures use additional indentation
- Arrays show each item indented under the key

**CONFIG-DRIVEN CONTEXT**: You have access to the current accumulated configuration state containing outputs from all completed stages. This enables dynamic context injection without hardcoded dependencies on specific stage parameters.

**WHITEBOARD CONTEXT**: You have access to persistent global memory that provides accumulated insights across iterations and sessions. This helps maintain consistency with previous analyses.

**STAGE-TO-STAGE FLOW**: Your current analysis will inform subsequent stages through config updates, ensuring information flows properly through the pipeline.

**Data Format**: You will receive structured data in YAML format. Inside the configuration state, data is formatted as indented key-value pairs:
- Keys are on their own lines
- Values are indented under their keys
- Nested structures use additional indentation
- Arrays show each item indented under the key

**COHERENCE GUIDANCE**: When prior file analyses are provided in the whiteboard, ensure your decisions are consistent with established patterns for handling unclear requirements and design choices.

Your task is to conduct detailed logic analysis for implementing each component, ensuring it accurately reproduces the research methodology while maintaining coherence with ALL accumulated knowledge from previous stages.

Focus on PREDICATES FIRST (methods/actions), then ENTITIES (classes/agents).

**COHERENCE PRINCIPLE**: When prior file analyses are provided, ensure your decisions are consistent with established patterns for handling unclear requirements and design choices.

Key principles:
1. STAY TRUE to the paper's methodology - don't add unnecessary complexity
2. MAINTAIN CONSISTENCY with accumulated config state and whiteboard knowledge 
3. Focus on PRACTICAL implementation details
4. Consider data flow, error handling, and modularity  
5. Specify interfaces between components clearly
6. Address any ambiguities or missing details from the paper
7. Use accumulated architecture knowledge as structural guide
8. Consider dependency ranking for implementation priorities
9. Leverage strategic insights from Six Hats analysis
10. Build upon previous file analyses for consistency

**WHITEBOARD UPDATES**: At the end of your response, provide updates for the shared whiteboard in key.value format. Use dot notation for nested keys (e.g., "analysis.files.filename.status"). Use empty string '' to delete a key."""
        },
        {
            "role": "user",
            "content": f"""## Current Whiteboard State (Global Memory)
```yaml
{whiteboard_yaml}
```

## Current Configuration State (All Previous Stage Outputs)
```yaml
{format_config_as_tokens(config_state)}
```

## Original Paper
{paper_content}

## Analysis Task
Analyze the implementation logic for '{todo_file_name}'{f", which is intended for '{todo_file_desc}'" if todo_file_desc.strip() else ""}.

## Requirements Analysis

Based on the accumulated configuration state and whiteboard knowledge, conduct comprehensive analysis:

### 1. Core Functionality
- What specific algorithms/methods from the paper does this file implement?
- What are the key inputs, outputs, and transformations?
- How does this relate to the dependency ranking from accumulated config state?
- How does this file's functionality complement the overall architecture?
- What strategic priorities from Six Hats analysis does this file address?
- How does this align with identified critical path from strategic analysis?
- **If prior analyses exist**: How does this file's functionality complement previously analyzed components?

### 2. Implementation Strategy  
- How should this component be structured (classes, functions)?
- What design patterns would be most appropriate given accumulated architectural decisions?
- How does it interface with other components per accumulated architecture?
- How does it align with the code structure design from config state?
- What is its priority based on dependency analysis in config state?
- How do strategic insights influence the implementation approach?
- What risk mitigation strategies should be incorporated?
- **If prior analyses exist**: Ensure consistency with established architectural patterns

### 3. Technical Considerations
- What are the computational requirements based on paper methodology?
- What error handling is needed considering identified risks?
- Are there performance considerations from strategic analysis?
- How do technical constraints from accumulated stages affect implementation?
- What monitoring or logging should be included based on strategic priorities?
- **If prior analyses exist**: Maintain consistency with established technical approaches

### 4. Dependencies and Data Flow
- What external libraries are required (from accumulated package analysis)?
- What data structures are needed (consistent with accumulated architecture)?
- How does data flow in and out of this component?
- How does it connect to other components per accumulated architecture?
- What are the dependency relationships with other files?
- How do interfaces align with accumulated design decisions?
- What configuration parameters does this file need?
- **If prior analyses exist**: Ensure data flow compatibility with previously analyzed components

### 5. Testing and Validation
- How can this component be tested independently?
- What validation checks are needed based on paper requirements?
- How does testing strategy align with accumulated quality considerations?
- What metrics should be captured for strategic monitoring?
- **If prior analyses exist**: Maintain consistency with established testing approaches

### 6. Paper-Specific Requirements
- What specific details from the paper must be preserved?
- Are there any ambiguities that need clarification?
- How do similar ambiguities get resolved based on accumulated knowledge?
- What paper methodology constraints affect implementation?
- **If prior analyses exist**: How have similar ambiguities been resolved in previous files?

### 7. Focused Requirements (Product Design)
Structure your analysis around components from accumulated config state:
- **Methods Focus**: Specific method signatures and interactions this file implements (reference accumulated methods from config)
- **Classes Focus**: Class definitions and member variables this file contains (align with accumulated class design)
- **Predicate Interactions**: How this file's methods interact with other components (consistent with accumulated architecture)
- **Dependency Priority**: How this file fits into the overall dependency ranking from config state
- **Strategic Alignment**: How this implementation supports strategic priorities and risk mitigation
- **Coherence Check**: Ensure approach is consistent with ALL accumulated knowledge patterns

### 8. Integration Considerations
- How does this file integrate with the overall system architecture?
- What configuration sections does it require?
- How should it handle different deployment environments?
- What interfaces does it expose to other components?
- How does it support the strategic implementation roadmap?

## Response Format
- core_functionality: What this file implements from the paper (reference specific paper sections)
- implementation_strategy: Technical approach incorporating accumulated architectural decisions and strategic insights
- technical_considerations: Computational requirements, error handling, performance considerations aligned with strategic priorities
- dependencies_and_data_flow: External libraries, data structures, interfaces consistent with accumulated architecture
- testing_and_validation: Testing strategy aligned with accumulated quality approach and strategic monitoring needs
- paper_specific_requirements: Paper methodology preservation and ambiguity resolution using accumulated knowledge
- focused_requirements: Structured around accumulated config state components:
  * methods_focus: Method signatures and interactions (reference accumulated methods)
  * classes_focus: Class definitions and member variables (align with accumulated design)
  * predicate_interactions: Method interactions with other components (consistent with architecture)
- strategic_alignment: How implementation supports strategic priorities and addresses identified risks
- configuration_needs: What configuration parameters this file requires
- integration_handoff: Key information for subsequent file implementations
- updates: Array of "key.value" strings to update shared whiteboard knowledge about this file's analysis

Provide practical implementation guidance that leverages ALL accumulated knowledge for robust, maintainable code that follows the accumulated architecture, dependency priorities, strategic insights, and maintains global coherence with the overall system design."""
        }
    ]

def get_file_organization_prompt(paper_content, config_state="", whiteboard_yaml=""):
    """Generate file organization prompt with config-driven context + original stage rules"""
    return [
        {
            "role": "system",
            "content": """You are an expert software architect specializing in development workflow optimization.

**Data Format**: You will receive structured data in YAML format wrapped in XML-like tokens (e.g., <planning>...</planning>). Inside these tokens, data is formatted as indented key-value pairs:
- Keys are on their own lines  
- Values are indented under their keys
- Nested structures use additional indentation
- Arrays show each item indented under the key

Your task is to analyze the files from the task breakdown and organize them into an optimal development order that respects both priority and file type dependencies.

**CONFIG-DRIVEN CONTEXT**: You have access to accumulated configuration state including task lists, analysis summaries, and dependency rankings.

**WHITEBOARD CONTEXT**: Global memory provides accumulated insights about file organization and development strategies across iterations.

**WHITEBOARD UPDATES**: At the end of your response, provide updates for the shared whiteboard in key.value format.

**Key Principles:**
1. **File Types**: Utility functions before classes before main orchestration
2. **Priority**: Process higher priority files within each type first
3. **Dependencies**: Never generate a file that depends on unbuilt files
4. **No Duplicates**: Each file appears exactly once in the final order

**File Type Categories:**
- **utility**: Files containing standalone functions, helpers, utilities (no classes)
- **class**: Files containing class definitions and their methods
- **main**: main.py and orchestration files that coordinate everything

**Development Order Logic:**
1. High priority utilities
2. High priority classes  
3. Medium priority utilities
4. Medium priority classes
5. Low priority utilities
6. Low priority classes
7. main.py (always last)"""
        },
        {
            "role": "user",
            "content": f"""## Current Whiteboard State (Global Memory)
```yaml
{whiteboard_yaml}
```

## Current Configuration State (Task Lists + Analysis)
```yaml
{format_config_as_tokens(config_state)}
```

## Original Paper
{paper_content}

## Task
Analyze each file from the accumulated task data to determine its type and priority, then create an optimal development order.

## Analysis Requirements

For each file in the accumulated task list:

### 1. File Type Classification
- **utility**: Contains only standalone functions, no classes
- **class**: Contains class definitions and methods
- **main**: Orchestration file (main.py)

### 2. Priority Mapping
- Map requirements from task metadata to files
- Assign highest priority present in any requirement touching this file
- Consider dependency relationships from accumulated config

### 3. Dependency Analysis
- Ensure utilities come before classes that use them
- Ensure classes come before main that orchestrates them
- Within same type+priority, order by internal dependencies

## Response Requirements

Provide structured analysis with:

### deliberation
Your reasoning process for categorization and ordering:
- How you classified each file type
- How you determined priorities
- Why you chose this specific ordering
- Any dependency considerations

### file_analysis  
For each file, provide:
- filename: exact filename from task list
- file_type: utility/class/main classification
- priority: high/medium/low based on requirements mapping
- requirements_mapped: which requirements/features this file implements
- justification: explanation of the categorization

### development_order
Final ordered list of filenames following the logic:
1. High priority utilities
2. High priority classes
3. Medium priority utilities  
4. Medium priority classes
5. Low priority utilities
6. Low priority classes
7. main.py (always last)

### updates
Array of "key.value" strings to update shared knowledge about file organization

Create an optimal development workflow that builds from foundation up."""
        }
    ]

def get_coding_prompt_smart_context(todo_file_name: str, detailed_logic_analysis: str,
                                   utility_description: str, paper_content: str, config_yaml: str, 
                                   shared_context: Dict[str, str], interface_context: str,
                                   max_context_tokens: int) -> List[Dict]:
    """Generate coding prompt with smart interface context and whiteboard state"""
    
    def estimate_tokens(text: str) -> int:
        return len(text) // 4
    
    # Calculate context usage
    paper_tokens = estimate_tokens(paper_content)
    interface_tokens = estimate_tokens(interface_context)
    
    # Build whiteboard context from shared_context
    whiteboard_yaml = shared_context.get('whiteboard_yaml', 'whiteboard: {}\n')
    whiteboard_tokens = estimate_tokens(whiteboard_yaml)
    
    # Essential content that must be included
    essential_content = f"""# Implementation Context

## Target File: {todo_file_name}
**Expected Utility:** {utility_description}

## Detailed Analysis for {todo_file_name}
{detailed_logic_analysis}

## Configuration
```yaml
{config_yaml}
```

# Implementation Task
Implement **{todo_file_name}** based on the research paper, analysis, and accumulated whiteboard knowledge.

## Response Requirements

### deliberation
Your reasoning process for implementing this file:
- How you approached the problem
- Key design decisions and why
- How it fits into the overall architecture

### utility  
The core value proposition of this module:
- What specific functionality it provides
- How it contributes to the overall system

### files
Array containing:
- file_name: "{todo_file_name}"
- diff_file: Complete Python implementation

### interface_exports
What this file provides to other files:
- classes: Class names and key methods
- functions: Function signatures
- constants: Important constants

### interface_imports  
What this file needs from other files:
- dependencies: Array of {{from_file, imports, usage_context}}

### integration_handoff
Key integration notes for subsequent implementations

### updates
Array of "key.value" strings to update the whiteboard (use dot notation for nested keys)

## Implementation Requirements
1. **Complete Implementation**: Write fully functional code with no placeholders
2. **Follow Architecture**: Strictly adhere to design specifications from whiteboard
3. **Paper Fidelity**: Accurately implement research methodology
4. **Code Quality**: Include docstrings, type hints, error handling
5. **Integration**: Consider interfaces with other components
6. **Whiteboard Updates**: Update shared knowledge about this file's implementation

Generate the complete structured response for {todo_file_name}."""
    
    essential_tokens = estimate_tokens(essential_content)
    
    # Reserve tokens for response (4K buffer)
    available_tokens = max_context_tokens - 4000
    used_tokens = essential_tokens + whiteboard_tokens + interface_tokens
    
    # Smart paper content management
    if used_tokens + paper_tokens <= available_tokens:
        final_paper = paper_content
        print(f"📊 Context: {used_tokens + paper_tokens:,}/{max_context_tokens:,} tokens ({(used_tokens + paper_tokens)/max_context_tokens*100:.1f}%)")
    else:
        # Use intelligent paper truncation - keep methodology sections
        remaining_tokens = available_tokens - used_tokens
        target_chars = remaining_tokens * 4  # Convert back to characters
        
        if target_chars < len(paper_content):
            # Try to keep methodology sections
            sections = paper_content.split('\n## ')
            important_sections = []
            current_length = 0
            
            for section in sections:
                section_lower = section.lower()
                is_important = any(keyword in section_lower for keyword in 
                                 ['method', 'approach', 'algorithm', 'implementation', 
                                  'experiment', 'evaluation', 'model', 'architecture'])
                
                if is_important or len(important_sections) == 0:  # Always keep first
                    if current_length + len(section) <= target_chars:
                        important_sections.append(section)
                        current_length += len(section)
                    else:
                        break
            
            final_paper = '\n## '.join(important_sections) if important_sections else paper_content[:target_chars]
        else:
            final_paper = paper_content
        
        final_tokens = used_tokens + estimate_tokens(final_paper)
        print(f"📊 Context: {final_tokens:,}/{max_context_tokens:,} tokens ({final_tokens/max_context_tokens*100:.1f}%) - paper truncated")
    
    return [
        {
            "role": "system",
            "content": """You are an expert software engineer implementing academic research in Python.

Key requirements:
- Write complete, functional code (no TODOs or placeholders)
- Include proper docstrings and type hints
- Add error handling and input validation
- Follow PEP 8 style guidelines

WHITEBOARD UPDATES: At the end of your response, provide updates for the shared whiteboard in key.value format. Use dot notation for nested keys. Use empty string '' to delete a key.

IMPORTANT: Your response must include interface information for next iterations:
- interface_exports: What classes/functions/constants this file provides
- interface_imports: What this file needs from other files  
- integration_handoff: Key notes for subsequent implementations
- updates: Whiteboard updates about this file's implementation"""
        },
        {
            "role": "user",
            "content": f"""## Current Whiteboard State
```yaml
{whiteboard_yaml}
```

## Original Research Paper
{final_paper}

-----

{interface_context}

{essential_content}"""
        }
    ]
    
def get_file_classification_prompt(paper_content, config_state="", whiteboard_yaml=""):
    return [
        {
            "role": "system",
            "content": """You are a software architect who simplifies complex projects into 4 essential files.

Your task is to analyze all the accumulated whiteboard knowledge and classify everything into exactly 4 files:
- imports.py (library imports and dependencies)
- constants.py (configuration values, hyperparameters, settings)
- functions.py (utility functions, classes, core business logic)
- main.py (high-level execution flow, orchestration, entry point)

WHITEBOARD UPDATES: Provide updates to record the classification decisions."""
        },
        {
            "role": "user",
            "content": f"""## Current Whiteboard State
```yaml
{whiteboard_yaml}
```

## Current Configuration State
```yaml
{format_config_as_tokens(config_state)}
```

## Task
Based on all the accumulated whiteboard knowledge (features, analysis, task lists, etc.), classify the functionality into 4 simple files.

For each file, describe:
- **What specific functionality goes there**
- **Why it belongs in that file**
- **Key components/functions/classes it will contain**

## Requirements

**imports.py**: List all the library imports needed
- Standard library imports (os, sys, json, etc.)
- Third-party imports (torch, transformers, pandas, etc.)
- Relative imports between the 4 files

**constants.py**: Configuration and settings
- Hyperparameters from the paper
- File paths and configuration
- Model parameters
- Any constant values used across files

**functions.py**: Core business logic
- All utility functions
- Class definitions
- Core algorithms from the paper
- Data processing functions
- Model implementation

**main.py**: Execution orchestration
- High-level workflow
- Function calls in proper order
- Entry point logic
- Integration of all components

**Whiteboard Updates:**
- updates: Array of "key.value" strings to update shared knowledge

Focus on practical implementation - what would actually work as a 4-file Python project."""
        }
    ]

def get_category_implementation_prompt(category: str, items_list: List[str], whiteboard_yaml: str, paper_content: str = None) -> List[Dict]:
    """Generate prompt for implementing specific category items"""
    
    # Strip paper content if too long to avoid token limits
    paper_excerpt = paper_content
    
    # Fix the f-string backslash issue by creating the join string separately
    items_text = '\n\n'.join(items_list)
    
    # Add file structure guidance (MOVE THIS UP)
    file_structure_guidance = {
        "corrected_functions": "All functions go in functions.py",
        "corrected_classes": "All classes go in classes.py", 
        "corrected_constants": "All constants go in constants.py",
        "corrected_imports": "All imports go in imports.py",
        "corrected_main": "Main orchestration goes in main.py",
        "corrected_config": "Configuration goes in config.yaml"
    }
    
    guidance = file_structure_guidance.get(category, f"Implementation for {category}")  # ✅ NOW it's defined
    
    # Create the conditional paper content section
    paper_section = f"""## Original Paper Content (excerpt)
{paper_excerpt}

""" if paper_content else ""
    
    return [
        {
            "role": "system",
            "content": f"""You are implementing ONLY {category} improvements.

            FILE STRUCTURE: {guidance}
            
CRITICAL RULES:
1. Write COMPLETE, FUNCTIONAL code with NO placeholders
2. Include proper error handling and validation
3. Follow PEP 8 style guidelines
4. All implementations go directly into the target files"""
        },
        {
            "role": "user",
            "content": f"""## Current Whiteboard State
```yaml
{whiteboard_yaml}
```

{paper_section}## Items to Implement ({category})
{items_text}

## Task
Implement these {len(items_list)} items for {category}.

implementation: Complete code for the target file
items_completed: List of implemented items
updates: Whiteboard updates ("key=value" format)"""
        }
    ]

def get_gap_analysis_prompt(current_code_files: str, whiteboard_yaml: str) -> List[Dict]:
    """Generates gap analysis prompt with whiteboard context."""
    return [
        {
            "role": "system",
            "content": """Analyze implementation gaps against technical requirements.

Output format:
{
  "undefined_items": {
    "corrected_functions": [missing functions],
    "corrected_classes": [missing methods], 
    "corrected_constants": [missing configs],
    "corrected_imports": [missing imports],
    "corrected_main": [main.py gaps],
    "corrected_config": [config.yaml gaps]
  },
  "updates": ["key=value" whiteboard updates]
}"""
        },
        {
            "role": "user",
            "content": f"""## Architectural Context
```yaml
{whiteboard_yaml}
```

## Current Implementation
{current_code_files}  # Safely truncated

Identify ALL technical gaps preventing production readiness."""
        }
    ]