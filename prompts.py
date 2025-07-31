# prompts.py
"""
JSON Schema definitions and prompt templates for the planning pipeline
"""

from imports import *

# JSON Schema Definitions
# Updated PLANNING_SCHEMA in prompts.py

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
        "summary": {"type": "string"}
    },
    "required": ["title", "intent", "entities", "predicates", "features", 
                "methods", "datasets", "experimental_setup", "model_architecture", 
                "hyper_params", "evaluation_metrics", "remaining_questions", 
                "outline", "summary", "implied_requirements", "abstract_classes"],
    "additionalProperties": False
}

#arguably critical_path and risk_mitigation could go to dependency_schema
# Updated SIX_HATS_SCHEMA with strategic conclusions
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
        }
    },
    "required": ["white_hat", "blue_hat", "black_hat", "red_hat", 
                "yellow_hat", "green_hat", "critical_path", "risk_mitigation"],
    "additionalProperties": False
}

# Add to prompts.py

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
        }
    },
    "required": ["deliberation", "ranked"],
    "additionalProperties": False
}

# not used due to problematic responses
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

# Used instead of UML
# In functions.py, replace UML_SCHEMA with:
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
        }
    },
    "required": ["utility_functions", "class_headers", "class_member_functions", "main_processing", "file_structure"],
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
        "anything_unclear": {"type": "string"}
    },
    "required": ["implementation_approach", "file_list", 
                "data_structures_and_interfaces", "program_call_flow", 
                "anything_unclear"],
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
        "anything_unclear": {"type": "string"}
    },
    "required": ["required_packages", "required_other_language_third_party_packages",
                "logic_analysis", "task_list", "task_metadata", "full_api_spec", 
                "shared_knowledge", "anything_unclear"],
    "additionalProperties": False
}

CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "config_yaml": {"type": "string"},
        "parameter_sources": {"type": "string"},
        "missing_parameters": {"type": "string"}
    },
    "required": ["config_yaml", "parameter_sources", "missing_parameters"],
    "additionalProperties": False
}

# Prompt Templates
# Updated prompts reflecting the logical flow


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
        }
    },
    "required": ["core_functionality", "implementation_strategy", "technical_considerations",
                "dependencies_and_data_flow", "testing_and_validation", 
                "paper_specific_requirements", "focused_requirements"],
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
        }
    },
    "required": ["deliberation", "file_analysis", "development_order"],
    "additionalProperties": False
}

# Add this to your existing prompts.py file

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
        }
    },
    "required": ["deliberation", "utility", "files"],
    "additionalProperties": False
}


def get_planning_prompt(paper_content):
    return [
        {
            "role": "system", 
            "content": """You are an expert researcher and strategic planner with a deep understanding of experimental design and reproducibility in scientific research. 

CRITICAL: You will receive a research paper in markdown format below. You must read and analyze the ACTUAL paper content provided, not make assumptions about what paper it might be.

Your task is to create a detailed implementation plan that identifies the core components from the paper. Strategic analysis (critical path, risk mitigation) will be handled in a subsequent stage.

Focus on:
1. Identifying PREDICATES first (actions, processes, interactions, transformations)
2. Then ENTITIES (components, agents, objects that interact via predicates)
3. FEATURES needed to implement the methodology
4. Methods are function signatures that implement predicates
5. Classes are containers for entities with member variables"""
        },
        {
            "role": "user",
            "content": f"""## Paper Content
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

Focus on WHAT to implement based on the paper's actual content. Strategic priorities and risk analysis will be handled in the next phase."""
        }
    ]

def get_six_hats_prompt(paper_content, planning_response):
    return [
        {
            "role": "system",
            "content": """You are an expert strategic analyst who applies the Six Thinking Hats methodology to evaluate implementation plans.

Your task is to provide Six Thinking Hats analysis AND determine the strategic critical path and risk mitigation based on that analysis.

**Token Format**: You will receive structured data wrapped in XML-like tokens (e.g., <planning>...</planning>). Inside these tokens, data is formatted as indented key-value pairs without colons or braces.

This is where strategic decisions about HOW to implement are made based on the WHAT identified in planning."""
        },
        {
            "role": "user",
            "content": f"""## Original Paper
{paper_content}

## Implementation Plan
{planning_response}

## Task
Apply Six Thinking Hats analysis and determine strategic implementation approach:

**Six Hats Analysis:**

**White Hat (Facts)**: What concrete, factual information do we have? What data is available?

**Red Hat (Feelings)**: What are your intuitive reactions? What feels right or wrong about this approach?

**Black Hat (Caution)**: What could go wrong? What are the risks, problems, and potential failures?

**Yellow Hat (Benefits)**: What are the advantages? What opportunities and positive outcomes could result?

**Green Hat (Creativity)**: What alternative approaches could work? What new ideas or creative solutions emerge?

**Blue Hat (Process)**: How should we manage and control the thinking process? What meta-level insights about the approach?

**Strategic Conclusions (Based on Hats Analysis):**

**Critical Path**: Based on the analysis above (especially Blue Hat process insights and Yellow Hat opportunities), what should be the priority sequence for implementation?

**Risk Mitigation**: Based on the Black Hat analysis, what specific strategies should be used to address the identified risks?

Let the six hats analysis inform your critical path and risk mitigation decisions."""
        }
    ]

# New dependency prompt function
# Updated dependency prompt function
def get_dependency_prompt(paper_content, planning_response, six_hats_response):
    return [
        {
            "role": "system",
            "content": """You are an expert project manager and dependency analyst specializing in research implementations.

Your task is to analyze the features identified in planning and create a dependency-aware ranking that considers critical path, utility, effort, and file impact mapping.

**Token Format**: You will receive structured data wrapped in XML-like tokens (e.g., <planning>...</planning>). Inside these tokens, data is formatted as indented key-value pairs without colons or braces:
- Keys are on their own lines
- Values are indented under their keys
- Nested structures use additional indentation
- Arrays show each item indented under the key

**Ranking Logic**:
- Critical path features that can be done immediately = rank "high"
- Critical path features blocked by dependencies = rank "medium" or "low" until dependencies resolved
- Non-critical path features ranked by utility vs effort ratio
- Root dependencies in critical chains get highest priority

**File Mapping**:
- Identify which specific files need to be created/modified for each feature
- Consider file dependencies when ranking (features affecting shared files may have dependencies)
- Use file overlap to identify potential conflicts or sequential dependencies"""
        },
        {
            "role": "user",
            "content": f"""## Original Paper
{paper_content}

## Implementation Plan
{planning_response}

## Strategic Analysis (Six Hats)
{six_hats_response}

## Task
Based on the features identified in planning and insights from six hats analysis, create a dependency-aware ranking of features with file impact mapping.

## Analysis Requirements

**From Planning Data**: You have access to:
- features: The list of necessary features to implement
- critical_path: Priority sequence information
- risk_mitigation: Risk considerations
- methods: Function definitions that may inform file structure
- abstract_classes: Class definitions that may inform file structure

**From Six Hats Data**: You have strategic insights about:
- Risks and challenges (black hat)
- Benefits and opportunities (yellow hat)
- Process priorities (blue hat)

## Ranking Criteria

**Rank Assignment**:
- **high**: Critical path features with no blocking dependencies (can start immediately)
- **medium**: Important features with some dependencies or moderate complexity
- **low**: Non-critical features or those heavily dependent on others

**Utility Assessment** (value delivered):
- **high**: Core functionality, critical path components, high-impact features
- **medium**: Important supporting features, moderate impact
- **low**: Nice-to-have features, minimal immediate impact

**Effort Assessment** (implementation complexity):
- **high**: Complex algorithms, significant integration work, high technical risk
- **medium**: Moderate complexity, some technical challenges
- **low**: Straightforward implementation, minimal complexity

**File Impact Mapping**:
- **affected_files**: List the specific Python files that need to be created or modified
- Consider file dependencies (if Feature A modifies file X and Feature B also modifies file X, there may be a dependency)
- Think about logical file groupings (related functionality should be in same files)
- Include main.py, config files, utility files as appropriate

## Response Format
- deliberation: Detailed reasoning for the ranking decisions, considering dependencies, critical path, file impacts, and strategic insights
- ranked: Array of objects with rank, title, utility, effort, and affected_files for each feature

Focus on creating a practical implementation roadmap that respects dependencies, maps file impacts, and maximizes early value delivery."""
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

def get_code_structure_prompt(paper_content, planning_data, six_hats_data, dependency_data):
    return [
        {
            "role": "system", 
            "content": """You are a software architect creating a high-level code structure outline. 

IMPORTANT: Do NOT write actual code. Only provide one-line descriptions and headers.

Based on the planning data, create structured sections for:
- utility_functions: Function names with brief purpose
- class_headers: Class names with __init__ parameters  
- class_member_functions: Method names with brief descriptions
- main_processing: Execution flow steps
- file_structure: Recommended file organization

Keep each item to ONE LINE maximum. Focus on structure, not implementation."""
        },
        {
            "role": "user",
            "content": f"""Create a code structure outline for this project:

<paper_content>
{paper_content}
</paper_content>

<planning_data>
{planning_data}
</planning_data>

<six_hats_analysis>
{six_hats_data}  
</six_hats_analysis>

<dependency_analysis>
{dependency_data}
</dependency_analysis>

Provide structured sections as requested. Remember: ONE LINE per element, NO actual code implementation."""
        }
    ]
    
def get_architecture_prompt(paper_content, planning_response, six_hats_response, dependency_response, uml_response):
    return [
        {
            "role": "system",
            "content": """You are an expert software architect specializing in research paper implementations.

**Token Format**: You will receive structured data wrapped in XML-like tokens (e.g., <planning>...</planning>). Inside these tokens, data is formatted as indented key-value pairs without colons or braces:
- Keys are on their own lines
- Values are indented under their keys
- Nested structures use additional indentation
- Arrays show each item indented under the key"""
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

## UML Design
{uml_response}

## Task
Based on the paper analysis, strategic insights, dependency ranking, and UML design above, create a software system architecture that implements the methodology.

## Requirements
- Base architecture on the UML class and component design
- Incorporate insights from the strategic analysis
- Address identified risks and leverage opportunities
- Follow the dependency ranking for implementation priorities
- Create a modular, testable design aligned with UML structure
- Prioritize high-ranked features in file organization

## Response Format
- implementation_approach: Technical approach incorporating UML design, strategic insights, and dependency priorities
- file_list: List of Python files needed (based on UML components and dependency ranking, always include main.py)
- data_structures_and_interfaces: Detailed interfaces based on UML class relationships
- program_call_flow: Use mermaid sequenceDiagram syntax showing execution flow per UML design
- anything_unclear: List any ambiguities from the paper that need clarification

**Dependency-Driven Architecture**: File structure should reflect both UML design and dependency ranking priorities."""
        }
    ]

def get_task_list_prompt(paper_content, planning_response, six_hats_response, dependency_response, uml_response, architecture_response):
    return [
        {
            "role": "system",
            "content": """You are an expert software project manager and architect specializing in research paper implementations.

**Token Format**: You will receive structured data wrapped in XML-like tokens (e.g., <planning>...</planning>). Inside these tokens, data is formatted as indented key-value pairs without colons or braces:
- Keys are on their own lines
- Values are indented under their keys
- Nested structures use additional indentation
- Arrays show each item indented under the key"""
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

## UML Design
{uml_response}

## Architecture Design
{architecture_response}

## Task
Based on all the analysis above, create a detailed task breakdown that incorporates strategic priorities, dependency ranking, UML design, and architectural structure.

## Requirements
- Order tasks according to dependency ranking (high-ranked features first)
- Address risks identified in the analysis
- Base breakdown on the SPECIFIC paper's requirements
- Ensure tasks align with UML class structure and architecture design
- Use dependency analysis utility and effort assessments
- **Assign metadata for each task**:
  - **critical_path**: Boolean indicating if task is on critical path
  - **priority**: high/medium/low based on dependency ranking
  - **utility**: high/medium/low from dependency analysis
  - **effort**: high/medium/low from dependency analysis

## Response Format
- required_packages: Python packages needed (e.g., 'torch>=1.9.0', 'transformers>=4.0.0')
- required_other_language_third_party_packages: Non-Python dependencies or ["No third-party dependencies required"]
- logic_analysis: Array of [filename, detailed_description] pairs explaining what each file should implement
- task_list: Ordered list of files to implement (dependency order following ranking)
- task_metadata: Array of objects with filename, critical_path (bool), priority, utility, effort (from dependency analysis)
- full_api_spec: API documentation if needed (can be empty string)
- shared_knowledge: Common utilities, data structures, or patterns used across files
- anything_unclear: Missing details from the paper that affect implementation

**Dependency-Driven Tasks**: Task breakdown should directly reflect the dependency ranking and UML structure."""
        }
    ]

# Update get_analysis_prompt to include UML:

def get_analysis_prompt(paper_content, planning_response, six_hats_response, dependency_response, architecture_response, uml_response, task_list_response, todo_file_name, todo_file_desc=""):
    return [
        {
            "role": "system",
            "content": """You are an expert software architect and researcher specializing in implementing academic research.

Your task is to conduct detailed logic analysis for implementing each component, ensuring it accurately reproduces the research methodology.

**Token Format**: You will receive structured data wrapped in XML-like tokens (e.g., <planning>...</planning>). Inside these tokens, data is formatted as indented key-value pairs without colons or braces:
- Keys are on their own lines
- Values are indented under their keys
- Nested structures use additional indentation
- Arrays show each item indented under the key

Focus on PREDICATES FIRST (methods/actions), then ENTITIES (classes/agents).

Key principles:
1. STAY TRUE to the paper's methodology - don't add unnecessary complexity
2. Focus on PRACTICAL implementation details
3. Consider data flow, error handling, and modularity  
4. Specify interfaces between components clearly
5. Address any ambiguities or missing details from the paper
6. Use the UML design as a structural guide
7. Consider dependency ranking for implementation priorities"""
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

## UML Design
{uml_response}

## Software Architecture
{architecture_response}

## Task Breakdown
{task_list_response}

## Analysis Task
Analyze the implementation logic for '{todo_file_name}'{f", which is intended for '{todo_file_desc}'" if todo_file_desc.strip() else ""}.

## Requirements Analysis

Focus on **METHODS FIRST, CLASSES SECOND**:

### 1. Core Functionality
- What specific algorithms/methods from the paper does this file implement?
- What are the key inputs, outputs, and transformations?
- How does this relate to the dependency ranking?

### 2. Implementation Strategy  
- How should this component be structured (classes, functions)?
- What design patterns would be most appropriate?
- How does it interface with other components?
- How does it align with the UML class and component design?
- What is its priority based on dependency analysis?

### 3. Technical Considerations
- What are the computational requirements?
- What error handling is needed?
- Are there performance considerations?

### 4. Dependencies and Data Flow
- What external libraries are required?
- What data structures are needed?
- How does data flow in and out of this component?
- How does it connect to other components per the UML design?
- What are the dependency relationships with other files?

### 5. Testing and Validation
- How can this component be tested independently?
- What validation checks are needed?

### 6. Paper-Specific Requirements
- What specific details from the paper must be preserved?
- Are there any ambiguities that need clarification?

### 7. Focused Requirements (Product Design)
Structure your analysis around:
- **Methods Focus**: Specific method signatures and interactions this file implements
- **Classes Focus**: Class definitions and member variables this file contains (reference UML design)
- **Predicate Interactions**: How this file's methods interact with other components
- **Dependency Priority**: How this file fits into the overall dependency ranking

Provide practical implementation guidance for robust, maintainable code that follows the UML design and dependency priorities."""
        }
    ]
    
# Update get_config_prompt to include UML:

def get_config_prompt(paper_content, planning_response, six_hats_response, dependency_response, uml_response, architecture_response, task_list_response):
    return [
        {
            "role": "system",
            "content": """You are an expert configuration specialist for research paper implementations.

**Token Format**: You will receive structured data wrapped in XML-like tokens (e.g., <planning>...</planning>). Inside these tokens, data is formatted as indented key-value pairs without colons or braces:
- Keys are on their own lines
- Values are indented under their keys
- Nested structures use additional indentation
- Arrays show each item indented under the key"""
        },
        {
            "role": "user",
            "content": f"""## Original Paper
{paper_content}

## Implementation Plan
{planning_response}

## Strategic Analysis
{six_hats_response}

## Dependency Analysis
{dependency_response}

## UML Design
{uml_response}

## Architecture Design
{architecture_response}

## Task Breakdown
{task_list_response}

## Task
Based on all the analysis above, generate a configuration file that incorporates strategic considerations, dependency priorities, UML design, and architectural structure.

Extract hyperparameters, training settings, and model configurations ONLY from the paper content.
DO NOT fabricate values - only use what is explicitly mentioned in the paper.

## Requirements
- Include parameters mentioned in the paper
- Add configuration for identified risks and mitigation strategies
- Structure config to support the UML design and architecture
- Consider dependency ranking for feature toggles or phased implementation
- Include comments explaining parameter sources

## Response Format
- config_yaml: Complete YAML configuration as a string
- parameter_sources: Explanation of where each parameter came from
- missing_parameters: List of parameters not specified in paper that need tuning"""
        }
    ]

# prompts.py - Add CODE_SCHEMA for structured code generation
"""
Enhanced prompts.py with CODE_SCHEMA for structured code responses
"""

def get_file_organization_prompt(paper_content: str, task_list_context: str, 
                                task_list_response: str, analysis_summaries: str) -> List[Dict]:
    """Generate file organization prompt
    
    Args:
        paper_content: Original paper content (unused for this stage)
        task_list_context: Formatted task list context from dependencies (unused)
        task_list_response: The actual task list response data
        analysis_summaries: Analysis summaries for all files
    """
    
    return [
        {
            "role": "system",
            "content": """You are an expert software architect specializing in development workflow optimization.

Your task is to analyze the files from the task breakdown and their individual analysis results, then organize them into an optimal development order that respects both priority and file type dependencies.

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
            "content": f"""## Task Breakdown
{task_list_response}

## Individual File Analysis
{analysis_summaries}

## Task
Analyze each file to determine its type and priority, then create an optimal development order.

## Analysis Requirements

For each file in the task list:

### 1. File Type Classification
- **utility**: Contains only standalone functions, no classes
- **class**: Contains class definitions and methods
- **main**: Orchestration file (main.py)

### 2. Priority Mapping
- Map requirements from task metadata to files
- Assign highest priority present in any requirement touching this file
- Consider dependency relationships

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

Create an optimal development workflow that builds from foundation up."""
        }
    ]

# Updated coding prompt to use structured output
def get_coding_prompt(todo_file_name: str, detailed_logic_analysis: str,
                     utility_description: str, paper_content: str, config_yaml: str, 
                     shared_context: Dict[str, str], code_files: str = "") -> List[Dict]:
    """Generate structured coding prompt for a specific file using all available context"""
    
    # Build context sections dynamically from whatever is available
    context_sections = []
    
    # Add all context sections that exist
    for key, value in shared_context.items():
        if key.startswith('context_') and value.strip():
            # Convert context key to readable section name
            section_name = key.replace('context_', '').replace('_', ' ').title()
            context_sections.append(f"""## {section_name}
{value}

-----""")
    
    # Join all context sections
    all_context = "\n".join(context_sections)
    
    return [
        {
            "role": "system",
            "content": """You are an expert software engineer and researcher specializing in implementing academic research in Python.

You excel at:
- Translating research methodologies into clean, efficient code
- Writing modular, well-documented Python code
- Following software engineering best practices
- Implementing machine learning and data processing pipelines
- Creating robust, testable code architectures

You will provide structured responses with deliberation, utility description, and code as diff format.

Key requirements:
- Write complete, functional code (no TODOs or placeholders)
- Follow PEP 8 style guidelines
- Include proper docstrings and type hints
- Add error handling and input validation
- Make code modular and reusable
- Ensure compatibility with the specified configuration
- Follow the UML design structure and class relationships

Response format:
- deliberation: Your reasoning process and implementation approach
- utility: The core value proposition and purpose of this module
- files: Array with file_name and diff_file (complete Python code)"""
        },
        {
            "role": "user",
            "content": f"""# Implementation Context

## Target File: {todo_file_name}
**Expected Utility:** {utility_description}

## Original Research Paper
{paper_content}

-----

{all_context}

## Configuration
```yaml
{config_yaml}
```

-----

## Previously Implemented Files
{code_files}

-----

## Detailed Analysis for {todo_file_name}
{detailed_logic_analysis}

-----

# Implementation Task

Implement **{todo_file_name}** based on the research paper, analysis, and context above.

## Response Requirements

### deliberation
Your reasoning process for implementing this file:
- How you approached the problem
- Key design decisions and why
- How it fits into the overall architecture
- Any challenges or considerations

### utility  
The core value proposition of this module:
- What specific functionality it provides
- How it contributes to the overall system
- The key benefit it delivers (think main docstring summary)

### files
Array containing:
- file_name: "{todo_file_name}"
- diff_file: Complete Python implementation

## Implementation Requirements
1. **Complete Implementation**: Write fully functional code with no placeholders
2. **Follow Architecture**: Strictly adhere to the class/interface design specified
3. **Follow Design**: Implement classes and relationships as shown in the context
4. **Paper Fidelity**: Ensure the implementation accurately reflects the research methodology
5. **Code Quality**: 
   - Include proper docstrings and type hints
   - Add error handling and input validation
   - Follow PEP 8 style guidelines
   - Make code modular and testable
6. **Configuration Integration**: Use settings from config.yaml appropriately
7. **Dependencies**: Import all required libraries and handle dependencies correctly

Generate the complete structured response for {todo_file_name}."""
        }
    ]