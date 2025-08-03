# functions.py
"""
Core functions and classes for the whiteboard-based planning pipeline
"""

from imports import *
import yaml

import difflib
import ast
import re
  
try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
except:
    enc = None


class WhiteboardManager:
    """Manages persistent JSON whiteboard state with YAML formatting for LLM context"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.whiteboard_file = os.path.join(output_dir, "whiteboard.json")
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize whiteboard if it doesn't exist
        if not os.path.exists(self.whiteboard_file):
            self.save_whiteboard({})
    def take_snapshot(self, name: str) -> None:
        """Save current state with timestamp"""
        snapshot = {
            'timestamp': time.time(),
            'state': self.load_whiteboard(),
            'name': name
        }
        self.save_whiteboard({'last_snapshot': snapshot})
        print(f"📸 Whiteboard snapshot '{name}' saved")

    def get_snapshot_diff(self, snapshot_name: str) -> Dict[str, Any]:
        """Compare current state to snapshot"""
        current = self.load_whiteboard()
        snapshot = current.get('snapshots', {}).get(snapshot_name, {})
        
        diff = {
            'added': {},
            'removed': {},
            'changed': {}
        }
        
        # Simple diff logic - can be enhanced
        for k, v in current.items():
            if k not in snapshot:
                diff['added'][k] = v
            elif snapshot[k] != v:
                diff['changed'][k] = {'old': snapshot[k], 'new': v}
        
        for k in snapshot.keys() - current.keys():
            diff['removed'][k] = snapshot[k]
            
        return diff
    def load_whiteboard(self) -> Dict[str, Any]:
        """Load current whiteboard state"""
        try:
            with open(self.whiteboard_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def save_whiteboard(self, whiteboard: Dict[str, Any]) -> None:
        """Save whiteboard state to disk"""
        with open(self.whiteboard_file, 'w') as f:
            json.dump(whiteboard, f, indent=2)
    
    def get_whiteboard_yaml(self) -> str:
        """Get current whiteboard state formatted as YAML for LLM context"""
        whiteboard = self.load_whiteboard()
        if not whiteboard:
            return "whiteboard: {}\n"
        
        try:
            return yaml.dump(whiteboard, default_flow_style=False, sort_keys=False)
        except Exception as e:
            print(f"Warning: YAML formatting failed: {e}")
            return f"whiteboard: {json.dumps(whiteboard, indent=2)}\n"
                
    # Replace the existing _set_nested_key logic in WhiteboardManager.apply_updates() with this:

    def apply_updates(self, updates: List[str]) -> Dict[str, Any]:
        """Apply array of key.value updates to whiteboard with improved validation"""
        whiteboard = self.load_whiteboard()
        
        if not isinstance(updates, list):
            print(f"Warning: updates is not a list: {type(updates)} - {updates}")
            return whiteboard
        
        for update in updates:
            if not isinstance(update, str):
                print(f"Warning: Skipping non-string update: {type(update)} - {update}")
                continue
                
            if not update.strip():
                continue
                
            try:
                # Try key=value format first
                if '=' in update:
                    key_path, value = update.split('=', 1)
                    key_path = key_path.strip()
                    value = value.strip()
                    
                    # Handle deletion (empty value)
                    if value == '':
                        self._delete_nested_key(whiteboard, key_path)
                    else:
                        # Auto-convert common types
                        if value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                        elif value.isdigit():
                            value = int(value)
                        elif value.replace('.', '').isdigit() and value.count('.') <= 1:
                            try:
                                value = float(value)
                            except ValueError:
                                pass  # Keep as string
                        
                        self._set_nested_key(whiteboard, key_path, value)
                
                # Try key:value format (colon separator)
                elif ':' in update:
                    key_path, value = update.split(':', 1)
                    key_path = key_path.strip()
                    value = value.strip()
                    
                    if value == 'completed':
                        value = True
                    elif value == 'true':
                        value = True
                    elif value == 'false':
                        value = False
                    
                    self._set_nested_key(whiteboard, key_path, value)
                
                # If it's descriptive text, try to extract meaningful info
                else:
                    # Skip descriptive updates that don't follow key=value format
                    # Just log them for debugging but don't try to parse
                    print(f"Info: Descriptive update (not applied): {update[:50]}...")
                    continue
                    
            except Exception as e:
                print(f"Warning: Failed to apply update '{update}': {e}")
                continue
        
        self.save_whiteboard(whiteboard)
        return whiteboard
    
    def _set_nested_key(self, data: Dict[str, Any], key_path: str, value: Any) -> None:
        """Set nested dictionary key using dot notation"""
        keys = key_path.split('.')
        current = data
        
        # Navigate to parent of target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                current[key] = {}  # Overwrite non-dict values
            current = current[key]
        
        # Set the final key
        current[keys[-1]] = value
    
    def _delete_nested_key(self, data: Dict[str, Any], key_path: str) -> None:
        """Delete nested dictionary key using dot notation"""
        keys = key_path.split('.')
        current = data
        
        # Navigate to parent of target key
        try:
            for key in keys[:-1]:
                current = current[key]
            
            # Delete the final key if it exists
            if keys[-1] in current:
                del current[keys[-1]]
        except (KeyError, TypeError):
            # Key path doesn't exist, nothing to delete
            pass
    
    def add_stage_completion(self, stage_name: str, structured_data: Dict[str, Any]) -> None:
        """Add stage completion data to whiteboard"""
        updates = [
            f"pipeline.stages.{stage_name}.completed=true",
            f"pipeline.stages.{stage_name}.timestamp={time.time()}"
        ]
        
        # Add key insights from stage
        if stage_name == 'planning':
            if 'features' in structured_data:
                for i, feature in enumerate(structured_data['features']):
                    updates.append(f"knowledge.features.{i}={feature}")
                    
        elif stage_name == 'dependency':
            if 'ranked' in structured_data:
                for i, item in enumerate(structured_data['ranked']):
                    rank = item.get('rank', 'unknown')
                    title = item.get('title', 'unknown')
                    updates.append(f"knowledge.priorities.{rank}_features.{i}={title}")
        
        # Apply stage-specific updates
        self.apply_updates(updates)



class StreamMonitor:
    """External monitor that watches streaming content and can kill the stream"""
    
    def __init__(self, repetition_threshold: int = 50):
        self.content_buffer = ""
        self.should_terminate = False
        self.repetition_threshold = repetition_threshold
        self.max_line_length = 1500  # Hard limit for line length
        self.consecutive_long_lines = 0
        self.max_consecutive_long_lines = 2
        
        # Content context detection
        self.in_json_response = False
        self.json_brace_count = 0
        self.technical_terms_seen = 0
        
        # Try to import language detection
        self.langdetect_available = False
        try:
            from langdetect import detect, LangDetectException
            self.detect_lang = detect
            self.LangDetectException = LangDetectException
            self.langdetect_available = True
        except ImportError:
            try:
                import detectlang
                self.detect_lang = detectlang.detect
                self.LangDetectException = Exception
                self.langdetect_available = True
            except ImportError:
                print("⚠️ Language detection libraries not available. Install: pip install langdetect")

    def add_content(self, new_content: str) -> bool:
        """Add new content and check for repetition. Returns True if should continue."""
        self.content_buffer += new_content
        
        # Update context awareness
        self._update_content_context(new_content)
        
        # CHECK FOR CORRUPTION ON EVERY CHUNK - This is critical
        if self._detect_immediate_corruption(new_content):
            print(f"\n⚠️ StreamMonitor: Immediate corruption detected, terminating stream...")
            self.should_terminate = True
            return False
            
        # Keep buffer manageable
        if len(self.content_buffer) > 3000:
            self.content_buffer = self.content_buffer[-1500:]
            
        # Check for repetition patterns every chunk
        if len(self.content_buffer) > self.repetition_threshold:
            if self._detect_repetition():
                print(f"\n⚠️ StreamMonitor: Repetition detected, terminating stream...")
                self.should_terminate = True
                return False
        return True

    def _update_content_context(self, content: str) -> None:
        """Update context awareness based on content patterns"""
        # Detect JSON context
        self.json_brace_count += content.count('{') - content.count('}')
        
        # Check if we're likely in a JSON response
        json_indicators = ['"type":', '"properties":', '"required":', '"items":', 
                          '"description":', '"enum":', '"class_headers":', '"updates":']
        
        if any(indicator in content for indicator in json_indicators):
            self.in_json_response = True
        
        # Count technical terms that might confuse language detection
        technical_patterns = [
            '_', 'Transformer', 'Encoder', 'Parser', 'Generator', 'Sampler', 
            'Module', 'Analyzer', 'py', 'Config', 'Schema', 'API', 'ICL',
            'PQL', 'subgraph', 'embeddings', 'gradients'
        ]
        
        for pattern in technical_patterns:
            self.technical_terms_seen += content.lower().count(pattern.lower())

    def _detect_immediate_corruption(self, content: str) -> bool:
        """Enhanced corruption detection with context awareness"""
        lines = content.split('\n')
        
        # Check for extremely long lines (CRITICAL CHECK) - but be more lenient for JSON
        max_allowed = self.max_line_length * 2 if self.in_json_response else self.max_line_length
        
        for line in lines:
            if len(line) > max_allowed:
                print(f"⚠️ CRITICAL: Extremely long line detected ({len(line)} chars) - terminating")
                return True
        
        # Check minimum newline frequency (missing newlines) - more lenient for structured content
        newline_threshold = self.max_line_length * 3 if self.in_json_response else self.max_line_length * 2
        
        if len(content) > newline_threshold and '\n' not in content:
            print(f"⚠️ CRITICAL: Missing newlines in {len(content)} char chunk - terminating")
            return True
            
        # Check for garbage characters (non-printable flood) - but allow JSON special chars
        if len(content) > 100:
            non_printable = sum(1 for c in content if not c.isprintable() and c not in '\n\t\r ')
            # Be more lenient with structured content
            threshold = 0.5 if self.in_json_response else 0.3
            
            if non_printable > len(content) * threshold:
                print(f"⚠️ CRITICAL: High non-printable character ratio ({non_printable/len(content):.1%}) - terminating")
                return True
                
        return False

    def _detect_repetition(self) -> bool:
        """Context-aware repetition detection"""
        if len(self.content_buffer) < 50:
            return False
        recent_content = self.content_buffer[-300:]
        
        # Check all corruption detection methods with context awareness
        if self._detect_language_corruption(recent_content):
            return True
        if self._detect_excessive_repetition(recent_content):
            return True
        if self._detect_structural_corruption(recent_content):
            return True
            
        return False
        
    def _detect_language_corruption(self, content: str) -> bool:
        """Context-aware language detection to avoid false positives"""
        if not self.langdetect_available:
            return False
        
        # Skip language detection entirely for JSON responses with technical terms
        if self.in_json_response and self.technical_terms_seen > 5:
            print(f"🔧 Skipping language detection: JSON context with {self.technical_terms_seen} technical terms")
            return False
            
        lines = content.split('\n')
        corruption_signals = 0
        total_checks = 0
        
        for line in lines:
            clean_line = line.strip()
            
            # Skip empty lines and very short lines
            if len(clean_line) < 60:  # Increased threshold
                continue
            
            # Skip lines that look like code, JSON, or technical content
            if self._is_technical_content(clean_line):
                continue
                
            # Check substantially long lines (code or prose)
            if len(clean_line) > 80:  # Increased threshold
                try:
                    detected_lang = self.detect_lang(clean_line)
                    total_checks += 1
                    
                    # For code, we expect English or code-like content
                    if detected_lang not in ['en', 'ca', 'es', 'fr', 'de', 'nl', 'af']:
                        corruption_signals += 1
                        
                except self.LangDetectException:
                    # Language detection failed - might be technical content, not gibberish
                    if len(clean_line) > 200 and not self._is_technical_content(clean_line):
                        corruption_signals += 1
                        total_checks += 1
                except KeyboardInterrupt:
                    print("⚠️ Language detection interrupted - possible corruption")
                    return True
                except Exception as e:
                    if "timeout" in str(e).lower() or "memory" in str(e).lower():
                        print(f"⚠️ Language detection error: {e}")
                        return True
        
        # Only trigger on clear corruption with higher thresholds
        if total_checks >= 3:  # Require more samples
            corruption_rate = corruption_signals / total_checks
            # Much higher threshold for structured content
            threshold = 0.9 if self.in_json_response else 0.7
            
            if corruption_rate > threshold:
                print(f"⚠️ Language corruption detected: {corruption_rate:.1%} corrupted lines ({corruption_signals}/{total_checks})")
                return True
                
        return False

    def _is_technical_content(self, line: str) -> bool:
        """Identify technical content that shouldn't be language-checked"""
        technical_indicators = [
            # JSON structure
            line.strip().startswith(('{', '}', '"', '[')), 
            line.strip().endswith((',', ':', '{')),
            # Code patterns
            '_' in line and any(term in line for term in ['class', 'function', 'def', 'import']),
            # Technical terms
            any(term in line.lower() for term in [
                'transformer', 'encoder', 'parser', 'generator', 'sampler',
                'module', 'analyzer', 'config', 'schema', 'api', 'icl', 'pql'
            ]),
            # File paths and technical strings
            '.py' in line or '/' in line or '_' in line,
            # Method/class naming patterns
            any(char.isupper() and char.islower() for char in line),  # CamelCase
            line.count('_') > 2,  # snake_case with multiple underscores
        ]
        
        return any(technical_indicators)

    def _detect_excessive_repetition(self, content: str) -> bool:
        """Detect repetitive patterns that indicate looping/stuck generation"""
        # Check for repeated phrases
        words = content.split()
        if len(words) < 15:
            return False
            
        # Look for same word repeated many times - but be more lenient for structured content
        word_counts = {}
        for word in words[-75:]:
            if len(word) > 3:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Higher threshold for JSON/structured content
        max_repetition_threshold = 25 if self.in_json_response else 15
        max_repetition = max(word_counts.values()) if word_counts else 0
        
        if max_repetition > max_repetition_threshold:
            problematic_word = max(word_counts, key=word_counts.get)
            print(f"⚠️ Excessive word repetition: '{problematic_word}' appears {max_repetition} times")
            return True
            
        # Check for repeated lines - but allow some repetition in JSON
        lines = content.split('\n')
        if len(lines) > 3:
            recent_lines = [line.strip() for line in lines[-15:] if line.strip()]
            line_counts = {}
            for line in recent_lines:
                if len(line) > 20:
                    line_counts[line] = line_counts.get(line, 0) + 1
            
            # Higher threshold for structured content
            max_line_threshold = 8 if self.in_json_response else 4
            max_line_repetition = max(line_counts.values()) if line_counts else 0
            
            if max_line_repetition > max_line_threshold:
                repeated_line = max(line_counts, key=line_counts.get)
                print(f"⚠️ Excessive line repetition: line repeated {max_line_repetition} times")
                return True
                
        return False

    def _detect_structural_corruption(self, content: str) -> bool:
        """Detect structural issues that indicate corruption"""
        lines = content.split('\n')
        
        # Check for extremely long lines - more lenient for JSON
        long_line_threshold = 2000 if self.in_json_response else 1200
        long_line_count = 0
        
        for line_idx, line in enumerate(lines):
            if len(line) > long_line_threshold:
                long_line_count += 1
                if long_line_count >= 3:  # Allow more long lines for structured content
                    print(f"⚠️ Multiple long lines detected")
                    return True
        
        # Check for too many spaces (whitespace explosion) - but allow JSON indentation
        space_runs = []
        current_spaces = 0
        for char in content:
            if char == ' ':
                current_spaces += 1
            else:
                if current_spaces > 0:
                    space_runs.append(current_spaces)
                    # Much higher threshold for structured content
                    space_threshold = 500 if self.in_json_response else 200
                    
                    if current_spaces > space_threshold:
                        print(f"⚠️ Extreme whitespace run detected: {current_spaces} spaces")
                        return True
                current_spaces = 0
                
        # Check for character density issues - more lenient for JSON
        if len(content) > 150:
            printable_chars = sum(1 for c in content if c.isprintable())
            printable_ratio = printable_chars / len(content)
            # More lenient threshold for structured content
            threshold = 0.6 if self.in_json_response else 0.7
            
            if printable_ratio < threshold:
                print(f"⚠️ High non-printable character ratio: {printable_ratio:.1%}")
                return True
                
        return False
        
def format_dict_as_yaml_style(data: Dict[str, Any], token_name: str, indent_level: int = 0) -> str:
    """Convert dictionary to token-wrapped YAML-style indented format without braces and colons"""
    if not data:
        return f"<{token_name}>\n(empty)\n</{token_name}>"
    
    lines = [f"<{token_name}>"]
    indent = "  " * indent_level
    
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{indent}{key}")
            lines.append(_format_nested_dict(value, indent_level + 1))
        elif isinstance(value, list):
            lines.append(f"{indent}{key}")
            for item in value:
                if isinstance(item, dict):
                    lines.append(_format_nested_dict(item, indent_level + 1))
                else:
                    lines.append(f"{indent}  {item}")
        else:
            lines.append(f"{indent}{key}")
            lines.append(f"{indent}  {value}")
    
    lines.append(f"</{token_name}>")
    return "\n".join(lines)

def _format_nested_dict(data: Dict[str, Any], indent_level: int) -> str:
    """Helper function for nested dictionary formatting"""
    lines = []
    indent = "  " * indent_level
    
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{indent}{key}")
            lines.append(_format_nested_dict(value, indent_level + 1))
        elif isinstance(value, list):
            lines.append(f"{indent}{key}")
            for item in value:
                if isinstance(item, dict):
                    lines.append(_format_nested_dict(item, indent_level + 1))
                else:
                    lines.append(f"{indent}  {item}")
        else:
            lines.append(f"{indent}{key}")
            lines.append(f"{indent}  {value}")
    
    return "\n".join(lines)

def parse_structured_response(response_content: str) -> Dict[str, Any]:
    """Parse structured JSON response from API using json-repair for robust parsing"""
    from json_repair import repair_json
    
    try:
        # First try standard JSON parsing
        return json.loads(response_content)
    except json.JSONDecodeError as e:
        print(f"⚠️ Standard JSON parse failed: {e}")
        print("🔧 Attempting JSON repair...")
        
        try:
            # Use json-repair to fix malformed JSON
            repaired_json = repair_json(response_content)
            result = json.loads(repaired_json)
            print("✅ JSON repair successful!")
            return result
        except Exception as repair_error:
            print(f"❌ JSON repair also failed: {repair_error}")
            print(f"First 200 chars: {repr(response_content[:200])}")
            print(f"Last 200 chars: {repr(response_content[-200:])}")
            raise

def load_paper_content(paper_path: str) -> str:
    """Load paper content from markdown file"""
    try:
        with open(paper_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Paper file not found at {paper_path}")
        raise
    except Exception as e:
        print(f"Error reading paper file: {e}")
        raise

def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser with enhanced resume options"""
    parser = argparse.ArgumentParser(description="Whiteboard-based code generation pipeline")
    parser.add_argument('--resume_from_refinement', action='store_true',
                   help='Skip to iterative refinement phase')
    parser.add_argument('--paper_name', type=str, required=True)
    parser.add_argument('--reasoning_model', type=str, 
                       default="granite3.3:2b")
    parser.add_argument('--coding_model', type=str, 
                       default="hf.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF:DeepSeek-R1-0528-Qwen3-8B-Q6_K.gguf")
    parser.add_argument('--paper_markdown_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--output_repo_dir', type=str, required=True)
    parser.add_argument('--api_base_url', type=str, 
                       default="http://localhost:11434")
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--max_parallel', type=int, default=1,
                       help='Maximum parallel code generation tasks')
    parser.add_argument('--timeout', type=int, default=600,
                       help='Request timeout for code generation')
    parser.add_argument('--seed', type=int, default=42)
    
    # Enhanced resume options
    parser.add_argument('--resume_from_analysis', action='store_true',
                       help='Skip planning and resume from analysis phase if data exists')
    parser.add_argument('--resume_from_coding', action='store_true',
                       help='Skip to coding phase and resume from existing generated files')
    parser.add_argument('--force_regenerate', action='store_true',
                       help='Force regeneration of all files (ignore existing generated files)')
    parser.add_argument('--clear_whiteboard', action='store_true',
                       help='Clear whiteboard state and start fresh')
    
    # AutoGen validation options
    parser.add_argument('--autogen_validation_only', action='store_true',
                       help='Run only AutoGen validation phase on existing generated code')
    parser.add_argument('--enable_autogen_validation', action='store_true',
                       help='Enable AutoGen validation phase after regular coding')
    parser.add_argument('--enable_iterative_refinement', action='store_true',
                       help='Enable TRIZ-based iterative refinement of generated code')
    return parser

def check_pipeline_state(output_dir: str) -> Dict[str, bool]:
    """Check what pipeline phases have been completed by examining disk artifacts"""
    state = {
        'planning_complete': False,
        'analysis_complete': False,
        'file_organization_complete': False,
        'coding_started': False,
        'coding_complete': False,
        'refinement_complete': False,
        'autogen_validation_complete': False
    }
    
    print(f"🔍 Checking pipeline state for: {output_dir}")
    
    # Check planning completion - look for config file
    planning_config = os.path.join(output_dir, "planning_config.yaml")
    state['planning_complete'] = os.path.exists(planning_config)
    print(f"   Planning complete: {state['planning_complete']} (config file exists)")
    
    # Check analysis completion - look for analysis files
    analysis_files = []
    if os.path.exists(output_dir):
        all_files = os.listdir(output_dir)
        for f in all_files:
            if (f.endswith('_analysis_response.json') or 
                f.endswith('_simple_analysis_response.json') or
                f.endswith('_analysis_structured.json')):
                analysis_files.append(f)
    
    state['analysis_complete'] = len(analysis_files) > 0
    print(f"   Analysis complete: {state['analysis_complete']} ({len(analysis_files)} analysis files)")
    
    # Check file organization completion - look for file organization artifact
    file_org_file = os.path.join(output_dir, "file_organization_structured.json")
    state['file_organization_complete'] = os.path.exists(file_org_file)
    print(f"   File organization complete: {state['file_organization_complete']} (org file exists)")
    
    # Check coding completion - look for generated code files in repo directory
    # Infer repo directory from output directory pattern
    repo_dir = output_dir.replace('output/', 'repos/') if 'output/' in output_dir else output_dir + '_repo'
    
    if os.path.exists(repo_dir):
        py_files = [f for f in os.listdir(repo_dir) if f.endswith('.py')]
        yaml_files = [f for f in os.listdir(repo_dir) if f.endswith('.yaml')]
        
        # For simplified 5-file structure
        simplified_files = ['imports.py', 'constants.py', 'functions.py', 'classes.py', 'main.py']
        has_simplified_structure = all(f in py_files for f in simplified_files)
        
        # For any substantial code generation
        has_substantial_code = len(py_files) >= 3 and 'main.py' in py_files
        
        state['coding_complete'] = has_simplified_structure or has_substantial_code
        state['coding_started'] = len(py_files) > 0
        
        print(f"   Coding complete: {state['coding_complete']} (repo: {len(py_files)} py files)")
        print(f"   Simplified structure: {has_simplified_structure}")
        print(f"   Generated files: {py_files + yaml_files}")
    else:
        print(f"   Coding complete: False (no repo dir: {repo_dir})")
    
    # Check refinement completion - look for corrected_* files
    if os.path.exists(repo_dir):
        corrected_files = [f for f in os.listdir(repo_dir) if f.startswith('corrected_')]
        state['refinement_complete'] = len(corrected_files) > 0
        print(f"   Refinement complete: {state['refinement_complete']} ({len(corrected_files)} corrected files)")
    else:
        print(f"   Refinement complete: False (no repo dir)")
    
    # Check autogen validation completion - look for validation artifacts
    autogen_artifacts = [
        os.path.join(output_dir, "autogen_validation.json"),
        os.path.join(output_dir, "autogen_results.json"),
        os.path.join(repo_dir, "autogen_validation_complete.txt") if os.path.exists(repo_dir) else None
    ]
    
    state['autogen_validation_complete'] = any(
        artifact and os.path.exists(artifact) for artifact in autogen_artifacts
    )
    print(f"   AutoGen validation complete: {state['autogen_validation_complete']} (validation artifacts)")
    
    return state

def validate_autogen_prerequisites(output_dir: str, output_repo_dir: str) -> bool:
    """Check if prerequisites for AutoGen validation are met"""
    
    # Check if coding phase is complete
    pipeline_state = check_pipeline_state(output_dir)
    
    if not pipeline_state['coding_complete']:
        print("❌ AutoGen validation requires completed coding phase")
        print("   Run the full pipeline first or use --resume_from_coding")
        return False
    
    # Check if generated Python files exist
    if not os.path.exists(output_repo_dir):
        print(f"❌ Repository directory not found: {output_repo_dir}")
        return False
    
    py_files = [f for f in os.listdir(output_repo_dir) if f.endswith('.py')]
    if not py_files:
        print(f"❌ No Python files found in repository: {output_repo_dir}")
        return False
    
    print(f"✅ Prerequisites met for AutoGen validation")
    print(f"   Found {len(py_files)} Python files: {', '.join(py_files)}")
    return True

def print_response(response: Dict[str, Any]) -> None:
    """Print formatted response for debugging"""
    print("="*50)
    print(f"Model: {response.get('model_used', 'Unknown')}")
    print(f"Stage: {response.get('stage', 'Unknown')}")
    print("-"*50)
    
    content = response['choices'][0]['message']['content']
    
    # Try to pretty print if it's JSON
    try:
        parsed = json.loads(content)
        print(json.dumps(parsed, indent=2))
    except:
        # If not JSON, print as is
        print(content)
    
    print("="*50)

class APIClient:
    """Handles API calls to OpenAI-compatible endpoints with timeout and generation setting rotation"""
    
    def __init__(self, base_url: str = "http://localhost:11434", api_key: Optional[str] = None, 
                 initial_seed: int = 42, default_timeout: int = 180):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {"Content-Type": "application/json"}
        self.current_seed = initial_seed
        self.default_timeout = default_timeout
        
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        
        # Generation settings rotation for all attempts
        self.generation_settings = [
            {"name": "balanced", "temperature": 0.33, "top_p": 0.92, "repeat_penalty": 1.3, "frequency_penalty": 1.3, "presence_penalty": 1.1, "top_k": 55},
            {"name": "precise", "temperature": 0.13, "top_p": 0.78, "repeat_penalty": 1.4, "frequency_penalty": 1.4, "presence_penalty": 1.2, "top_k": 34},
            {"name": "creative", "temperature": 0.45, "top_p": 0.95, "repeat_penalty": 1.2, "frequency_penalty": 1.2, "presence_penalty": 1.0, "top_k": 66}        ]
        self.current_setting_index = 0
    
    def _increment_seed(self):
        """Increment seed for retry attempts"""
        self.current_seed += 1
        print(f"🎲 Incremented seed to {self.current_seed}")
    
    def _get_current_generation_settings(self) -> Tuple[str, Dict[str, float]]:
        """Get current generation settings and rotate to next"""
        current = self.generation_settings[self.current_setting_index]
        self.current_setting_index = (self.current_setting_index + 1) % len(self.generation_settings)
        settings = {k: v for k, v in current.items() if k != "name"}
        settings["seed"] = self.current_seed
        return current["name"], settings
    
    def chat_completion(self, model: str, messages: List[Dict],
                   response_format: Optional[Dict] = None,
                   timeout: int = None,
                   max_retries: int = 3,
                   stream: bool = False) -> Dict[str, Any]:
        """Make a chat completion request with timeout and retry logic"""
        
        # Use provided timeout or fall back to default
        if timeout is None:
            timeout = self.default_timeout
        
        for attempt in range(max_retries):
            try:
                # Always get settings from the iteration list and increment seed
                if attempt > 0:
                    self._increment_seed()
                
                setting_name, current_params = self._get_current_generation_settings()
                
                print(f"🎯 Using {setting_name} settings (attempt {attempt + 1}/{max_retries}, timeout: {timeout}s)")
                print(f"   Settings: temp={current_params.get('temperature')}, seed={current_params.get('seed')}")
                
                payload = {
                    "model": model,
                    "messages": messages,
                    "stream": stream,
                    **current_params
                }
                
                if response_format:
                    payload["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "structured_response",
                            "strict": True,
                            "schema": response_format
                        }
                    }
                
                # Handle different API endpoints
                if "openrouter.ai" in self.base_url:
                    url = f"{self.base_url}/chat/completions"
                else:
                    # Ollama endpoint
                    url = f"{self.base_url}/v1/chat/completions"
                
                response = requests.post(
                    url, 
                    headers=self.headers, 
                    json=payload,
                    timeout=timeout,
                    stream=stream
                )
                response.raise_for_status()
                
                # Handle streaming vs non-streaming responses
                if stream:
                    content = self._handle_streaming_response(response)
                    result = {
                        'choices': [{'message': {'content': content}}]
                    }
                else:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                
                # Validate JSON if structured format requested
                if response_format:
                    try:
                        json.loads(content)
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Invalid JSON in structured response: {e}")
                        if attempt < max_retries - 1:
                            continue
                        else:
                            print("⚠️  Returning invalid JSON after max retries")
                
                print(f"✅ Success with {setting_name}")
                return result
                            
            except requests.exceptions.Timeout:
                print(f"⏰ Timeout on attempt {attempt + 1} (waited {timeout}s)")
                if attempt < max_retries - 1:
                    print(f"🔄 Retrying with different generation settings and incremented seed...")
                    time.sleep(2)  # Brief pause before retry
                else:
                    raise Exception(f"Request timed out after {max_retries} attempts ({timeout}s each)")
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ API request failed on attempt {attempt + 1}: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Response content: {e.response.text}")
                if attempt < max_retries - 1:
                    print(f"🔄 Retrying...")
                    time.sleep(2)
                else:
                    raise

    def _handle_streaming_response(self, response) -> str:
        """Handle streaming response with external monitor"""
        full_content = ""
        monitor = StreamMonitor()
        print(f"🔄 Streaming response...")
        
        try:
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data: '):
                        data_text = line_text[6:]
                        if data_text.strip() == '[DONE]':
                            break
                        try:
                            data = json.loads(data_text)
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    print(content, end='', flush=True)
                                    full_content += content
                                    
                                    # EXTERNAL MONITOR - check if we should continue
                                    if not monitor.add_content(content):
                                        print(f"\n⚠️ Stream terminated by monitor")
                                        break
                                        
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"\n⚠️ Streaming error: {e}")
            
        print()
        return full_content

class WhiteboardPipeline:
    """Manages the whiteboard-based planning pipeline execution"""
    
    def __init__(self, reasoning_model: str, coding_model: str, api_client: APIClient, whiteboard_manager: WhiteboardManager):
        self.reasoning_model = reasoning_model
        self.coding_model = coding_model
        self.api_client = api_client
        self.whiteboard_manager = whiteboard_manager
    
    def execute_stage(self, stage_name: str, paper_content: str, 
                     prompt_func, schema: Optional[Dict] = None, **extra_args) -> Dict[str, Any]:
        """Execute a single pipeline stage with whiteboard context"""
        
        model = self.reasoning_model if stage_name != "config" else self.coding_model
        
        print(f"[{stage_name.upper()}] Using {model}")
        
        # Get current whiteboard state as YAML context
        whiteboard_yaml = self.whiteboard_manager.get_whiteboard_yaml()
        
        # Generate messages with whiteboard context
        if stage_name in ['analysis']:
            # Analysis needs file-specific args
            messages = prompt_func(paper_content, whiteboard_yaml, **extra_args)
        else:
            # Other stages just need paper and whiteboard
            messages = prompt_func(paper_content, whiteboard_yaml)
        
        try:
            completion = self.api_client.chat_completion(
                model=model,
                messages=messages,
                response_format=schema,
                stream=True
            )
            
            # Parse response and extract updates
            response_content = completion['choices'][0]['message']['content']
            structured_data = parse_structured_response(response_content)
            
            # Apply whiteboard updates if present
            updates = structured_data.get('updates', [])
            if updates and isinstance(updates, list):
                print(f"📝 Applying {len(updates)} whiteboard updates...")
                self.whiteboard_manager.apply_updates(updates)
                
                # Show applied updates
                for update in updates[:3]:  # Show first 3
                    print(f"   • {update}")
                if len(updates) > 3:
                    print(f"   • ... and {len(updates) - 3} more")
            elif updates:
                print(f"⚠️ Updates field is not a list: {type(updates)}")
            
            # Add stage completion marker
            self.whiteboard_manager.add_stage_completion(stage_name, structured_data)
            
            return {
                'choices': [{
                    'message': {
                        'role': 'assistant',
                        'content': response_content
                    }
                }],
                'model_used': model,
                'stage': stage_name,
                'whiteboard_updates_applied': len(updates) if isinstance(updates, list) else 0
            }
            
        except Exception as e:
            print(f"Error in {stage_name} stage: {e}")
            raise

class ArtifactManager:
    """Handles saving and loading of pipeline artifacts with whiteboard integration"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_response(self, stage_name: str, response: Dict[str, Any]) -> None:
        """Save individual stage response"""
        filename = f"{stage_name}_response.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(response, f, indent=2)
        
        print(f"✅ Saved {stage_name} response to {filepath}")
    
    def save_structured_data(self, stage_name: str, structured_data: Dict[str, Any]) -> None:
        """Save parsed structured data"""
        filename = f"{stage_name}_structured.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(structured_data, f, indent=2)
        
        print(f"✅ Saved {stage_name} structured data to {filepath}")
    def save_trajectories(self, trajectories: List[Dict]) -> None:
        """Save complete conversation trajectories"""
        filepath = os.path.join(self.output_dir, "planning_trajectories.json")
        
        with open(filepath, 'w') as f:
            json.dump(trajectories, f, indent=2)
        
        print(f"✅ Saved trajectories to {filepath}")
        
    def save_config_yaml(self, config_content: str) -> None:
            """Save the configuration YAML file"""
            filepath = os.path.join(self.output_dir, "planning_config.yaml")
            
            with open(filepath, 'w') as f:
                f.write(config_content)
            
            print(f"✅ Saved config YAML to {filepath}")
    
    def save_model_config(self, reasoning_model: str, coding_model: str) -> None:
        """Save the model configuration used in the pipeline"""
        model_config = {
            'reasoning_model': reasoning_model,
            'coding_model': coding_model,
            'timestamp': time.time()
        }
        
        filepath = os.path.join(self.output_dir, "model_config.json")
        
        with open(filepath, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        print(f"✅ Saved model config to {filepath}")
    
    def save_analysis_response(self, filename: str, response: Dict[str, Any]) -> None:
        """Save individual analysis response for a specific file"""
        safe_filename = filename.replace("/", "_").replace("\\", "_")
        filepath = os.path.join(self.output_dir, f"{safe_filename}_simple_analysis_response.json")
        
        with open(filepath, 'w') as f:
            json.dump([response], f, indent=2)  # Wrap in array for compatibility
        
        print(f"✅ Saved analysis response for {filename}")
    
    def save_analysis_structured(self, filename: str, structured_data: Dict[str, Any]) -> None:
        """Save parsed analysis structured data"""
        safe_filename = filename.replace("/", "_").replace("\\", "_")
        filepath = os.path.join(self.output_dir, f"{safe_filename}_simple_analysis_structured.json")
        
        with open(filepath, 'w') as f:
            json.dump(structured_data, f, indent=2)
        
        print(f"✅ Saved analysis structured data for {filename}")
    
    def get_task_list_from_whiteboard(self, whiteboard_manager: WhiteboardManager) -> List[str]:
        """Extract task list from whiteboard state"""
        whiteboard = whiteboard_manager.load_whiteboard()
        return whiteboard.get('knowledge', {}).get('task_list', [])
    
    def get_logic_analysis_from_whiteboard(self, whiteboard_manager: WhiteboardManager) -> Dict[str, str]:
        """Extract logic analysis mapping from whiteboard state"""
        whiteboard = whiteboard_manager.load_whiteboard()
        logic_analysis = whiteboard.get('knowledge', {}).get('logic_analysis', [])
        
        logic_dict = {}
        for desc in logic_analysis:
            if isinstance(desc, list) and len(desc) >= 2:
                logic_dict[desc[0]] = desc[1]
            elif isinstance(desc, dict) and 'filename' in desc and 'description' in desc:
                logic_dict[desc['filename']] = desc['description']
        
        return logic_dict
        
    def check_analysis_completion(self, file_list: List[str]) -> bool:
        """Check if analysis phase is complete for all specified files"""
        if not file_list:
            return False
            
        for filename in file_list:
            if filename == "config.yaml":
                continue
                
            safe_filename = filename.replace("/", "_").replace("\\", "_")
            analysis_response_file = os.path.join(self.output_dir, f"{safe_filename}_simple_analysis_response.json")
            analysis_structured_file = os.path.join(self.output_dir, f"{safe_filename}_simple_analysis_structured.json")
            
            if not os.path.exists(analysis_response_file) or not os.path.exists(analysis_structured_file):
                return False
        
        return True

    def get_utility_descriptions_from_analysis(self, file_list: List[str]) -> Dict[str, str]:
        """Extract utility descriptions from analysis files for specified files"""
        utility_descriptions = {}
        
        for filename in file_list:
            if filename == "config.yaml":
                continue
                
            safe_filename = filename.replace("/", "_").replace("\\", "_")
            analysis_file = os.path.join(self.output_dir, f"{safe_filename}_simple_analysis_structured.json")
            
            if os.path.exists(analysis_file):
                try:
                    with open(analysis_file, 'r') as f:
                        analysis_data = json.load(f)
                    
                    # Extract core functionality as utility description
                    core_functionality = analysis_data.get('core_functionality', '')
                    utility_descriptions[filename] = core_functionality
                    
                except Exception as e:
                    print(f"Warning: Could not load analysis for {filename}: {e}")
                    utility_descriptions[filename] = f"Core implementation for {filename}"
            else:
                utility_descriptions[filename] = f"Core implementation for {filename}"
        
        return utility_descriptions

    def save_file_organization_response(self, response: Dict[str, Any]) -> None:
        """Save file organization response"""
        filepath = os.path.join(self.output_dir, "file_organization_response.json")
        
        with open(filepath, 'w') as f:
            json.dump(response, f, indent=2)
        
        print(f"✅ Saved file organization response to {filepath}")

    def save_file_organization_structured(self, structured_data: Dict[str, Any]) -> None:
        """Save file organization structured data"""
        filepath = os.path.join(self.output_dir, "file_organization_structured.json")
        
        with open(filepath, 'w') as f:
            json.dump(structured_data, f, indent=2)
        
        print(f"✅ Saved file organization structured data to {filepath}")

    def get_development_order(self) -> List[str]:
        """Load development order from file organization phase"""
        filepath = os.path.join(self.output_dir, "file_organization_structured.json")
        
        try:
            with open(filepath, 'r') as f:
                file_org_data = json.load(f)
            return file_org_data.get('development_order', [])
        except FileNotFoundError:
            print(f"Warning: No file organization found at {filepath}")
            return []

def load_context_and_analysis_from_whiteboard(output_dir: str, whiteboard_manager: WhiteboardManager) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, str]]:
    """Load context and analysis data from whiteboard state"""
    
    # Get task list from whiteboard
    whiteboard = whiteboard_manager.load_whiteboard()
    knowledge = whiteboard.get('knowledge', {})
    
    # Handle case where knowledge might be a JSON string
    if isinstance(knowledge, str):
        try:
            knowledge = json.loads(knowledge)
        except json.JSONDecodeError:
            knowledge = {}
    
    task_list_raw = knowledge.get('task_list', [])
    
    # Handle case where task_list might be stored as JSON string
    if isinstance(task_list_raw, str):
        try:
            task_list = json.loads(task_list_raw)
        except json.JSONDecodeError:
            task_list = []
    else:
        task_list = task_list_raw if isinstance(task_list_raw, list) else []
    
    # Get development order from whiteboard
    dev_order_raw = knowledge.get('development_order', [])
    if isinstance(dev_order_raw, str):
        try:
            development_order = json.loads(dev_order_raw)
        except json.JSONDecodeError:
            development_order = []
    else:
        development_order = dev_order_raw if isinstance(dev_order_raw, list) else []
    
    # Use development order if available, otherwise fall back to task list
    files_to_process = development_order if development_order else task_list
    
    print(f"📋 Task list contains {len(task_list)} items")
    print(f"📋 Development order contains {len(development_order)} items")
    print(f"📋 Processing {len(files_to_process)} files for coding")
    
    # Load analysis for each file to process
    detailed_logic_analysis_dict = {}
    for todo_file_name in files_to_process:
        if todo_file_name == "config.yaml":
            continue
            
        safe_filename = todo_file_name.replace("/", "_").replace("\\", "_")
        analysis_file = f"{output_dir}/{safe_filename}_simple_analysis_response.json"
        
        if os.path.exists(analysis_file):
            with open(analysis_file) as f:
                analysis_response = json.load(f)
            detailed_logic_analysis_dict[todo_file_name] = analysis_response[0]['choices'][0]['message']['content']
            print(f"   ✅ Loaded analysis for {todo_file_name}")
        else:
            print(f"   ⚠️  No analysis file found for {todo_file_name}, creating placeholder")
            detailed_logic_analysis_dict[todo_file_name] = f"No detailed analysis found for {todo_file_name}. This file should implement core functionality as specified in the whiteboard knowledge."
    
    # Load utility descriptions from analysis structured data
    artifact_manager = ArtifactManager(output_dir)
    utility_descriptions = artifact_manager.get_utility_descriptions_from_analysis(files_to_process)
    
    # Context is now just whiteboard YAML and the files being processed
    context = {
        'task_list': files_to_process,  # Use the files we're actually processing
        'whiteboard_yaml': whiteboard_manager.get_whiteboard_yaml()
    }
    
    print(f"📋 Loaded context from whiteboard with {len(files_to_process)} files to process")
    
    return context, detailed_logic_analysis_dict, utility_descriptions
    
# Coding Pipeline Classes (Enhanced with Whiteboard)
class CodingPipeline:
    def __init__(self, api_client: APIClient, coding_model: str, output_dir: str, 
                 output_repo_dir: str, max_parallel: int = 4, max_context_tokens: int = 128000,
                 whiteboard_manager: WhiteboardManager = None):
        self.api_client = api_client
        self.coding_model = coding_model
        self.output_dir = output_dir
        self.output_repo_dir = output_repo_dir
        self.max_parallel = max_parallel
        self.max_context_tokens = max_context_tokens
        self.whiteboard_manager = whiteboard_manager
        self.done_files = ['config.yaml']
        self.done_file_dict = {}
        
        # Smart context tracking
        self.interface_context = {
            'exports': {},  # file -> {classes: [], functions: [], constants: []}
            'imports': {},  # file -> {dependencies: [...]}
            'handoff_notes': []  # Accumulated integration guidance
        }
        
        # Create output directories
        os.makedirs(output_repo_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/coding_artifacts", exist_ok=True)
        os.makedirs(f"{output_dir}/diffs", exist_ok=True)
        os.makedirs(f"{output_dir}/structured_code_responses", exist_ok=True)
        os.makedirs(f"{output_dir}/interface_context", exist_ok=True)
        
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token average)"""
        return len(text) // 4
        
    def build_interface_context_summary(self) -> str:
        """Build compact interface summary instead of full code context"""
        if not self.interface_context['exports']:
            return ""
        
        summary = "\n## Previously Generated Files - Interface Context\n"
        
        # Show what each file exports
        for filename, exports in self.interface_context['exports'].items():
            summary += f"\n### {filename} (Provides)\n"
            
            if exports.get('classes'):
                summary += "**Classes & Methods:**\n"
                for class_info in exports['classes']:
                    summary += f"- {class_info}\n"
            
            if exports.get('functions'):
                summary += "**Functions:**\n"
                for func_info in exports['functions']:
                    summary += f"- {func_info}\n"
            
            if exports.get('constants'):
                summary += "**Constants:**\n"
                for const_info in exports['constants']:
                    summary += f"- {const_info}\n"
        
        # Show dependency patterns
        summary += "\n### Integration Patterns\n"
        for filename, imports in self.interface_context['imports'].items():
            if imports.get('dependencies'):
                summary += f"\n**{filename} depends on:**\n"
                for dep in imports['dependencies']:
                    summary += f"- {dep['from_file']}: {', '.join(dep['imports'])} ({dep['usage_context']})\n"
        
        # Add accumulated handoff notes
        if self.interface_context['handoff_notes']:
            summary += "\n### Integration Guidance\n"
            for i, note in enumerate(self.interface_context['handoff_notes'], 1):
                summary += f"{i}. {note}\n"
        
        summary += "\n-----\n"
        return summary
                
    def extract_code_from_structured_response(self, structured_data: Dict[str, Any], filename: str) -> str:
        """Extract code from structured response - BACK TO ORIGINAL SIMPLE FORMAT"""
        files = structured_data.get('files', [])
        
        for file_data in files:
            # Handle case where file_data might be a list instead of dict
            if isinstance(file_data, list):
                # Skip malformed entries
                print(f"   ⚠️  Skipping malformed file_data (list): {file_data}")
                continue
            elif isinstance(file_data, dict):
                if file_data.get('file_name') == filename:
                    diff_file = file_data.get('diff_file', '')
                    # Handle case where LLM returns array instead of string
                    if isinstance(diff_file, list):
                        return '\n'.join(str(line) for line in diff_file)
                    return str(diff_file)
            else:
                print(f"   ⚠️  Unexpected file_data type: {type(file_data)}")
                continue
        
        # Fallback - return first file if exact match not found
        if files:
            first_file = files[0]
            if isinstance(first_file, dict):
                diff_file = first_file.get('diff_file', '')
                if isinstance(diff_file, list):
                    return '\n'.join(str(line) for line in diff_file)
                return str(diff_file)
            elif isinstance(first_file, list):
                # If it's a list, try to join it as code
                return '\n'.join(str(item) for item in first_file)
        
        print(f"   ⚠️  No code found for {filename}")
        return ""

    def update_interface_context(self, filename: str, structured_data: Dict[str, Any]):
        """Update interface context with new file's interface information"""
        # Update whiteboard with interface context
        if self.whiteboard_manager:
            updates = [
                f"coding.interfaces.{filename.replace('.', '_').replace('/', '_')}.exported=true",
                f"coding.progress.completed_files.{len(self.done_files)}={filename}"
            ]
            # Ensure we only pass strings to apply_updates
            string_updates = [str(update) for update in updates if isinstance(update, (str, int, float, bool))]
            self.whiteboard_manager.apply_updates(string_updates)

    def create_diff_file(self, filename: str, code: str) -> str:
        """Create a diff file for the generated code"""
        # Code is already in diff format from structured response
        safe_filename = filename.replace("/", "_").replace("\\", "_")
        diff_path = f"{self.output_dir}/diffs/{safe_filename}.diff"
        
        # If code doesn't start with diff header, add it
        if not code.startswith('---'):
            diff_content = f"""--- /dev/null
+++ {filename}
@@ -0,0 +1,{len(code.split(chr(10)))} @@
+{code.replace(chr(10), chr(10) + '+')}
"""
        else:
            diff_content = code
        
        with open(diff_path, 'w') as f:
            f.write(diff_content)
        
        return diff_path
    
    def clean_diff_to_code(self, diff_content: str) -> str:
        """Convert diff format back to clean Python code"""
        lines = diff_content.split('\n')
        code_lines = []
        
        for line in lines:
            # Skip diff headers and metadata
            if line.startswith('---') or line.startswith('+++') or line.startswith('@@'):
                continue
            # Remove '+' prefix from diff lines
            elif line.startswith('+'):
                code_lines.append(line[1:])  # Remove the '+' prefix
            # Keep regular lines (shouldn't happen in proper diff, but just in case)
            elif not line.startswith('-'):
                code_lines.append(line)
        
        return '\n'.join(code_lines)
    
    def generate_single_file(self, file_info: Tuple[str, str, str], shared_context: Dict[str, str]) -> Dict[str, Any]:
        """Generate code for a single file using smart interface context and whiteboard"""
        todo_file_name, detailed_logic_analysis, utility_description = file_info
        
        try:
            print(f"\n[CODING] {todo_file_name}")
            print(f"   Utility: {utility_description[:100]}{'...' if len(utility_description) > 100 else ''}")
            
            # Build smart interface context (much smaller than full code)
            interface_summary = self.build_interface_context_summary()
            interface_tokens = self.estimate_tokens(interface_summary)
            
            if len(self.done_files) > 1:
                print(f"   📚 Interface Context: {len(self.interface_context['exports'])} files, ~{interface_tokens:,} tokens")
            
            # Generate prompt with smart context management and whiteboard
            from prompts import get_coding_prompt_smart_context, CODE_SCHEMA
            messages = get_coding_prompt_smart_context(
                todo_file_name=todo_file_name,
                detailed_logic_analysis=detailed_logic_analysis,
                utility_description=utility_description,
                paper_content=shared_context['paper_content'],
                config_yaml=shared_context['config_yaml'],
                shared_context=shared_context,
                interface_context=interface_summary,
                max_context_tokens=self.max_context_tokens
            )
            
            # Make API call with enhanced structured response
            completion = self.api_client.chat_completion(
                model=self.coding_model,
                messages=messages,
                response_format=CODE_SCHEMA,
                stream=True
            )
            
            # Parse structured response
            content = completion['choices'][0]['message']['content']
            structured_data = parse_structured_response(content)
            
            # Apply whiteboard updates if present
            updates = structured_data.get('updates', [])
            if updates and self.whiteboard_manager:
                if isinstance(updates, list):
                    print(f"📝 Applying {len(updates)} coding whiteboard updates...")
                    self.whiteboard_manager.apply_updates(updates)
                else:
                    print(f"⚠️ Updates field is not a list: {type(updates)} - converting")
                    if isinstance(updates, str):
                        self.whiteboard_manager.apply_updates([updates])
            
            # Extract information from structured response
            deliberation = structured_data.get('deliberation', '')
            utility = structured_data.get('utility', '')
            diff_code = self.extract_code_from_structured_response(structured_data, todo_file_name)
            
            # Update interface context for next iterations
            self.update_interface_context(todo_file_name, structured_data)
            
            # Convert diff to clean code
            clean_code = self.clean_diff_to_code(diff_code)
            
            # Create diff file and save artifacts
            diff_path = self.create_diff_file(todo_file_name, diff_code)
            
            # Save artifacts
            safe_filename = todo_file_name.replace("/", "_").replace("\\", "_")
            
            with open(f"{self.output_dir}/structured_code_responses/{safe_filename}_structured.json", 'w') as f:
                json.dump(structured_data, f, indent=2)
            
            with open(f"{self.output_dir}/coding_artifacts/{safe_filename}_coding.txt", 'w') as f:
                f.write(content)
            
            with open(f"{self.output_dir}/coding_artifacts/{safe_filename}_deliberation.txt", 'w') as f:
                f.write(f"DELIBERATION:\n{deliberation}\n\nUTILITY:\n{utility}")
            
            # Write clean code file to repository
            if "/" in todo_file_name:
                todo_file_dir = '/'.join(todo_file_name.split("/")[:-1])
                os.makedirs(f"{self.output_repo_dir}/{todo_file_dir}", exist_ok=True)
            
            with open(f"{self.output_repo_dir}/{todo_file_name}", 'w') as f:
                f.write(clean_code)
            
            # Add to done files
            self.done_files.append(todo_file_name)
            self.done_file_dict[todo_file_name] = clean_code
            
            return {
                'filename': todo_file_name,
                'success': True,
                'code': clean_code,
                'diff_code': diff_code,
                'diff_path': diff_path,
                'deliberation': deliberation,
                'utility': utility,
                'structured_data': structured_data,
                'content': content,
                'interface_tokens': interface_tokens,
                'whiteboard_updates': len(updates) if isinstance(updates, list) else 0
            }
            
        except Exception as e:
            print(f"❌ Error generating {todo_file_name}: {e}")
            return {
                'filename': todo_file_name,
                'success': False,
                'error': str(e),
                'diff_path': None
            }

    def process_files_parallel(self, file_tasks: List[Tuple[str, str, str]], 
                              shared_context: Dict[str, str]) -> List[Dict[str, Any]]:
        """Process multiple files in parallel with structured responses and RESUME capability"""
        
        print(f"\n🔄 Checking for existing generated files...")
        
        # Filter out already-completed files and load them into context
        remaining_tasks = []
        completed_results = []
        
        for file_info in file_tasks:
            filename, detailed_logic_analysis, utility_description = file_info
            
            # Check if file already exists in repository
            repo_file_path = f"{self.output_repo_dir}/{filename}"
            safe_filename = filename.replace("/", "_").replace("\\", "_")
            
            # Check for all required artifacts
            structured_response_path = f"{self.output_dir}/structured_code_responses/{safe_filename}_structured.json"
            diff_path = f"{self.output_dir}/diffs/{safe_filename}.diff"
            coding_artifact_path = f"{self.output_dir}/coding_artifacts/{safe_filename}_coding.txt"
            
            if (os.path.exists(repo_file_path) and 
                os.path.exists(structured_response_path) and 
                os.path.exists(diff_path) and 
                os.path.exists(coding_artifact_path)):
                
                print(f"   ✅ Found existing: {filename}")
                
                # Load existing code into context for subsequent files
                try:
                    with open(repo_file_path, 'r') as f:
                        existing_code = f.read()
                    self.done_files.append(filename)
                    self.done_file_dict[filename] = existing_code
                    
                    # Load existing structured data for result compatibility
                    with open(structured_response_path, 'r') as f:
                        structured_data = json.load(f)
                    
                    # Create result entry for completed file
                    completed_results.append({
                        'filename': filename,
                        'success': True,
                        'code': existing_code,
                        'diff_path': diff_path,
                        'utility': structured_data.get('utility', utility_description),
                        'deliberation': structured_data.get('deliberation', 'Previously completed'),
                        'structured_data': structured_data,
                        'resumed': True,
                        'whiteboard_updates': 0
                    })
                    
                except Exception as e:
                    print(f"   ⚠️  Error loading existing {filename}: {e}")
                    print(f"      Will regenerate...")
                    remaining_tasks.append(file_info)
            else:
                remaining_tasks.append(file_info)
        
        if completed_results:
            print(f"📂 Resuming: {len(completed_results)} files already completed")
            print(f"🔄 Remaining: {len(remaining_tasks)} files to generate")
            
            for result in completed_results:
                print(f"   - {result['filename']} (loaded)")
        else:
            print(f"🆕 Starting fresh: {len(remaining_tasks)} files to generate")
        
        # If no remaining tasks, return completed results
        if not remaining_tasks:
            print("\n✅ All files already completed!")
            return completed_results
        
        # Process remaining files in parallel
        new_results = []
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
                # Submit only remaining tasks
                future_to_file = {
                    executor.submit(self.generate_single_file, file_info, shared_context): file_info[0]
                    for file_info in remaining_tasks
                }
                
                # Process completed tasks with progress bar
                with tqdm(total=len(remaining_tasks), desc="Generating remaining code") as pbar:
                    for future in concurrent.futures.as_completed(future_to_file):
                        filename = future_to_file[future]
                        try:
                            result = future.result()
                            new_results.append(result)
                            
                            if result['success']:
                                # Update shared state for subsequent files
                                self.done_files.append(result['filename'])
                                self.done_file_dict[result['filename']] = result['code']
                                print(f"   ✅ Generated: {result['filename']}")
                                print(f"      Utility: {result.get('utility', 'N/A')[:80]}{'...' if len(result.get('utility', '')) > 80 else ''}")
                                if result.get('whiteboard_updates', 0) > 0:
                                    print(f"      Whiteboard updates: {result['whiteboard_updates']}")
                            else:
                                print(f"   ❌ Failed: {result['filename']}")
                                
                        except Exception as e:
                            print(f"   ❌ Exception for {filename}: {e}")
                            new_results.append({
                                'filename': filename,
                                'success': False,
                                'error': str(e),
                                'diff_path': None
                            })
                        
                        pbar.update(1)
                        
        except KeyboardInterrupt:
            print(f"\n⚠️  KeyboardInterrupt received!")
            print(f"📊 Progress before interruption:")
            print(f"   - Completed before resume: {len(completed_results)}")
            print(f"   - Generated in this session: {len(new_results)}")
            print(f"   - Remaining: {len(remaining_tasks) - len(new_results)}")
            print(f"\n💡 You can resume by running the same command again.")
            print(f"   Already completed files will be detected and skipped.")
            
            # Re-raise to maintain normal Ctrl-C behavior
            raise
        
        # Combine completed and new results
        all_results = completed_results + new_results
        
        # Print summary
        resumed_count = len([r for r in all_results if r.get('resumed', False)])
        new_count = len([r for r in all_results if not r.get('resumed', False)])
        
        if resumed_count > 0:
            print(f"\n📊 Final Summary:")
            print(f"   - Resumed existing: {resumed_count}")
            print(f"   - Generated new: {new_count}")
            print(f"   - Total: {len(all_results)}")
        
        return all_results

    def process_files_sequential(self, file_tasks: List[Tuple[str, str]], 
                                shared_context: Dict[str, str]) -> List[Dict[str, Any]]:
        """Process files sequentially for whiteboard consistency"""
        
        results = []
        
        for filename, description in file_tasks:
            print(f"\n[CODING] {filename}")
            print(f"   Purpose: {description}")
            
            try:
                # Convert to format expected by generate_single_file
                file_info = (filename, description, description)  # (filename, analysis, utility)
                
                # Generate the file
                result = self.generate_single_file(file_info, shared_context)
                results.append(result)
                
                if result['success']:
                    print(f"   ✅ Generated {filename} ({len(result['code'])} chars)")
                    if result.get('whiteboard_updates', 0) > 0:
                        print(f"   📝 Applied {result['whiteboard_updates']} whiteboard updates")
                else:
                    print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"   ❌ Exception generating {filename}: {e}")
                results.append({
                    'filename': filename,
                    'success': False,
                    'error': str(e)
                })
        
        return results

# Add PipelineConfig class with whiteboard integration
class PipelineConfig:
    """Centralized pipeline configuration and stage management with whiteboard support"""
    
    # Define the pipeline stages and their dependencies
    STAGES = {
        'planning': {
            'dependencies': [],
            'schema': 'PLANNING_SCHEMA',
            'model': 'reasoning',
            'prompt_func': 'get_planning_prompt'
        },
        'six_hats': {
            'dependencies': ['planning'],
            'schema': 'SIX_HATS_SCHEMA', 
            'model': 'reasoning',
            'prompt_func': 'get_six_hats_prompt'
        },
        'dependency': {
            'dependencies': ['planning', 'six_hats'],
            'schema': 'DEPENDENCY_SCHEMA',
            'model': 'reasoning', 
            'prompt_func': 'get_dependency_prompt'
        },
        'code_structure': {
            'dependencies': ['planning', 'six_hats', 'dependency'],
            'schema': 'CODE_STRUCTURE_SCHEMA',
            'model': 'reasoning',
            'prompt_func': 'get_code_structure_prompt'
        },
        'architecture': {
            'dependencies': ['planning', 'six_hats', 'dependency', 'code_structure'],
            'schema': 'ARCHITECTURE_SCHEMA',
            'model': 'reasoning',
            'prompt_func': 'get_architecture_prompt'
        },
        'task_list': {
            'dependencies': ['planning', 'six_hats', 'dependency', 'code_structure', 'architecture'],
            'schema': 'TASK_LIST_SCHEMA',
            'model': 'reasoning',
            'prompt_func': 'get_task_list_prompt'
        },
        'config': {
            'dependencies': ['planning', 'six_hats', 'dependency', 'code_structure', 'architecture', 'task_list'],
            'schema': 'CONFIG_SCHEMA',
            'model': 'coding',
            'prompt_func': 'get_config_prompt'
        },
        'analysis': {
            'dependencies': ['planning', 'six_hats', 'dependency', 'architecture', 'code_structure', 'task_list'],
            'schema': 'ANALYSIS_SCHEMA',
            'model': 'reasoning',
            'prompt_func': 'get_analysis_prompt'
        },
        'file_organization': {
            'dependencies': ['task_list'],
            'schema': 'FILE_ORGANIZATION_SCHEMA',
            'model': 'reasoning',
            'prompt_func': 'get_file_organization_prompt'
        }
    }
    
    @classmethod
    def get_stage_dependencies(cls, stage_name: str) -> List[str]:
        """Get list of stages this stage depends on"""
        return cls.STAGES.get(stage_name, {}).get('dependencies', [])
    
    @classmethod
    def get_stage_schema(cls, stage_name: str) -> str:
        """Get schema name for stage"""
        return cls.STAGES.get(stage_name, {}).get('schema')
    
    @classmethod
    def get_stage_model(cls, stage_name: str) -> str:
        """Get model type for stage (reasoning/coding)"""
        return cls.STAGES.get(stage_name, {}).get('model', 'reasoning')
    
    @classmethod
    def get_stage_prompt_func(cls, stage_name: str) -> str:
        """Get prompt function name for stage"""
        return cls.STAGES.get(stage_name, {}).get('prompt_func')
    
    #TODO: Archived replaced by whiteboard, not used
    @classmethod
    def build_context_for_stage(cls, stage_name: str, structured_responses: Dict[str, Any]) -> Dict[str, str]:
        """Build context dictionary for a stage based on its dependencies"""
        context = {}
        dependencies = cls.get_stage_dependencies(stage_name)
        
        for dep_stage in dependencies:
            if dep_stage in structured_responses:
                context[dep_stage] = format_dict_as_yaml_style(
                    structured_responses[dep_stage], 
                    dep_stage
                )
        
        return context

    #TODO: Archived replaced by whiteboard, not used
    @classmethod
    def get_prompt_args(cls, stage_name: str, paper_content: str, structured_responses: Dict[str, Any], 
                       **extra_args) -> List:
        """Build prompt arguments for a stage dynamically"""
        dependencies = cls.get_stage_dependencies(stage_name)
        context = cls.build_context_for_stage(stage_name, structured_responses)
        
        # Base arguments that every prompt gets
        args = [paper_content]
        
        # Add dependency contexts in order
        for dep_stage in dependencies:
            if dep_stage in context:
                args.append(context[dep_stage])
        
        # Add any extra arguments (like file_name for analysis)
        for key, value in extra_args.items():
            args.append(value)
        
        return args
        
def load_generated_code_files(repo_dir: str, prefix: str = '') -> str:
    """Load generated files with optional prefix filtering"""
    if not os.path.exists(repo_dir):
        return "No generated files found."
    
    files = []
    for filename in os.listdir(repo_dir):
        if filename.startswith(prefix) and (filename.endswith('.py') or filename.endswith('.yaml')):
            try:
                with open(f"{repo_dir}/{filename}", 'r') as f:
                    files.append(f"## {filename}\n```python\n{f.read()}\n```")
            except Exception as e:
                print(f"Warning: Error reading {filename}: {e}")
                
    return '\n\n'.join(files) if files else f"No {prefix}* files found"

def save_category_implementation(category: str, implementation: str, output_dir: str, output_repo_dir: str) -> None:
    """Save category implementation directly to repository as corrected_*.py files."""
    
    # Still save JSON for debugging
    os.makedirs(f"{output_dir}/iterative_refinements", exist_ok=True)
    category_data = {
        "category": category,
        "implementation": implementation,
        "timestamp": time.time()
    }
    json_filepath = f"{output_dir}/iterative_refinements/{category}.json"
    with open(json_filepath, 'w') as f:
        json.dump(category_data, f, indent=2)
    
    # WRITE ACTUAL PYTHON FILES TO REPO WITH CORRECTED_ PREFIX
    if category == 'corrected_constants':
        repo_file = f"{output_repo_dir}/corrected_constants.py"
    elif category == 'corrected_imports':
        repo_file = f"{output_repo_dir}/corrected_imports.py"  
    elif category == 'corrected_functions':
        repo_file = f"{output_repo_dir}/corrected_functions.py"
    elif category == 'corrected_classes':
        repo_file = f"{output_repo_dir}/corrected_classes.py"
    elif category == 'corrected_main':
        repo_file = f"{output_repo_dir}/corrected_main.py"
    elif category == 'corrected_config':
        repo_file = f"{output_repo_dir}/corrected_config.yaml"
    else:
        # Generic fallback - keep the full category name
        filename = category + '.py'
        repo_file = f"{output_repo_dir}/{filename}"
    
    # Write the actual implementation to the repo
    try:
        with open(repo_file, 'w') as f:
            f.write(implementation)
        print(f"✅ Saved {category} implementation to {repo_file}")
        print(f"📝 Also saved debug JSON to {json_filepath}")
    except Exception as e:
        print(f"❌ Failed to write {repo_file}: {e}")
        print(f"✅ Debug JSON still saved to {json_filepath}")
        
def compile_refinement_summary(output_dir: str) -> str:
    """Generate a simple summary of what was refined (optional for debugging)."""
    refinements_dir = f"{output_dir}/iterative_refinements"
    
    if not os.path.exists(refinements_dir):
        return "No refinements found."
    
    summary = "# Refinement Summary\n\n"
    categories = ['corrected_constants', 'corrected_imports', 'corrected_functions', 
                 'corrected_classes', 'corrected_main', 'corrected_config']
    
    for category in categories:
        filepath = f"{refinements_dir}/{category}.json"
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                items_completed = data.get('items_completed', [])
                summary += f"## {category}\n"
                summary += f"- Items: {len(items_completed)}\n"
                summary += f"- Files: {', '.join(items_completed)}\n\n"
            except:
                continue
    
    # Optional: save summary (but not needed)
    summary_path = f"{output_dir}/refinement_summary.md"
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    return summary

# Add to functions.py - File Translation System

def create_file_mapping(classification_data: Dict[str, str], original_task_list: List[str]) -> Dict[str, str]:
    """Create mapping from original PDR file names to consolidated 5-file structure"""
    
    mapping = {}
    
    for original_file in original_task_list:
        if original_file == "config.yaml":
            mapping[original_file] = "config.yaml"  # Keep as-is
            continue
            
        # Default mapping rules based on file name patterns
        if any(keyword in original_file.lower() for keyword in 
               ['parse', 'parser', 'util', 'helper', 'process', 'validate']):
            mapping[original_file] = "functions.py"
        elif any(keyword in original_file.lower() for keyword in 
                 ['class', 'model', 'transformer', 'encoder', 'agent', 'module']):
            mapping[original_file] = "classes.py"
        elif original_file == "main.py":
            mapping[original_file] = "main.py"
        elif any(keyword in original_file.lower() for keyword in 
                 ['const', 'config', 'param', 'setting']):
            mapping[original_file] = "constants.py"
        elif any(keyword in original_file.lower() for keyword in 
                 ['import', 'deps', 'depend']):
            mapping[original_file] = "imports.py"
        else:
            # When in doubt: functions.py
            mapping[original_file] = "functions.py"
    
    return mapping

def translate_imports_in_main(main_code: str, file_mapping: Dict[str, str]) -> str:
    """Translate import statements in main.py from original names to consolidated files"""
    import re
    
    lines = main_code.split('\n')
    translated_lines = []
    
    for line in lines:
        # Match: from original_module import ...
        import_match = re.match(r'from (\w+) import (.+)', line)
        if import_match:
            original_module = import_match.group(1)
            imports = import_match.group(2)
            
            # Find what file this module was mapped to
            original_file = f"{original_module}.py"
            if original_file in file_mapping:
                target_file = file_mapping[original_file]
                target_module = target_file.replace('.py', '')
                
                # Translate the import
                new_line = f"from {target_module} import {imports}"
                translated_lines.append(new_line)
            else:
                # Keep original import (might be external library)
                translated_lines.append(line)
        else:
            translated_lines.append(line)
    
    return '\n'.join(translated_lines)

def update_coding_phase_with_translation():
    """Show how to integrate translation into the coding phase"""
    
    # In run_coding_phase(), after classification:
    """
    # 1. Create file mapping
    original_task_list = whiteboard_manager.load_whiteboard().get('knowledge', {}).get('task_list', [])
    file_mapping = create_file_mapping(classification_data, original_task_list)
    
    # 2. Store mapping in whiteboard for correction phase
    mapping_updates = [f"translation.file_mapping.{k.replace('.', '_')}={v}" 
                      for k, v in file_mapping.items()]
    whiteboard_manager.apply_updates(mapping_updates)
    
    # 3. After generating main.py, translate its imports
    if result['filename'] == 'main.py' and result['success']:
        translated_code = translate_imports_in_main(result['code'], file_mapping)
        
        # Write the translated version
        with open(f"{output_repo_dir}/main.py", 'w') as f:
            f.write(translated_code)
        
        result['code'] = translated_code
    """

def fix_gap_analysis_with_translation():
    """Update gap analysis to use translated file names"""
    
    # In get_gap_analysis_prompt(), add file mapping context:
    """
    ## File Translation Mapping
    The following files were consolidated:
    {file_mapping_context}
    
    When referencing corrections, use the TRANSLATED file names:
    - pql_parser.py → functions.py  
    - table_encoder.py → classes.py
    - etc.
    """

def fix_category_implementation_with_translation():
    """Update category implementation to reference correct files"""
    
    # In get_category_implementation_prompt():
    """
    IMPORTANT: When implementing {category}, remember the file translations:
    - All parser functions go in functions.py
    - All encoder/transformer classes go in classes.py  
    - All constants go in constants.py
    - All imports go in imports.py
    - Main orchestration goes in main.py
    
    Do NOT reference original file names like 'pql_parser.py' - 
    everything is consolidated into the 5-file structure.
    """

# Example usage in the pipeline:
def demonstrate_fix():
    """Example of how this fixes the import issue"""
    
    # Original PDR files:
    original_files = ["pql_parser.py", "table_encoder.py", "main.py"]
    
    # Classification mapping:
    classification = {
        "pql_parser.py": "functions.py",
        "table_encoder.py": "classes.py", 
        "main.py": "main.py"
    }
    
    # Original main.py content:
    original_main = """
from pql_parser import parse_pql_query
from table_encoder import TableInvariantEncoder
    """
    
    # Translated main.py content:
    translated_main = """
from functions import parse_pql_query  
from classes import TableInvariantEncoder
    """
    
    print("Translation fixes the import mismatch!")
    