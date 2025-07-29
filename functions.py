# functions.py
"""
Core functions and classes for the planning pipeline
"""

from imports import *
class StreamMonitor:
    """External monitor that watches streaming content and can kill the stream"""
    
    def __init__(self, repetition_threshold: int = 100):
        self.content_buffer = ""
        self.should_terminate = False
        self.repetition_threshold = repetition_threshold
    
    def add_content(self, new_content: str) -> bool:
        """Add new content and check for repetition. Returns True if should continue."""
        self.content_buffer += new_content
        
        # Keep buffer manageable
        if len(self.content_buffer) > 1000:
            self.content_buffer = self.content_buffer[-500:]
        
        # Check for repetition patterns every chunk
        if len(self.content_buffer) > self.repetition_threshold:
            if self._detect_repetition():
                print(f"\n⚠️ StreamMonitor: Repetition detected, terminating stream...")
                self.should_terminate = True
                return False
        
        return True
    
    def _detect_repetition(self) -> bool:
        """Aggressive repetition detection"""
        
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
    """Parse structured JSON response from API"""
    try:
        return json.loads(response_content)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        print(f"Response content: {response_content}")
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

# 3. Add resume flag to setup_argument_parser:
def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(description="Code generation pipeline")
    
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
    parser.add_argument('--resume_from_analysis', action='store_true',
                       help='Skip planning and analysis phases if data exists')
    
    return parser

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
                 initial_seed: int = 42, default_timeout: int = 180):  # NEW: Accept default timeout
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {"Content-Type": "application/json"}
        self.current_seed = initial_seed
        self.default_timeout = default_timeout  # NEW: Store default timeout
        
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        
        # Generation settings rotation for all attempts
        self.generation_settings = [
            {"name": "balanced", "temperature": 0.33, "top_p": 0.92, "repeat_penalty": 1.3, "presence_penalty": 1.1, "top_k": 55},
            {"name": "precise", "temperature": 0.13, "top_p": 0.78, "repeat_penalty": 1.3, "presence_penalty": 1.1, "top_k": 34},
            {"name": "creative", "temperature": 0.45, "top_p": 0.95, "repeat_penalty": 1.3, "presence_penalty": 1.1, "top_k": 66}
        ]
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
                   stream: bool = False) -> Dict[str, Any]:  # ADD: stream parameter
        """Make a chat completion request with timeout and retry logic"""
        
        # NEW: Use provided timeout or fall back to default
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
                    "stream": stream,  # ADD: streaming parameter
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
                    stream=stream  # ADD: stream to requests
                )
                response.raise_for_status()
                
                # ADD: Handle streaming vs non-streaming responses
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
                
                # Check repetition for ALL responses
                if self._has_repetition(content):
                    print(f"⚠️  Detected repetition in response, retrying with different settings...")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        print(f"⚠️  Max retries reached, using response with repetition")
                
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

    # ADD: New method to handle streaming
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

    
    def _has_repetition(self, text: str, threshold: int = 3) -> bool:
        """Detect if text has problematic repetition"""
        if not text or len(text) < 50:
            return False
            
        # Check for repeated "further" pattern
        words = text.lower().split()
        if len(words) < 10:
            return False
            
        # Count consecutive identical words
        max_consecutive = 1
        current_consecutive = 1
        for i in range(1, len(words)):
            if words[i] == words[i-1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
                
        if max_consecutive >= 3:  # 3+ consecutive identical words
            return True
            
        # Keep existing sentence-based check
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 3:
            return False
            
        sentence_counts = {}
        for sentence in sentences:
            if len(sentence) > 10:
                sentence_counts[sentence] = sentence_counts.get(sentence, 0) + 1
                
        max_count = max(sentence_counts.values()) if sentence_counts else 1
        return max_count >= threshold


class PlanningPipeline:
    """Manages the planning pipeline execution"""
    
    def __init__(self, reasoning_model: str, coding_model: str, api_client: APIClient):
        self.reasoning_model = reasoning_model
        self.coding_model = coding_model
        self.api_client = api_client
        
        # Stage configuration: (model, use_structured_output, schema)
        self.stages = [
            ("planning", reasoning_model, True),
            ("six_hats", reasoning_model, True),
            ("dependency", reasoning_model, True),
            ("architecture", reasoning_model, True),
            ("context_code_structure", reasoning_model, True),
            ("task_list", reasoning_model, True),
            ("config", coding_model, True)
        ]
    
    def execute_stage(self, stage_name: str, messages: List[Dict], 
                     use_structured: bool, schema: Optional[Dict] = None,
                     **generation_params) -> Dict[str, Any]:
        """Execute a single pipeline stage with simplified interface"""
        
        model = self.reasoning_model if stage_name != "config" else self.coding_model
        
        print(f"[{stage_name.upper()}] Using {model}")
        
        response_format = schema if use_structured else None
        
        try:
            completion = self.api_client.chat_completion(
                model=model,
                messages=messages,
                response_format=response_format,
                stream=True,  # ADD: stream parameter
                **generation_params
            )
            
            return {
                'choices': [{
                    'message': {
                        'role': 'assistant',
                        'content': completion['choices'][0]['message']['content']
                    }
                }],
                'model_used': model,
                'stage': stage_name
            }
            
        except Exception as e:
            print(f"Error in {stage_name} stage: {e}")
            raise


class ArtifactManager:
    """Handles saving and loading of pipeline artifacts"""
    
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
    
    def load_structured_responses(self) -> Dict[str, Any]:
        """Load all structured responses from previous stages"""
        filepath = os.path.join(self.output_dir, "all_structured_responses.json")
        
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: No structured responses found at {filepath}")
            return {}
    
    def get_task_list_from_responses(self, structured_responses: Dict[str, Any]) -> List[str]:
        """Extract task list from structured responses"""
        task_list_data = structured_responses.get('task_list', {})
        return task_list_data.get('task_list', [])
    
    def get_logic_analysis_from_responses(self, structured_responses: Dict[str, Any]) -> Dict[str, str]:
        """Extract logic analysis mapping from structured responses"""
        task_list_data = structured_responses.get('task_list', {})
        logic_analysis = task_list_data.get('logic_analysis', [])
        
        logic_dict = {}
        for desc in logic_analysis:
            if len(desc) >= 2:
                logic_dict[desc[0]] = desc[1]
            else:
                logic_dict[desc[0]] = ""
        
        return logic_dict
    
    def get_task_metadata_from_responses(self, structured_responses: Dict[str, Any]) -> Dict[str, Dict]:
        """Extract task metadata mapping from structured responses"""
        task_list_data = structured_responses.get('task_list', {})
        task_metadata = task_list_data.get('task_metadata', [])
        
        metadata_dict = {}
        for meta in task_metadata:
            if 'filename' in meta:
                metadata_dict[meta['filename']] = {
                    'critical_path': meta.get('critical_path', False),
                    'priority': meta.get('priority', 'medium'),
                    'utility': meta.get('utility', 'medium'), 
                    'effort': meta.get('effort', 'medium')
                }
        
        return metadata_dict
    
    def save_model_config(self, reasoning_model: str, coding_model: str) -> None:
        """Save model configuration"""
        model_config = {
            "reasoning_model": reasoning_model,
            "coding_model": coding_model,
            "stage_assignments": {
                "planning": reasoning_model,
                "six_hats": reasoning_model,
                "dependency": reasoning_model,
                "architecture": reasoning_model, 
                "context_code_structure": reasoning_model,
                "task_list": reasoning_model,
                "analysis": reasoning_model,
                "config": coding_model
            }
        }
        
        filepath = os.path.join(self.output_dir, "model_config.json")
        
        with open(filepath, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        print(f"✅ Saved model config to {filepath}")

    # 1. Add this function to ArtifactManager class:
    def get_utility_descriptions_from_analysis(self, task_list: List[str]) -> Dict[str, str]:
        """Extract utility descriptions from analysis files"""
        utility_descriptions = {}
        
        for filename in task_list:
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

    def check_analysis_completion(self, task_list: List[str]) -> bool:
        """Check if analysis phase is complete for all tasks"""
        if not task_list:
            return False
            
        for filename in task_list:
            if filename == "config.yaml":
                continue
                
            safe_filename = filename.replace("/", "_").replace("\\", "_")
            analysis_response_file = os.path.join(self.output_dir, f"{safe_filename}_simple_analysis_response.json")
            analysis_structured_file = os.path.join(self.output_dir, f"{safe_filename}_simple_analysis_structured.json")
            
            if not os.path.exists(analysis_response_file) or not os.path.exists(analysis_structured_file):
                return False
        
        return True

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


class CodingPipeline:
    """Manages parallel code generation with structured responses and diff output"""
    
    def __init__(self, api_client: APIClient, coding_model: str, output_dir: str, 
                 output_repo_dir: str, max_parallel: int = 4):
        self.api_client = api_client
        self.coding_model = coding_model
        self.output_dir = output_dir
        self.output_repo_dir = output_repo_dir
        self.max_parallel = max_parallel
        self.done_files = ['config.yaml']
        self.done_file_dict = {}
        
        # Create output directories
        os.makedirs(output_repo_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/coding_artifacts", exist_ok=True)
        os.makedirs(f"{output_dir}/diffs", exist_ok=True)
        os.makedirs(f"{output_dir}/structured_code_responses", exist_ok=True)
    
    def extract_code_from_structured_response(self, structured_data: Dict[str, Any], filename: str) -> str:
        """Extract code from structured response"""
        files = structured_data.get('files', [])
        
        for file_data in files:
            if file_data.get('file_name') == filename:
                return file_data.get('diff_file', '')
        
        # Fallback - return first file if exact match not found
        if files:
            return files[0].get('diff_file', '')
        
        return ""
    
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
            """Generate code for a single file using structured response"""
            todo_file_name, detailed_logic_analysis, utility_description = file_info
            
            try:
                print(f"\n[CODING] {todo_file_name}")
                print(f"   Utility: {utility_description[:100]}{'...' if len(utility_description) > 100 else ''}")
                
                # Build context of previously implemented files
                code_files = ""
                for done_file in self.done_files:
                    if done_file.endswith(".yaml"):
                        continue
                    if done_file in self.done_file_dict:
                        code_files += f"""
    ### {done_file}
    ```python
    {self.done_file_dict[done_file]}
    ```

    """
                
                # Generate prompt using the enhanced function
                from prompts import get_coding_prompt, CODE_SCHEMA
                messages = get_coding_prompt(
                    todo_file_name, 
                    detailed_logic_analysis,
                    utility_description,
                    shared_context['paper_content'],
                    shared_context['config_yaml'],
                    shared_context['context_plan'],
                    shared_context['context_six_hats'],
                    shared_context['context_architecture'],
                    #shared_context['context_uml'],
                    shared_context['context_code_structure'],
                    shared_context['context_tasks'],
                    code_files
                )
                
                # Make API call with structured response
                completion = self.api_client.chat_completion(
                    model=self.coding_model,
                    messages=messages,
                    response_format=CODE_SCHEMA,  # Use structured output
                    stream=True
                )
                
                # Parse structured response
                content = completion['choices'][0]['message']['content']
                structured_data = parse_structured_response(content)
                
                # Extract information from structured response
                deliberation = structured_data.get('deliberation', '')
                utility = structured_data.get('utility', '')
                diff_code = self.extract_code_from_structured_response(structured_data, todo_file_name)
                
                # Convert diff to clean code
                clean_code = self.clean_diff_to_code(diff_code)
                
                # Create diff file (keep original diff format)
                diff_path = self.create_diff_file(todo_file_name, diff_code)
                
                # Save artifacts
                safe_filename = todo_file_name.replace("/", "_").replace("\\", "_")
                
                # Save structured response
                with open(f"{self.output_dir}/structured_code_responses/{safe_filename}_structured.json", 'w') as f:
                    json.dump(structured_data, f, indent=2)
                
                # Save full response
                with open(f"{self.output_dir}/coding_artifacts/{safe_filename}_coding.txt", 'w') as f:
                    f.write(content)
                
                # Save deliberation and utility separately for easy access
                with open(f"{self.output_dir}/coding_artifacts/{safe_filename}_deliberation.txt", 'w') as f:
                    f.write(f"DELIBERATION:\n{deliberation}\n\nUTILITY:\n{utility}")
                
                # Write clean code file to repository
                if "/" in todo_file_name:
                    todo_file_dir = '/'.join(todo_file_name.split("/")[:-1])
                    os.makedirs(f"{self.output_repo_dir}/{todo_file_dir}", exist_ok=True)
                
                with open(f"{self.output_repo_dir}/{todo_file_name}", 'w') as f:
                    f.write(clean_code)
                
                return {
                    'filename': todo_file_name,
                    'success': True,
                    'code': clean_code,
                    'diff_code': diff_code,
                    'diff_path': diff_path,
                    'deliberation': deliberation,
                    'utility': utility,
                    'structured_data': structured_data,
                    'content': content
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
        """Process multiple files in parallel with structured responses"""
        
        results = []
        
        # Process in batches
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.generate_single_file, file_info, shared_context): file_info[0]
                for file_info in file_tasks
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(file_tasks), desc="Generating code") as pbar:
                for future in concurrent.futures.as_completed(future_to_file):
                    filename = future_to_file[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result['success']:
                            # Update shared state for subsequent files
                            self.done_files.append(result['filename'])
                            self.done_file_dict[result['filename']] = result['code']
                            print(f"   ✅ Generated: {result['filename']}")
                            print(f"      Utility: {result.get('utility', 'N/A')[:80]}{'...' if len(result.get('utility', '')) > 80 else ''}")
                        else:
                            print(f"   ❌ Failed: {result['filename']}")
                            
                    except Exception as e:
                        print(f"   ❌ Exception for {filename}: {e}")
                        results.append({
                            'filename': filename,
                            'success': False,
                            'error': str(e),
                            'diff_path': None
                        })
                    
                    pbar.update(1)
        
        return results



# functions.py - Missing functions and updates

# 1. Add this function to ArtifactManager class:
def get_utility_descriptions_from_analysis(self, task_list: List[str]) -> Dict[str, str]:
    """Extract utility descriptions from analysis files"""
    utility_descriptions = {}
    
    for filename in task_list:
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

def check_analysis_completion(self, task_list: List[str]) -> bool:
    """Check if analysis phase is complete for all tasks"""
    if not task_list:
        return False
        
    for filename in task_list:
        if filename == "config.yaml":
            continue
            
        safe_filename = filename.replace("/", "_").replace("\\", "_")
        analysis_response_file = os.path.join(self.output_dir, f"{safe_filename}_simple_analysis_response.json")
        analysis_structured_file = os.path.join(self.output_dir, f"{safe_filename}_simple_analysis_structured.json")
        
        if not os.path.exists(analysis_response_file) or not os.path.exists(analysis_structured_file):
            return False
    
    return True

# 2. Update the load_context_and_analysis function signature to return utility descriptions:
def load_context_and_analysis(output_dir: str) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, str]]:
    """Load planning context, analysis data, and utility descriptions"""
    
    # Load structured responses
    with open(f"{output_dir}/all_structured_responses.json") as f:
        structured_responses = json.load(f)
    
    # Load planning trajectories for context
    with open(f"{output_dir}/planning_trajectories.json") as f:
        trajectories = json.load(f)
    
    # Extract context from trajectories (planning, six_hats, architecture, code_structure, tasks responses)
    context_plan = ""
    context_six_hats = ""
    context_architecture = ""
    context_code_structure = ""  # CHANGED: context_uml -> context_code_structure
    context_tasks = ""
    
    for i, msg in enumerate(trajectories):
        if msg.get('role') == 'assistant':
            if 'planning' in trajectories[i-1].get('content', '').lower():
                context_plan = msg['content']
            elif 'six thinking hats' in trajectories[i-1].get('content', '').lower():
                context_six_hats = msg['content']
            elif 'architecture' in trajectories[i-1].get('content', '').lower():
                context_architecture = msg['content']
            elif 'code structure' in trajectories[i-1].get('content', '').lower():  # CHANGED: 'uml' -> 'code structure'
                context_code_structure = msg['content']  # CHANGED: context_uml -> context_code_structure
            elif 'task' in trajectories[i-1].get('content', '').lower():
                context_tasks = msg['content']
    
    # Get task list
    task_list_data = structured_responses.get('task_list', {})
    task_list = task_list_data.get('task_list', [])
    
    # Load analysis for each file
    detailed_logic_analysis_dict = {}
    for todo_file_name in task_list:
        if todo_file_name == "config.yaml":
            continue
            
        safe_filename = todo_file_name.replace("/", "_").replace("\\", "_")
        analysis_file = f"{output_dir}/{safe_filename}_simple_analysis_response.json"
        
        if os.path.exists(analysis_file):
            with open(analysis_file) as f:
                analysis_response = json.load(f)
            detailed_logic_analysis_dict[todo_file_name] = analysis_response[0]['choices'][0]['message']['content']
        else:
            detailed_logic_analysis_dict[todo_file_name] = f"No detailed analysis found for {todo_file_name}"
    
    # Load utility descriptions from analysis structured data
    artifact_manager = ArtifactManager(output_dir)
    utility_descriptions = artifact_manager.get_utility_descriptions_from_analysis(task_list)
    
    context = {
        'context_plan': context_plan,
        'context_six_hats': context_six_hats,
        'context_architecture': context_architecture,
        'context_code_structure': context_code_structure,  # CHANGED: 'context_uml' -> 'context_code_structure'
        'context_tasks': context_tasks,
        'task_list': task_list
    }
    
    return context, detailed_logic_analysis_dict, utility_descriptions


# AutoGen

# Add these imports to the top of your functions.py (after your existing imports)
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from autogen.cache import Cache
import tempfile
import subprocess
import sys
from io import StringIO
import contextlib

# Add these classes to your functions.py file

class CodeEvaluator:
    """Tool for executing and evaluating Python code safely"""
    
    def __init__(self, work_dir: str = None):
        self.work_dir = work_dir or tempfile.mkdtemp()
        os.makedirs(self.work_dir, exist_ok=True)
    
    def execute_code(self, code: str, filename: str = "test_code.py") -> Dict[str, Any]:
        """Execute Python code and return results"""
        try:
            # Create a temporary file for the code
            code_path = os.path.join(self.work_dir, filename)
            with open(code_path, 'w') as f:
                f.write(code)
            
            # Capture stdout and stderr
            result = subprocess.run(
                [sys.executable, code_path],
                cwd=self.work_dir,
                capture_output=True,
                text=True
            )
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode,
                'execution_time': 'completed'
            }
            
        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': f'Execution error: {str(e)}',
                'return_code': -1,
                'execution_time': 'error'
            }
    
    def validate_imports(self, code: str) -> Dict[str, Any]:
        """Check if all imports in code are available"""
        try:
            # Extract import statements
            import_lines = []
            for line in code.split('\n'):
                line = line.strip()
                if line.startswith('import ') or line.startswith('from '):
                    import_lines.append(line)
            
            # Test imports
            test_code = '\n'.join(import_lines) + '\nprint("Imports successful")'
            result = self.execute_code(test_code, "test_imports.py")
            
            return {
                'imports_valid': result['success'],
                'import_errors': result['stderr'] if not result['success'] else '',
                'tested_imports': import_lines
            }
            
        except Exception as e:
            return {
                'imports_valid': False,
                'import_errors': str(e),
                'tested_imports': []
            }


class AutoGenCodingPipeline:
    """Enhanced coding pipeline using AutoGen agents that integrates with existing CodingPipeline"""
    
    def __init__(self, api_client: APIClient, coding_model: str, output_dir: str, 
                 output_repo_dir: str, cache_seed: int = 42):
        self.api_client = api_client
        self.coding_model = coding_model
        self.output_dir = output_dir
        self.output_repo_dir = output_repo_dir
        self.cache_seed = cache_seed
        
        # Create output directories (same as your CodingPipeline)
        os.makedirs(output_repo_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/autogen_artifacts", exist_ok=True)
        os.makedirs(f"{output_dir}/coding_artifacts", exist_ok=True)
        os.makedirs(f"{output_dir}/diffs", exist_ok=True)
        os.makedirs(f"{output_dir}/structured_code_responses", exist_ok=True)
        
        # Initialize code evaluator
        self.evaluator = CodeEvaluator(f"{output_dir}/temp_execution")
        
        # Setup AutoGen configuration
        self.autogen_config = self._setup_autogen_config()
        
        # Initialize agents
        self._setup_agents()
        
        # Track completed files (same as your CodingPipeline)
        self.done_files = ['config.yaml']
        self.done_file_dict = {}
        # AutoGen config should respect APIClient's timeout
        self.autogen_config = [{
            "model": self.coding_model,
            "base_url": f"{self.api_client.base_url}/v1",
            "api_key": self.api_client.api_key or "ollama",
            "temperature": 0.1,
            "seed": self.cache_seed,
            "timeout": self.api_client.default_timeout  # NEW: Use APIClient's timeout
        }]
    def _setup_autogen_config(self) -> List[Dict]:
        """Convert your APIClient configuration to AutoGen format"""
        return [{
            "model": self.coding_model,
            "base_url": f"{self.api_client.base_url}/v1",
            "api_key": self.api_client.api_key or "ollama",
            "temperature": 0.1,
            "seed": self.cache_seed
        }]
    
    def _setup_agents(self):
        """Initialize AutoGen agents with research pipeline focus"""
        
        # Engineer Agent
        self.engineer = AssistantAgent(
            name="Engineer",
            llm_config={
                "config_list": self.autogen_config,
                "cache_seed": self.cache_seed,
            },
            system_message="""
You are an expert Python engineer specializing in research paper implementations.

Your expertise includes:
- Translating research methodologies into clean, efficient Python code
- Writing modular, well-documented code with comprehensive docstrings
- Following software engineering best practices and PEP 8
- Implementing machine learning and data processing pipelines
- Creating robust, testable code architectures

Key requirements for your code:
- Write complete, functional code (no TODOs or placeholders)
- Include proper type hints and comprehensive docstrings
- Add appropriate error handling and input validation
- Make code modular and reusable
- Follow the provided UML design and architecture specifications
- Ensure compatibility with the configuration provided

Focus on producing production-quality code that accurately implements the research methodology.
            """.strip()
        )
        
        # Critic Agent
        self.critic = AssistantAgent(
            name="Critic",
            llm_config={
                "config_list": self.autogen_config,
                "cache_seed": self.cache_seed,
            },
            system_message="""
You are an expert code reviewer specializing in research implementations.

Your task is to thoroughly evaluate code across multiple dimensions:

**Code Quality Dimensions (Score 1-10 for each):**
- **functionality**: Does the code correctly implement the research methodology?
- **code_structure**: Is the code well-organized, modular, and maintainable?
- **documentation**: Are docstrings, comments, and type hints comprehensive?
- **error_handling**: Does the code handle edge cases and errors appropriately?
- **performance**: Is the implementation efficient and scalable?
- **compliance**: Does the code follow the specified architecture and requirements?

**Response Format:**
Provide scores as: {functionality: X, code_structure: X, documentation: X, error_handling: X, performance: X, compliance: X}

After scoring, provide:
- **Strengths**: What the code does well
- **Issues**: Specific problems that need fixing
- **Recommendations**: Concrete actions to improve the code

If ANY critical bugs exist that prevent the code from running, functionality score MUST be < 5.
            """.strip()
        )
        
        # User Proxy Agent
        self.user_proxy = UserProxyAgent(
            name="CodeExecutor",
            code_execution_config={
                "work_dir": f"{self.output_dir}/temp_execution",
                "use_docker": False,
                "last_n_messages": 1,
            },
            human_input_mode="NEVER",
            is_termination_msg=lambda x: (
                x.get("content", "").rstrip().endswith("TERMINATE") or
                "final implementation" in x.get("content", "").lower()
            ),
            system_message="""
You are a code execution coordinator for research implementations.

Your role is to:
1. Execute code provided by the Engineer using built-in code execution
2. Validate imports and basic functionality
3. Report execution results to help improve the implementation
4. Coordinate the feedback loop between Engineer and Critic
5. Determine when the implementation is ready

When code is provided, test it and report:
- Whether it executes without errors
- Import validation results
- Any runtime issues or exceptions
- Basic functionality verification

Keep responses concise and focused on execution results.
            """.strip()
        )
        
        # Group Chat Setup
        self.group_chat = GroupChat(
            agents=[self.user_proxy, self.engineer, self.critic],
            messages=[],
            max_round=12,
            speaker_selection_method="round_robin",
        )
        
        # Group Chat Manager
        self.manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config={
                "config_list": self.autogen_config,
                "cache_seed": self.cache_seed,
            },
            system_message="""
You are a project manager coordinating research implementation between an Engineer and Critic.

Your role is to:
1. Ensure the Engineer produces code that matches requirements and analysis
2. Facilitate thorough code review by the Critic
3. Guide iterative improvement until quality standards are met
4. Decide when the implementation is ready

Success criteria:
- Code implements the research methodology correctly
- Follows software engineering best practices
- Passes code review with scores >= 7 in all dimensions
- Executes without critical errors

Orchestrate the conversation efficiently to achieve high-quality research implementations.
            """.strip()
        )
    
    def generate_file_with_autogen(self, file_info: Tuple[str, str, str], 
                                   shared_context: Dict[str, str]) -> Dict[str, Any]:
        """Generate a single file using AutoGen multi-agent collaboration"""
        
        filename, detailed_analysis, utility_description = file_info
        
        print(f"\n[AUTOGEN] Starting collaborative implementation: {filename}")
        print(f"   Expected utility: {utility_description[:100]}{'...' if len(utility_description) > 100 else ''}")
        
        try:
            # Build context of previously implemented files (same as your approach)
            code_files = ""
            for done_file in self.done_files:
                if done_file.endswith(".yaml"):
                    continue
                if done_file in self.done_file_dict:
                    code_files += f"""
### {done_file}
```python
{self.done_file_dict[done_file][:800]}{'...' if len(self.done_file_dict[done_file]) > 800 else ''}
```

"""
            
            # Create comprehensive implementation prompt
            implementation_prompt = f"""# AutoGen Implementation Task: {filename}

## Expected Utility Value
{utility_description}

## Research Paper Context (Key Sections)
{shared_context['paper_content'][:2000]}...

## Configuration
```yaml
{shared_context['config_yaml'][:1000]}{'...' if len(shared_context['config_yaml']) > 1000 else ''}
```

## Architecture Context
{shared_context['context_architecture'][:1500]}{'...' if len(shared_context['context_architecture']) > 1500 else ''}

## Code Structure
{shared_context['context_code_structure'][:1500]}{'...' if len(shared_context['context_code_structure']) > 1500 else ''}

## Detailed Analysis for {filename}
{detailed_analysis}

## Previously Implemented Files
{code_files}

---

**AUTOGEN COLLABORATION TASK**: Implement `{filename}` based on the research methodology and analysis above.

**SUCCESS CRITERIA**:
- Delivers the expected utility value as specified
- Follows UML design and architecture specifications
- Includes proper docstrings, type hints, and error handling
- Executes without critical errors
- Integrates properly with existing codebase

**DELIVERABLE**: Complete implementation of {filename} as executable Python code."""
            
            # Start AutoGen conversation with caching
            with Cache.disk(cache_seed=self.cache_seed) as cache:
                conversation_result = self.user_proxy.initiate_chat(
                    recipient=self.manager,
                    message=implementation_prompt,
                    cache=cache,
                )
            
            # Extract the final implementation from conversation
            final_code = self._extract_final_code(conversation_result.chat_history, filename)
            
            # Validate the final implementation
            validation_result = self.evaluator.execute_code(final_code, filename)
            import_result = self.evaluator.validate_imports(final_code)
            
            # Save conversation and artifacts (compatible with your structure)
            self._save_autogen_artifacts(filename, conversation_result.chat_history, final_code, utility_description)
            
            # Create structured response format (compatible with your structured responses)
            structured_data = self._create_structured_response(
                filename, final_code, utility_description, detailed_analysis, conversation_result.chat_history
            )
            
            # Create diff and save to repository (same as your approach)
            diff_path = self._create_diff_file(filename, final_code)
            self._save_to_repository(filename, final_code)
            
            return {
                'filename': filename,
                'success': True,
                'code': final_code,
                'diff_path': diff_path,
                'conversation_length': len(conversation_result.chat_history),
                'validation': validation_result,
                'imports': import_result,
                'utility': utility_description,
                'deliberation': self._extract_deliberation(conversation_result.chat_history),
                'structured_data': structured_data,
                'autogen_artifacts': f"{self.output_dir}/autogen_artifacts/{filename.replace('/', '_')}_conversation.json"
            }
            
        except Exception as e:
            print(f"❌ AutoGen error for {filename}: {e}")
            return {
                'filename': filename,
                'success': False,
                'error': str(e),
                'diff_path': None
            }
    
    def _extract_final_code(self, chat_history: List[Dict], filename: str) -> str:
        """Extract the final implementation code from conversation history"""
        
        # Look for the most recent complete Python code block from the Engineer
        for message in reversed(chat_history):
            if message.get('name') == 'Engineer':
                content = message.get('content', '')
                
                # Look for Python code blocks
                import re
                code_blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
                
                if code_blocks:
                    return code_blocks[-1].strip()  # Return the last code block
                
                # Fallback: look for filename header
                if f"# {filename}" in content:
                    return content.split(f"# {filename}", 1)[1].strip()
        
        # If no code found, return a placeholder
        return f"# {filename}\n# Implementation not found in conversation"
    
    def _create_structured_response(self, filename: str, code: str, utility: str, 
                                   analysis: str, chat_history: List[Dict]) -> Dict[str, Any]:
        """Create structured response compatible with your existing format"""
        
        deliberation = self._extract_deliberation(chat_history)
        
        # Format as diff content (compatible with your diff approach)
        diff_content = f"""--- /dev/null
+++ {filename}
@@ -0,0 +1,{len(code.split(chr(10)))} @@
+{code.replace(chr(10), chr(10) + '+')}
"""
        
        return {
            "deliberation": deliberation,
            "utility": utility,
            "files": [{
                "file_name": filename,
                "diff_file": diff_content
            }]
        }
    
    def _extract_deliberation(self, chat_history: List[Dict]) -> str:
        """Extract reasoning and deliberation from AutoGen conversation"""
        
        deliberations = []
        
        for message in chat_history:
            if message.get('name') == 'Engineer':
                content = message.get('content', '')
                if any(keyword in content.lower() for keyword in ['approach', 'design', 'implementation', 'reasoning']):
                    deliberations.append(f"Engineer: {content[:300]}{'...' if len(content) > 300 else ''}")
            
            elif message.get('name') == 'Critic':
                content = message.get('content', '')
                if 'recommendations' in content.lower() or 'feedback' in content.lower():
                    deliberations.append(f"Critic: {content[:300]}{'...' if len(content) > 300 else ''}")
        
        if deliberations:
            return "\n\n".join(deliberations)
        else:
            return "AutoGen collaborative implementation with multi-agent feedback and refinement."
    
    def _save_autogen_artifacts(self, filename: str, chat_history: List[Dict], 
                               final_code: str, utility_description: str):
        """Save AutoGen conversation and artifacts (compatible with your existing structure)"""
        
        safe_filename = filename.replace("/", "_").replace("\\", "_")
        
        # Save conversation history
        conversation_path = f"{self.output_dir}/autogen_artifacts/{safe_filename}_conversation.json"
        with open(conversation_path, 'w') as f:
            json.dump(chat_history, f, indent=2)
        
        # Save final code
        code_path = f"{self.output_dir}/autogen_artifacts/{safe_filename}_final_code.py"
        with open(code_path, 'w') as f:
            f.write(final_code)
        
        # Save full response (compatible with your coding_artifacts structure)
        full_response_path = f"{self.output_dir}/coding_artifacts/{safe_filename}_coding.txt"
        with open(full_response_path, 'w') as f:
            f.write(f"AutoGen Collaborative Implementation\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"File: {filename}\n")
            f.write(f"Utility: {utility_description}\n\n")
            f.write(f"Conversation Summary:\n")
            f.write(f"Messages: {len(chat_history)}\n\n")
            f.write(f"Final Code:\n")
            f.write(f"{'='*30}\n")
            f.write(final_code)
        
        # Save deliberation and utility (compatible with your format)
        deliberation = self._extract_deliberation(chat_history)
        deliberation_path = f"{self.output_dir}/coding_artifacts/{safe_filename}_deliberation.txt"
        with open(deliberation_path, 'w') as f:
            f.write(f"DELIBERATION:\n{deliberation}\n\nUTILITY:\n{utility_description}")
        
        # Save structured response (compatible with your structured_code_responses)
        structured_data = self._create_structured_response(filename, final_code, utility_description, "", chat_history)
        structured_path = f"{self.output_dir}/structured_code_responses/{safe_filename}_structured.json"
        with open(structured_path, 'w') as f:
            json.dump(structured_data, f, indent=2)
        
        print(f"✅ Saved AutoGen artifacts for {filename}")
    
    def _create_diff_file(self, filename: str, code: str) -> str:
        """Create a diff file for the generated code"""
        diff_content = f"""--- /dev/null
+++ {filename}
@@ -0,0 +1,{len(code.split(chr(10)))} @@
+{code.replace(chr(10), chr(10) + '+')}
"""
        
        safe_filename = filename.replace("/", "_").replace("\\", "_")
        diff_path = f"{self.output_dir}/diffs/{safe_filename}.diff"
        
        with open(diff_path, 'w') as f:
            f.write(diff_content)
        
        return diff_path
    
    def _save_to_repository(self, filename: str, code: str):
        """Save the code to the output repository"""
        
        # Create directory structure if needed
        if "/" in filename:
            file_dir = '/'.join(filename.split("/")[:-1])
            os.makedirs(f"{self.output_repo_dir}/{file_dir}", exist_ok=True)
        
        # Write the code file
        with open(f"{self.output_repo_dir}/{filename}", 'w') as f:
            f.write(code)
    
    def process_files_sequential(self, file_tasks: List[Tuple[str, str, str]], 
                                shared_context: Dict[str, str]) -> List[Dict[str, Any]]:
        """Process files sequentially using AutoGen"""
        
        results = []
        
        print(f"\n🤖 Starting AutoGen collaborative coding for {len(file_tasks)} files")
        print("   Processing sequentially for optimal agent collaboration...")
        
        for i, file_info in enumerate(file_tasks, 1):
            filename, detailed_analysis, utility_description = file_info
            print(f"\n[{i}/{len(file_tasks)}] AutoGen Processing: {filename}")
            print(f"   Expected utility: {utility_description[:80]}{'...' if len(utility_description) > 80 else ''}")
            
            result = self.generate_file_with_autogen(file_info, shared_context)
            results.append(result)
            
            if result['success']:
                # Update shared state for subsequent files
                self.done_files.append(result['filename'])
                self.done_file_dict[result['filename']] = result['code']
                print(f"   ✅ Success: {filename}")
                print(f"      Conversation: {result['conversation_length']} messages")
                print(f"      Validation: {'PASS' if result['validation']['success'] else 'FAIL'}")
                print(f"      Imports: {'VALID' if result['imports']['imports_valid'] else 'INVALID'}")
            else:
                print(f"   ❌ Failed: {filename} - {result.get('error', 'Unknown error')}")
            
            # Brief pause between files to avoid API rate limits
            if i < len(file_tasks):
                time.sleep(2)
        
        return results


# Add this function to replace your existing run_coding_phase
def run_autogen_coding_phase(paper_content: str, output_dir: str, output_repo_dir: str,
                            api_client: APIClient, coding_model: str, 
                            development_order: List[str] = None, 
                            cache_seed: int = 42) -> List[Dict[str, Any]]:
    """Enhanced coding phase using AutoGen that replaces your existing run_coding_phase"""
    
    print("\n" + "="*60)
    print("🤖 AUTOGEN ENHANCED CODING PHASE")
    print("="*60)
    print("   Multi-agent collaboration: Engineer + Critic + CodeExecutor + Manager")
    print("   Real-time code validation and iterative improvement")
    print("   Compatible with existing pipeline structure and outputs")
    
    # Check if structured responses exist (same as your original)
    if not os.path.exists(f'{output_dir}/all_structured_responses.json'):
        print("❌ No structured responses found. Run planning and analysis phases first.")
        return []
    
    # Check if config exists (same as your original)
    if not os.path.exists(f'{output_dir}/planning_config.yaml'):
        print("❌ No config file found. Run planning phase first.")
        return []
    
    # Load config (same as your original)
    with open(f'{output_dir}/planning_config.yaml') as f:
        config_yaml = f.read()
    
    # Load context and analysis (using your existing function)
    context, detailed_logic_analysis_dict, utility_descriptions = load_context_and_analysis(output_dir)
    
    # Use development order if provided, otherwise use original task list (same as your original)
    if development_order:
        ordered_files = [f for f in development_order if f != "config.yaml"]
        print(f"\n📝 Using AutoGen with development order ({len(ordered_files)} files):")
    else:
        ordered_files = [f for f in context['task_list'] if f != "config.yaml"]
        print(f"\n📝 Using AutoGen with original task order ({len(ordered_files)} files):")
    
    for i, filename in enumerate(ordered_files, 1):
        utility = utility_descriptions.get(filename, f"Core implementation for {filename}")
        print(f"   {i}. {filename}")
        print(f"      └─ {utility[:60]}{'...' if len(utility) > 60 else ''}")
    
    # Prepare file tasks with utility descriptions using development order (same structure as your original)
    file_tasks = [
        (filename, detailed_logic_analysis_dict[filename], utility_descriptions.get(filename, f"Core implementation for {filename}"))
        for filename in ordered_files
        if filename in detailed_logic_analysis_dict
    ]
    
    if not file_tasks:
        print("❌ No files to generate")
        return []
    
    # Prepare shared context (same as your original)
    shared_context = {
        'paper_content': paper_content,
        'config_yaml': config_yaml,
        'context_plan': context['context_plan'],
        'context_six_hats': context['context_six_hats'],
        'context_architecture': context['context_architecture'],
        #'context_uml': context['context_uml'],
        'context_code_structure': context['context_code_structure'],
        'context_tasks': context['context_tasks']
    }
    
    # Initialize AutoGen coding pipeline (using AutoGen instead of regular CodingPipeline)
    autogen_pipeline = AutoGenCodingPipeline(
        api_client=api_client,
        coding_model=coding_model,
        output_dir=output_dir,
        output_repo_dir=output_repo_dir,
        cache_seed=cache_seed
    )
    
    # Process files sequentially using AutoGen
    results = autogen_pipeline.process_files_sequential(file_tasks, shared_context)
    
    # Copy config file to output repo (same as your original)
    shutil.copy(f'{output_dir}/planning_config.yaml', f'{output_repo_dir}/config.yaml')
    
    # Generate enhanced summary (adapted from your original with AutoGen additions)
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n" + "="*60)
    print("✅ AutoGen collaborative coding completed!")
    print(f"📊 Results: {len(successful)} successful, {len(failed)} failed")
    print(f"📁 Repository: {output_repo_dir}")
    print(f"🤖 Conversation logs: {output_dir}/autogen_artifacts/")
    print(f"📄 Diff files: {output_dir}/diffs/")
    
    if successful:
        print(f"\n✅ AutoGen generated files with multi-agent collaboration:")
        for result in successful:
            print(f"   - {result['filename']}")
            if 'validation' in result:
                status = "✅ PASS" if result['validation']['success'] else "❌ FAIL"
                print(f"     └─ Execution: {status}")
            if 'conversation_length' in result:
                print(f"     └─ Conversation: {result['conversation_length']} messages")
    
    if failed:
        print(f"\n❌ Failed files:")
        for result in failed:
            print(f"   - {result['filename']}: {result.get('error', 'Unknown error')}")
    
    # Generate enhanced results summary (enhanced from your original)
    results_summary = {
        'total_files': len(file_tasks),
        'successful': len(successful),
        'failed': len(failed),
        'autogen_enhanced': True,
        'development_order_used': development_order is not None,
        'cache_seed': cache_seed,
        'results': results,
        'conversation_logs': [r.get('autogen_artifacts', '') for r in successful if 'autogen_artifacts' in r],
        'diff_files': [r['diff_path'] for r in successful if r.get('diff_path')],
        'validation_summary': {
            'passed': len([r for r in successful if r.get('validation', {}).get('success', False)]),
            'failed': len([r for r in successful if not r.get('validation', {}).get('success', True)])
        }
    }
    
    with open(f"{output_dir}/autogen_coding_results.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    return results