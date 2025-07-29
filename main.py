# main.py
"""
Code generation script that processes analysis results and generates implementation files
Supports parallel processing and diff-based output
"""

from imports import *
from functions import (
    APIClient, PlanningPipeline, ArtifactManager, CodingPipeline,
    parse_structured_response, load_paper_content, 
    setup_argument_parser, print_response, load_context_and_analysis,
    format_dict_as_yaml_style
)
from prompts import (
    PLANNING_SCHEMA, SIX_HATS_SCHEMA, DEPENDENCY_SCHEMA, ARCHITECTURE_SCHEMA, CODE_STRUCTURE_SCHEMA,
    TASK_LIST_SCHEMA, CONFIG_SCHEMA, ANALYSIS_SCHEMA, FILE_ORGANIZATION_SCHEMA,  # NEW
    get_planning_prompt, get_six_hats_prompt, get_dependency_prompt, get_architecture_prompt, get_code_structure_prompt,
    get_task_list_prompt, get_config_prompt, get_analysis_prompt, get_file_organization_prompt  # NEW
)


def run_planning_phase(paper_content: str, pipeline: PlanningPipeline, 
                      artifact_manager: ArtifactManager) -> Dict[str, Any]:
    """Run the complete planning phase and return structured responses"""
    
    # Check if we should resume from existing data
    structured_responses = artifact_manager.load_structured_responses()
    
    if structured_responses:
        print(f"\n📂 Found existing structured responses, skipping planning phase...")
        print(f"   Available stages: {list(structured_responses.keys())}")
        return structured_responses
    
    print("\n" + "="*60)
    print("📋 STARTING PLANNING PHASE")
    print("="*60)
    
    # Track responses and trajectories
    responses = []
    trajectories = []
    structured_responses = {}
    
    # Stage 1: Planning
    print("\n" + "="*60)
    stage_name = "planning"
    messages = get_planning_prompt(paper_content)
    trajectories.extend(messages)
    
    response = pipeline.execute_stage(stage_name, messages, True, PLANNING_SCHEMA)
    print_response(response)
    
    responses.append(response)
    trajectories.append({'role': 'assistant', 'content': response['choices'][0]['message']['content']})
    
    # Parse and save structured data
    planning_data = parse_structured_response(response['choices'][0]['message']['content'])
    structured_responses[stage_name] = planning_data
    
    artifact_manager.save_response(stage_name, response)
    artifact_manager.save_structured_data(stage_name, planning_data)
    
    # Stage 2: Six Thinking Hats Analysis
    print("\n" + "="*60)
    stage_name = "six_hats"
    messages = get_six_hats_prompt(paper_content, format_dict_as_yaml_style(planning_data, "planning"))
    trajectories.extend(messages)
    
    response = pipeline.execute_stage(stage_name, messages, True, SIX_HATS_SCHEMA)
    print_response(response)
    
    responses.append(response)
    trajectories.append({'role': 'assistant', 'content': response['choices'][0]['message']['content']})
    
    # Parse and save structured data
    six_hats_data = parse_structured_response(response['choices'][0]['message']['content'])
    structured_responses[stage_name] = six_hats_data
    
    artifact_manager.save_response(stage_name, response)
    artifact_manager.save_structured_data(stage_name, six_hats_data)
    
    # Stage 3: Dependency Analysis
    print("\n" + "="*60)
    stage_name = "dependency"
    messages = get_dependency_prompt(
        paper_content,
        format_dict_as_yaml_style(planning_data, "planning"),
        format_dict_as_yaml_style(six_hats_data, "six_hats")
    )
    trajectories.extend(messages)
    
    response = pipeline.execute_stage(stage_name, messages, True, DEPENDENCY_SCHEMA)
    print_response(response)
    
    responses.append(response)
    trajectories.append({'role': 'assistant', 'content': response['choices'][0]['message']['content']})
    
    # Parse and save structured data
    dependency_data = parse_structured_response(response['choices'][0]['message']['content'])
    structured_responses[stage_name] = dependency_data
    
    artifact_manager.save_response(stage_name, response)
    artifact_manager.save_structured_data(stage_name, dependency_data)
        
    # Stage 4: Code Structure
    print("\n" + "="*60)
    stage_name = "code_structure"
    messages = get_code_structure_prompt(
        paper_content,
        format_dict_as_yaml_style(planning_data, "planning"),
        format_dict_as_yaml_style(six_hats_data, "six_hats"),
        format_dict_as_yaml_style(dependency_data, "dependency")
    )
    trajectories.extend(messages)
    
    response = pipeline.execute_stage(stage_name, messages, True, CODE_STRUCTURE_SCHEMA)
    print_response(response)
    
    responses.append(response)
    trajectories.append({'role': 'assistant', 'content': response['choices'][0]['message']['content']})
    
    # Parse and save structured data
    code_structure_data = parse_structured_response(response['choices'][0]['message']['content'])  # FIXED: Change uml_data to code_structure_data
    structured_responses[stage_name] = code_structure_data  # FIXED: Change uml_data to code_structure_data
    
    artifact_manager.save_response(stage_name, response)
    artifact_manager.save_structured_data(stage_name, code_structure_data)  # FIXED: Change uml_data to code_structure_data
    
    # Stage 5: Architecture Design
    print("\n" + "="*60)
    stage_name = "architecture"
    messages = get_architecture_prompt(
        paper_content, 
        format_dict_as_yaml_style(planning_data, "planning"),
        format_dict_as_yaml_style(six_hats_data, "six_hats"),
        format_dict_as_yaml_style(dependency_data, "dependency"),
        format_dict_as_yaml_style(code_structure_data, "code_structure")  # FIXED: Use code_structure_data
    )
    trajectories.extend(messages)
    
    response = pipeline.execute_stage(stage_name, messages, True, ARCHITECTURE_SCHEMA)
    print_response(response)
    
    responses.append(response)
    trajectories.append({'role': 'assistant', 'content': response['choices'][0]['message']['content']})
    
    # Parse and save structured data
    architecture_data = parse_structured_response(response['choices'][0]['message']['content'])
    structured_responses[stage_name] = architecture_data
    
    artifact_manager.save_response(stage_name, response)
    artifact_manager.save_structured_data(stage_name, architecture_data)
    
    # Stage 6: Task List
    print("\n" + "="*60)
    stage_name = "task_list"
    messages = get_task_list_prompt(
        paper_content,
        format_dict_as_yaml_style(planning_data, "planning"),
        format_dict_as_yaml_style(six_hats_data, "six_hats"),
        format_dict_as_yaml_style(dependency_data, "dependency"),
        format_dict_as_yaml_style(code_structure_data, "code_structure"),  # FIXED
        format_dict_as_yaml_style(architecture_data, "architecture")
    )
    trajectories.extend(messages)
    
    response = pipeline.execute_stage(stage_name, messages, True, TASK_LIST_SCHEMA)
    print_response(response)
    
    responses.append(response)
    trajectories.append({'role': 'assistant', 'content': response['choices'][0]['message']['content']})
    
    # Parse and save structured data
    task_list_data = parse_structured_response(response['choices'][0]['message']['content'])
    structured_responses[stage_name] = task_list_data
    
    artifact_manager.save_response(stage_name, response)
    artifact_manager.save_structured_data(stage_name, task_list_data)
    
    # Stage 7: Configuration
    print("\n" + "="*60)
    stage_name = "config"
    messages = get_config_prompt(
        paper_content,
        format_dict_as_yaml_style(planning_data, "planning"),
        format_dict_as_yaml_style(six_hats_data, "six_hats"),
        format_dict_as_yaml_style(dependency_data, "dependency"),
        format_dict_as_yaml_style(code_structure_data, "code_structure"),  # FIXED
        format_dict_as_yaml_style(architecture_data, "architecture"),
        format_dict_as_yaml_style(task_list_data, "task_list")
    )
    trajectories.extend(messages)
    
    response = pipeline.execute_stage(stage_name, messages, True, CONFIG_SCHEMA)
    print_response(response)
    
    responses.append(response)
    trajectories.append({'role': 'assistant', 'content': response['choices'][0]['message']['content']})
    
    # Parse and save structured data
    config_data = parse_structured_response(response['choices'][0]['message']['content'])
    structured_responses[stage_name] = config_data
    
    artifact_manager.save_response(stage_name, response)
    artifact_manager.save_structured_data(stage_name, config_data)
    
    # Save the YAML config file
    artifact_manager.save_config_yaml(config_data['config_yaml'])
    
    # Save final artifacts
    artifact_manager.save_trajectories(trajectories)
    artifact_manager.save_model_config(pipeline.reasoning_model, pipeline.coding_model)
    
    # Save combined structured responses for easy access
    with open(f"{artifact_manager.output_dir}/all_structured_responses.json", 'w') as f:
        json.dump(structured_responses, f, indent=2)
    
    print("\n✅ Planning phase completed successfully!")
    
    return structured_responses


def run_analysis_phase(paper_content: str, structured_responses: Dict[str, Any], 
                       pipeline: PlanningPipeline, artifact_manager: ArtifactManager) -> None:
    """Run analysis phase iterating over task list"""
    
    print("\n" + "="*60)
    print("🔍 STARTING ANALYSIS PHASE")
    print("="*60)
    
    # Extract task list and logic analysis
    task_list = artifact_manager.get_task_list_from_responses(structured_responses)
    logic_analysis_dict = artifact_manager.get_logic_analysis_from_responses(structured_responses)
    
    if not task_list:
        print("❌ No task list found in structured responses")
        return
    
    # Check if analysis is already complete
    if artifact_manager.check_analysis_completion(task_list):
        print(f"📂 Analysis phase already completed for all {len(task_list)} files")
        print("   Skipping analysis phase...")
        return
    
    print(f"📋 Analyzing {len(task_list)} files:")
    for i, filename in enumerate(task_list, 1):
        print(f"   {i}. {filename}")
    
    # Prepare context strings using token wrapping
    planning_str = format_dict_as_yaml_style(structured_responses.get('planning', {}), "planning")
    six_hats_str = format_dict_as_yaml_style(structured_responses.get('six_hats', {}), "six_hats")
    dependency_str = format_dict_as_yaml_style(structured_responses.get('dependency', {}), "dependency")
    architecture_str = format_dict_as_yaml_style(structured_responses.get('architecture', {}), "architecture")
    #uml_str = format_dict_as_yaml_style(structured_responses.get('uml', {}), "uml")
    code_structure_str = format_dict_as_yaml_style(structured_responses.get('code_structure', {}), "code_structure")  # FIXED
    task_list_str = format_dict_as_yaml_style(structured_responses.get('task_list', {}), "task_list")
    
    # Process each file
    for todo_file_name in tqdm(task_list, desc="Analyzing files"):
        if todo_file_name == "config.yaml":
            continue
            
        # Check if this file's analysis already exists
        safe_filename = todo_file_name.replace("/", "_").replace("\\", "_")
        analysis_response_file = os.path.join(artifact_manager.output_dir, f"{safe_filename}_simple_analysis_response.json")
        analysis_structured_file = os.path.join(artifact_manager.output_dir, f"{safe_filename}_simple_analysis_structured.json")
        
        if os.path.exists(analysis_response_file) and os.path.exists(analysis_structured_file):
            print(f"\n[ANALYSIS] {todo_file_name} - Already exists, skipping")
            continue
            
        print(f"\n[ANALYSIS] {todo_file_name}")
        
        # Get file description from logic analysis
        todo_file_desc = logic_analysis_dict.get(todo_file_name, "")
        
        # Generate analysis prompt
        from prompts import get_analysis_prompt, ANALYSIS_SCHEMA
        messages = get_analysis_prompt(
            paper_content, 
            planning_str,
            six_hats_str, 
            dependency_str,
            architecture_str,
            #uml_str,
            code_structure_str,
            task_list_str,
            todo_file_name, 
            todo_file_desc
        )
        
        # Execute analysis
        response = pipeline.execute_stage("analysis", messages, True, ANALYSIS_SCHEMA)
        print_response(response)
        
        # Parse and save structured data
        analysis_data = parse_structured_response(response['choices'][0]['message']['content'])
        
        # Save artifacts for this file
        artifact_manager.save_analysis_response(todo_file_name, response)
        artifact_manager.save_analysis_structured(todo_file_name, analysis_data)
    
    print(f"\n✅ Analysis phase completed for {len([f for f in task_list if f != 'config.yaml'])} files")

def run_file_organization_phase(structured_responses: Dict[str, Any], 
                               pipeline: PlanningPipeline, 
                               artifact_manager: ArtifactManager) -> Dict[str, Any]:
    """Run file organization phase to order files by development dependencies"""
    
    print("\n" + "="*60)
    print("📁 STARTING FILE ORGANIZATION PHASE")
    print("="*60)
    
    # Check if already completed
    file_org_file = os.path.join(artifact_manager.output_dir, "file_organization_structured.json")
    if os.path.exists(file_org_file):
        print("📂 File organization already completed, loading existing data...")
        with open(file_org_file, 'r') as f:
            return json.load(f)
    
    # Get task list data
    task_list_data = structured_responses.get('task_list', {})
    task_list_str = format_dict_as_yaml_style(task_list_data, "task_list")
    
    # Gather analysis summaries for all files
    task_list = artifact_manager.get_task_list_from_responses(structured_responses)
    analysis_summaries = ""
    
    for filename in task_list:
        if filename == "config.yaml":
            continue
            
        safe_filename = filename.replace("/", "_").replace("\\", "_")
        analysis_file = os.path.join(artifact_manager.output_dir, f"{safe_filename}_simple_analysis_structured.json")
        
        if os.path.exists(analysis_file):
            with open(analysis_file, 'r') as f:
                analysis_data = json.load(f)
            
            # Create summary for this file
            analysis_summaries += f"""
## {filename}
Core Functionality: {analysis_data.get('core_functionality', 'N/A')}
Methods Focus: {', '.join(analysis_data.get('focused_requirements', {}).get('methods_focus', []))}
Classes Focus: {', '.join(analysis_data.get('focused_requirements', {}).get('classes_focus', []))}

"""
    
    print(f"📋 Organizing {len([f for f in task_list if f != 'config.yaml'])} files by development order")
    
    # Generate file organization prompt
    from prompts import get_file_organization_prompt, FILE_ORGANIZATION_SCHEMA
    messages = get_file_organization_prompt(task_list_str, analysis_summaries)
    
    # Execute file organization
    response = pipeline.execute_stage("file_organization", messages, True, FILE_ORGANIZATION_SCHEMA)
    print_response(response)
    
    # Parse and save structured data
    file_org_data = parse_structured_response(response['choices'][0]['message']['content'])
    
    # Save artifacts
    artifact_manager.save_file_organization_response(response)
    artifact_manager.save_file_organization_structured(file_org_data)
    
    # Display development order
    development_order = file_org_data.get('development_order', [])
    print(f"\n📋 Development Order ({len(development_order)} files):")
    for i, filename in enumerate(development_order, 1):
        print(f"   {i}. {filename}")
    
    print("\n✅ File organization phase completed!")
    
    return file_org_data

def run_coding_phase(paper_content: str, output_dir: str, output_repo_dir: str,
                    api_client: APIClient, coding_model: str, max_parallel: int,
                    development_order: List[str] = None) -> List[Dict[str, Any]]:  # ADD THIS
    """Run the coding phase with resume capability"""
    
    print("\n" + "="*60)
    print("💻 STARTING CODING PHASE")
    print("="*60)
    
    # Check if structured responses exist
    if not os.path.exists(f'{output_dir}/all_structured_responses.json'):
        print("❌ No structured responses found. Run planning and analysis phases first.")
        return []
    
    # Check if config exists
    if not os.path.exists(f'{output_dir}/planning_config.yaml'):
        print("❌ No config file found. Run planning phase first.")
        return []
    
    # Load config
    with open(f'{output_dir}/planning_config.yaml') as f:
        config_yaml = f.read()
    
    # Load context and analysis 
    context, detailed_logic_analysis_dict, utility_descriptions = load_context_and_analysis(output_dir)
    
    # Use development order if provided, otherwise use original task list
    if development_order:
        ordered_files = [f for f in development_order if f != "config.yaml"]
        print(f"\n📝 Using development order ({len(ordered_files)} files):")
    else:
        ordered_files = [f for f in context['task_list'] if f != "config.yaml"]
        print(f"\n📝 Using original task order ({len(ordered_files)} files):")
    
    for i, filename in enumerate(ordered_files, 1):
        utility = utility_descriptions.get(filename, f"Core implementation for {filename}")
        print(f"   {i}. {filename}")
        print(f"      └─ {utility[:60]}{'...' if len(utility) > 60 else ''}")
    
    # Prepare file tasks with utility descriptions using development order
    file_tasks = [
        (filename, detailed_logic_analysis_dict[filename], utility_descriptions.get(filename, f"Core implementation for {filename}"))
        for filename in ordered_files
        if filename in detailed_logic_analysis_dict
    ]
    
    if not file_tasks:
        print("❌ No files to generate")
        return []
    
    print(f"\n📝 Generating {len(file_tasks)} files with structured responses:")
    for i, (filename, _, utility) in enumerate(file_tasks, 1):
        print(f"   {i}. {filename}")
        print(f"      └─ {utility[:60]}{'...' if len(utility) > 60 else ''}")
    
    # Initialize enhanced coding pipeline
    coding_pipeline = CodingPipeline(
        api_client=api_client,
        coding_model=coding_model,
        output_dir=output_dir,
        output_repo_dir=output_repo_dir,
        max_parallel=max_parallel
    )

    # Add this before line 384:
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
    
    # Process files in parallel with structured responses
    results = coding_pipeline.process_files_parallel(file_tasks, shared_context)
    
    # Copy config file to output repo
    shutil.copy(f'{output_dir}/planning_config.yaml', f'{output_repo_dir}/config.yaml')
    
    # Generate enhanced summary with utility information
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n" + "="*60)
    print("✅ Coding phase completed!")
    print(f"📊 Results: {len(successful)} successful, {len(failed)} failed")
    print(f"📁 Repository: {output_repo_dir}")
    print(f"📄 Diff files: {output_dir}/diffs/")
    print(f"📋 Structured responses: {output_dir}/structured_code_responses/")
    
    if successful:
        print(f"\n✅ Generated files with utilities:")
        for result in successful:
            print(f"   - {result['filename']}")
            if 'utility' in result:
                print(f"     └─ {result['utility'][:80]}{'...' if len(result['utility']) > 80 else ''}")
    
    if failed:
        print(f"\n❌ Failed files:")
        for result in failed:
            print(f"   - {result['filename']}: {result.get('error', 'Unknown error')}")
    
    # Generate enhanced results summary
    results_summary = {
        'total_files': len(file_tasks),
        'successful': len(successful),
        'failed': len(failed),
        'results': results,
        'diff_files': [r['diff_path'] for r in successful if r['diff_path']],
        'utilities': {r['filename']: r.get('utility', '') for r in successful}
    }
    
    with open(f"{output_dir}/coding_results.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    return results

# Add this function to your main.py (after your other phase functions)

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
    
    # Load context and analysis (using your existing function from functions.py)
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
    # Import the AutoGen class from functions.py
    from functions import AutoGenCodingPipeline
    
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

# 5. Updated main() function with resume capability:
# 2. Update main() to pass timeout to APIClient constructor
def main():
    """Main execution function"""
    
    # Parse arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Initialize components with timeout ONLY passed to APIClient
    api_client = APIClient(base_url=args.api_base_url, api_key=args.api_key, 
                          initial_seed=args.seed, default_timeout=args.timeout)  # NEW: Pass timeout here
    pipeline = PlanningPipeline(args.reasoning_model, args.coding_model, api_client)  # NO timeout here
    artifact_manager = ArtifactManager(args.output_dir)
    
    print(f"\n🚀 Starting pipeline for: {args.paper_name}")
    print(f"Reasoning model: {args.reasoning_model}")
    print(f"Coding model: {args.coding_model}")
    print(f"API timeout: {args.timeout}s")  # NEW: Show timeout
    print(f"Max parallel tasks: {args.max_parallel}")
    print(f"Output directory: {args.output_dir}")
    print(f"Repository directory: {args.output_repo_dir}")
    print(f"Initial seed: {args.seed}")
    print(f"Resume from analysis: {args.resume_from_analysis}")
    
    # Rest of main() remains exactly the same - no timeout parameters passed anywhere else
    
    # Load paper content
    print(f"\n📄 Loading paper from: {args.paper_markdown_path}")
    paper_content = load_paper_content(args.paper_markdown_path)
    
    # Check resume conditions
    if args.resume_from_analysis:
        print(f"\n🔄 Checking resume conditions...")
        
        # Check if structured responses exist
        structured_responses = artifact_manager.load_structured_responses()
        if not structured_responses:
            print("   ❌ No structured responses found, running full pipeline")
            args.resume_from_analysis = False
        else:
            # Check if analysis is complete
            task_list = artifact_manager.get_task_list_from_responses(structured_responses)
            if not artifact_manager.check_analysis_completion(task_list):
                print("   ⚠️  Analysis incomplete, resuming from analysis phase")
                # Run analysis phase only
                run_analysis_phase(paper_content, structured_responses, pipeline, artifact_manager)
            else:
                print("   ✅ Analysis complete, skipping to coding phase")
    
    # Run pipeline phases based on resume state
    if not args.resume_from_analysis:
        # Run planning phase
        structured_responses = run_planning_phase(paper_content, pipeline, artifact_manager)
        
        # Run analysis phase
        run_analysis_phase(paper_content, structured_responses, pipeline, artifact_manager)
    else:
        # Load existing structured responses
        structured_responses = artifact_manager.load_structured_responses()
        print(f"📂 Loaded existing structured responses: {list(structured_responses.keys())}")
    
    # Run file organization phase
    file_org_data = run_file_organization_phase(structured_responses, pipeline, artifact_manager)
    
    
    # Run coding phase with organized file order
    results = run_coding_phase(
        paper_content=paper_content,
        output_dir=args.output_dir,
        output_repo_dir=args.output_repo_dir,
        api_client=api_client,
        coding_model=args.coding_model,
        max_parallel=args.max_parallel,
        development_order=file_org_data.get('development_order', [])  # NEW: Pass development order
    )
    """
        
    results = run_autogen_coding_phase(
        paper_content=paper_content,
        output_dir=args.output_dir,
        output_repo_dir=args.output_repo_dir,
        api_client=api_client,
        coding_model=args.coding_model,
        development_order=file_org_data.get('development_order', []),
        cache_seed=args.seed
    )
    """
    # Final summary
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n" + "="*60)
    print("✅ Complete pipeline finished!")
    print(f"📊 Final Results: {len(successful)} successful, {len(failed)} failed")
    print(f"📁 All artifacts saved to: {args.output_dir}")
    
    print("\n📋 Generated artifacts:")
    if not args.resume_from_analysis:
        print("- Planning phase: planning_response.json & planning_structured.json")
        print("- Six hats: six_hats_response.json & six_hats_structured.json") 
        print("- Dependency: dependency_response.json & dependency_structured.json")
        print("- UML: uml_response.json & uml_structured.json")
        print("- Architecture: architecture_response.json & architecture_structured.json")
        print("- Task list: task_list_response.json & task_list_structured.json")
        print("- Config: config_response.json & config_structured.json")
        print("- planning_config.yaml")
        print("- planning_trajectories.json")
        print("- model_config.json")
        print("- all_structured_responses.json")
    
    print("- Individual analysis files for each task")
    print("- Structured code responses in /structured_code_responses/")
    print("- Deliberation and utility files in /coding_artifacts/")
    print(f"- coding_results.json")
    
    if successful:
        print(f"\n🎯 Generated {len(successful)} files with utilities:")
        for result in successful:
            utility = result.get('utility', 'No utility description')
            print(f"   {result['filename']}: {utility[:60]}{'...' if len(utility) > 60 else ''}")


if __name__ == "__main__":
    main()