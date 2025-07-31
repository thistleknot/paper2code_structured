# main.py
"""
Code generation script that processes analysis results and generates implementation files
Supports parallel processing and diff-based output
"""

from imports import *
from functions import (
    APIClient, PlanningPipeline, EnhancedPlanningPipeline, ArtifactManager, CodingPipeline,  # ADD EnhancedPlanningPipeline
    parse_structured_response, load_paper_content, 
    setup_argument_parser, print_response, load_context_and_analysis,
    format_dict_as_yaml_style, check_pipeline_state, build_shared_context  # ADD missing functions
)
from prompts import (
    PLANNING_SCHEMA, SIX_HATS_SCHEMA, DEPENDENCY_SCHEMA, ARCHITECTURE_SCHEMA, CODE_STRUCTURE_SCHEMA,
    TASK_LIST_SCHEMA, CONFIG_SCHEMA, ANALYSIS_SCHEMA, FILE_ORGANIZATION_SCHEMA,
    get_planning_prompt, get_six_hats_prompt, get_dependency_prompt, get_architecture_prompt, get_code_structure_prompt,
    get_task_list_prompt, get_config_prompt, get_analysis_prompt, get_file_organization_prompt
)


def run_planning_phase(paper_content: str, pipeline: EnhancedPlanningPipeline, 
                      artifact_manager: ArtifactManager) -> Dict[str, Any]:
    """Run the complete planning phase using configuration-driven approach"""
    
    # Check if we should resume from existing data
    structured_responses = artifact_manager.load_structured_responses()
    
    if structured_responses:
        print(f"\n📂 Found existing structured responses, skipping planning phase...")
        print(f"   Available stages: {list(structured_responses.keys())}")
        return structured_responses
    
    print("\n" + "="*60)
    print("📋 STARTING PLANNING PHASE")
    print("="*60)
    
    # Track responses
    responses = []
    structured_responses = {}
    
    # Get ordered list of planning stages (excluding analysis and file_organization)
    planning_stages = ['planning', 'six_hats', 'dependency', 'code_structure', 'architecture', 'task_list', 'config']
    
    for stage_name in planning_stages:
        print("\n" + "="*60)
        print(f"🔄 STAGE: {stage_name.upper()}")
        print("="*60)
        
        # Execute stage using configuration
        response = pipeline.execute_stage_from_config(
            stage_name, paper_content, structured_responses
        )
        print_response(response)
        
        # Track responses
        responses.append(response)
        
        # Parse and save structured data
        response_content = response['choices'][0]['message']['content']
        structured_data = parse_structured_response(response_content)
        structured_responses[stage_name] = structured_data
        
        # Save artifacts
        artifact_manager.save_response(stage_name, response)
        artifact_manager.save_structured_data(stage_name, structured_data)
        
        # Save YAML config if this is the config stage
        if stage_name == 'config':
            artifact_manager.save_config_yaml(structured_data['config_yaml'])
    
    # Save final artifacts
    artifact_manager.save_trajectories(responses)
    artifact_manager.save_model_config(pipeline.reasoning_model, pipeline.coding_model)
    
    # Save combined structured responses for easy access
    with open(f"{artifact_manager.output_dir}/all_structured_responses.json", 'w') as f:
        json.dump(structured_responses, f, indent=2)
    
    print("\n✅ Planning phase completed successfully!")
    
    return structured_responses


def run_analysis_phase(paper_content: str, structured_responses: Dict[str, Any], 
                       pipeline: EnhancedPlanningPipeline, artifact_manager: ArtifactManager) -> None:
    """Run analysis phase using configuration-driven approach"""
    
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
    
    # Process each file using configuration
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
        
        # Execute analysis using configuration (with extra args for file-specific data)
        response = pipeline.execute_stage_from_config(
            stage_name='analysis',
            paper_content=paper_content, 
            structured_responses=structured_responses,
            todo_file_name=todo_file_name, 
            todo_file_desc=todo_file_desc
        )
        print_response(response)
        
        # Parse and save structured data
        analysis_data = parse_structured_response(response['choices'][0]['message']['content'])
        
        # Save artifacts for this file
        artifact_manager.save_analysis_response(todo_file_name, response)
        artifact_manager.save_analysis_structured(todo_file_name, analysis_data)
    
    print(f"\n✅ Analysis phase completed for {len([f for f in task_list if f != 'config.yaml'])} files")


def run_file_organization_phase(structured_responses: Dict[str, Any], 
                               pipeline: EnhancedPlanningPipeline, 
                               artifact_manager: ArtifactManager) -> Dict[str, Any]:
    """Run file organization phase using configuration-driven approach"""
    
    print("\n" + "="*60)
    print("📁 STARTING FILE ORGANIZATION PHASE")
    print("="*60)
    
    # Check if already completed
    file_org_file = os.path.join(artifact_manager.output_dir, "file_organization_structured.json")
    if os.path.exists(file_org_file):
        print("📂 File organization already completed, loading existing data...")
        with open(file_org_file, 'r') as f:
            return json.load(f)
    
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
    
    # Execute file organization using configuration
    response = pipeline.execute_stage_from_config(
        stage_name='file_organization',
        paper_content='',  # No paper content needed
        structured_responses=structured_responses,
        task_list_response=format_dict_as_yaml_style(structured_responses.get('task_list', {}), "task_list"),
        analysis_summaries=analysis_summaries
    )
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


# Update run_coding_phase() to use configuration-driven context building

def run_coding_phase(paper_content: str, output_dir: str, output_repo_dir: str,
                    api_client: APIClient, coding_model: str, max_parallel: int,
                    development_order: List[str] = None) -> List[Dict[str, Any]]:
    """Run the coding phase with configuration-driven context building"""
    
    print("\n" + "="*60)
    print("💻 STARTING CODING PHASE")
    print("="*60)
    
    # Check dependencies
    if not os.path.exists(f'{output_dir}/all_structured_responses.json'):
        print("❌ No structured responses found. Run planning and analysis phases first.")
        return []
    
    if not os.path.exists(f'{output_dir}/planning_config.yaml'):
        print("❌ No config file found. Run planning phase first.")
        return []
    
    # Load config
    with open(f'{output_dir}/planning_config.yaml') as f:
        config_yaml = f.read()
    
    # Load context and analysis using existing function
    context, detailed_logic_analysis_dict, utility_descriptions = load_context_and_analysis(output_dir)
    
    # Load structured responses for configuration-driven context building
    with open(f'{output_dir}/all_structured_responses.json') as f:
        structured_responses = json.load(f)
    
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

    # Build shared context using configuration-driven approach
    shared_context = build_shared_context(structured_responses, paper_content, config_yaml, context)
    
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

# 5. Updated main() function with resume capability, to pass timeout to APIClient constructor
def main():
    """Main execution function with enhanced resume capability"""
    
    # Parse arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Check pipeline state
    pipeline_state = check_pipeline_state(args.output_dir)
    
    print(f"\n🚀 Starting pipeline for: {args.paper_name}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📊 Pipeline state:")
    print(f"   Planning: {'✅' if pipeline_state['planning_complete'] else '❌'}")
    print(f"   Analysis: {'✅' if pipeline_state['analysis_complete'] else '❌'}")  
    print(f"   File Org: {'✅' if pipeline_state['file_organization_complete'] else '❌'}")
    print(f"   Coding:   {'🔄' if pipeline_state['coding_started'] else '❌'}")
    
    # Enhanced resume logic using pipeline state
    if hasattr(args, 'resume_from_coding') and args.resume_from_coding:
        if not pipeline_state['planning_complete']:
            print("❌ Cannot resume from coding: Planning not complete")
            return
        if not pipeline_state['analysis_complete']:
            print("❌ Cannot resume from coding: Analysis not complete")
            return
        print("🔄 Resuming from coding phase...")
        skip_to_coding = True
    elif args.resume_from_analysis:
        if not pipeline_state['planning_complete']:
            print("❌ Cannot resume from analysis: Planning not complete")
            return
        print("🔄 Resuming from analysis phase...")
        skip_to_coding = False
    else:
        # Auto-detect resume point based on pipeline state
        if pipeline_state['planning_complete'] and pipeline_state['analysis_complete']:
            print("🔄 Auto-resume: Planning and analysis complete, resuming from coding...")
            skip_to_coding = True
        elif pipeline_state['planning_complete']:
            print("🔄 Auto-resume: Planning complete, resuming from analysis...")
            skip_to_coding = False
        else:
            print("🆕 Starting fresh pipeline...")
            skip_to_coding = False
    
    # Initialize components
    api_client = APIClient(base_url=args.api_base_url, api_key=args.api_key, 
                          initial_seed=args.seed, default_timeout=args.timeout)
    pipeline = EnhancedPlanningPipeline(args.reasoning_model, args.coding_model, api_client)
    artifact_manager = ArtifactManager(args.output_dir)
    
    print(f"Reasoning model: {args.reasoning_model}")
    print(f"Coding model: {args.coding_model}")
    print(f"API timeout: {args.timeout}s")
    print(f"Max parallel tasks: {args.max_parallel}")
    print(f"Repository directory: {args.output_repo_dir}")
    print(f"Initial seed: {args.seed}")
    
    # Load paper content
    print(f"\n📄 Loading paper from: {args.paper_markdown_path}")
    paper_content = load_paper_content(args.paper_markdown_path)
    
    # Execute pipeline phases based on resume state
    if skip_to_coding:
        # Load existing structured responses
        structured_responses = artifact_manager.load_structured_responses()
        print(f"📂 Loaded existing structured responses: {list(structured_responses.keys())}")
        
        # Load or run file organization
        if pipeline_state['file_organization_complete']:
            with open(f"{args.output_dir}/file_organization_structured.json", 'r') as f:
                file_org_data = json.load(f)
            print("📂 Loaded existing file organization")
        else:
            print("🔄 Running file organization phase...")
            file_org_data = run_file_organization_phase(structured_responses, pipeline, artifact_manager)
        
    else:
        # Run planning phase if needed
        if not pipeline_state['planning_complete']:
            structured_responses = run_planning_phase(paper_content, pipeline, artifact_manager)
        else:
            structured_responses = artifact_manager.load_structured_responses()
            print(f"📂 Loaded existing structured responses: {list(structured_responses.keys())}")
        
        # Run analysis phase if needed
        if not pipeline_state['analysis_complete']:
            run_analysis_phase(paper_content, structured_responses, pipeline, artifact_manager)
        else:
            print("📂 Analysis already complete, skipping...")
        
        # Run file organization phase
        if pipeline_state['file_organization_complete']:
            with open(f"{args.output_dir}/file_organization_structured.json", 'r') as f:
                file_org_data = json.load(f)
            print("📂 Loaded existing file organization")
        else:
            file_org_data = run_file_organization_phase(structured_responses, pipeline, artifact_manager)
    
    # Run coding phase with enhanced resume capability
    print(f"\n{'='*60}")
    print("💻 CODING PHASE WITH FILE-LEVEL RESUME")
    print(f"{'='*60}")
    
    # Optional: Force regenerate (if you had this flag)
    if hasattr(args, 'force_regenerate') and args.force_regenerate:
        print("⚠️  Force regenerate enabled - will overwrite existing files")
        # Clear existing generated files
        for dir_name in ['structured_code_responses', 'diffs', 'coding_artifacts']:
            dir_path = f"{args.output_dir}/{dir_name}"
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                os.makedirs(dir_path, exist_ok=True)
        
        # Clear repository directory
        if os.path.exists(args.output_repo_dir):
            for file in os.listdir(args.output_repo_dir):
                if file.endswith('.py'):
                    os.remove(f"{args.output_repo_dir}/{file}")
    
    try:
        results = run_coding_phase(
            paper_content=paper_content,
            output_dir=args.output_dir,
            output_repo_dir=args.output_repo_dir,
            api_client=api_client,
            coding_model=args.coding_model,
            max_parallel=args.max_parallel,
            development_order=file_org_data.get('development_order', [])
        )
        
        # Enhanced final summary
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        resumed = [r for r in results if r.get('resumed', False)]
        
        print(f"\n" + "="*60)
        print("✅ Complete pipeline finished!")
        print(f"📊 Final Results:")
        print(f"   Total files: {len(results)}")
        print(f"   Successful: {len(successful)}")
        print(f"   Failed: {len(failed)}")
        if resumed:
            print(f"   Resumed: {len(resumed)}")
        print(f"📁 All artifacts saved to: {args.output_dir}")
        
        if successful:
            print(f"\n✅ Generated files:")
            for result in successful:
                status = "📂" if result.get('resumed', False) else "✨"
                print(f"   {status} {result['filename']}")
        
        if failed:
            print(f"\n❌ Failed files:")
            for result in failed:
                print(f"   - {result['filename']}: {result.get('error', 'Unknown error')}")
        
        print(f"\n💡 Resume capability:")
        print(f"   If interrupted, rerun the same command to continue where you left off")
        print(f"   Completed files are automatically detected and skipped")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Pipeline interrupted by user (Ctrl-C)")
        print(f"💡 To resume where you left off, run the exact same command:")
        print(f"   python main.py --paper_name {args.paper_name} --paper_markdown_path {args.paper_markdown_path} --output_dir {args.output_dir} --output_repo_dir {args.output_repo_dir}")
        print(f"   The system will automatically detect completed files and skip them.")


if __name__ == "__main__":
    main()