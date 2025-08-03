# main.py
"""
Code generation script that processes analysis results and generates implementation files
Supports parallel processing and diff-based output
"""

from imports import *
from functions import (
    APIClient, WhiteboardPipeline, WhiteboardManager, ArtifactManager, CodingPipeline,
    parse_structured_response, load_paper_content, 
    setup_argument_parser, print_response, load_context_and_analysis_from_whiteboard,
    check_pipeline_state, validate_autogen_prerequisites,
    load_generated_code_files, save_category_implementation
)
from prompts import (
    PLANNING_SCHEMA, SIX_HATS_SCHEMA, DEPENDENCY_SCHEMA, ARCHITECTURE_SCHEMA, CODE_STRUCTURE_SCHEMA,
    TASK_LIST_SCHEMA, CONFIG_SCHEMA, ANALYSIS_SCHEMA, FILE_ORGANIZATION_SCHEMA,
    get_planning_prompt, get_six_hats_prompt, get_dependency_prompt, get_architecture_prompt, get_code_structure_prompt,
    get_task_list_prompt, get_config_prompt, get_analysis_prompt, get_file_organization_prompt, get_file_classification_prompt,
    get_gap_analysis_prompt, GAP_ANALYSIS_SCHEMA, get_category_implementation_prompt, CATEGORY_IMPLEMENTATION_SCHEMA, FILE_CLASSIFICATION_SCHEMA
)



def analyze_gaps(whiteboard_manager: WhiteboardManager,
                 output_repo_dir: str,
                 api_client: APIClient,
                 reasoning_model: str,
                 snapshot: Optional[str] = None) -> Dict[str, Any]:
    """Analyzes gaps against current implementation with optional snapshot comparison."""
    # Load appropriate code files
    prefix = 'corrected_' if snapshot else ''
    code_files = load_generated_code_files(output_repo_dir, prefix=prefix)
    
    if "No generated" in code_files:
        return {"error": "No code files to analyze"}

    # Load comparison context
    context = whiteboard_manager.get_whiteboard_yaml()
    if snapshot:
        snapshot_data = whiteboard_manager.load_whiteboard().get('snapshots', {}).get(snapshot)
        if snapshot_data:
            context += f"\n# Comparison Snapshot '{snapshot}':\n" + yaml.dump(snapshot_data['state'])

    # Execute gap analysis
    response = api_client.chat_completion(
        model=reasoning_model,
        messages=get_gap_analysis_prompt(
            current_code_files=code_files,
            whiteboard_yaml=context
        ),
        response_format=GAP_ANALYSIS_SCHEMA,
        stream=True
    )

    gap_data = parse_structured_response(response['choices'][0]['message']['content'])
    
    # Standardize counts by corrected_* category
    gap_counts = {
        "corrected_functions": len(gap_data.get('undefined_items', {}).get('corrected_functions', [])),
        "corrected_classes": len(gap_data.get('undefined_items', {}).get('corrected_classes', [])),
        "corrected_constants": len(gap_data.get('undefined_items', {}).get('corrected_constants', [])),
        "corrected_imports": len(gap_data.get('undefined_items', {}).get('corrected_imports', [])),
        "corrected_main": len(gap_data.get('undefined_items', {}).get('corrected_main', [])),
        "corrected_config": len(gap_data.get('undefined_items', {}).get('corrected_config', []))
    }
    
    return {
        "total_gaps": sum(gap_counts.values()),
        "by_category": gap_counts,
        "updates": gap_data.get('updates', [])
    }
    
    
def run_planning_phase(paper_content: str, pipeline: WhiteboardPipeline, 
                      artifact_manager: ArtifactManager, whiteboard_manager: WhiteboardManager) -> Dict[str, Any]:
    """Run the complete planning phase using whiteboard-driven approach"""
    
    # Check whiteboard state for existing planning
    whiteboard = whiteboard_manager.load_whiteboard()
    planning_completed = whiteboard.get('pipeline', {}).get('stages', {}).get('config', {}).get('completed', False)
    
    if planning_completed:
        print(f"\n📂 Found existing planning in whiteboard, skipping planning phase...")
        print(f"   Whiteboard contains accumulated knowledge from previous stages")
        return whiteboard
    
    print("\n" + "="*60)
    print("📋 STARTING WHITEBOARD-BASED PLANNING PHASE")
    print("="*60)
    print("   Persistent knowledge accumulation via JSON whiteboard")
    print("   Each stage updates shared state for downstream processes")
    
    # Track responses for traditional compatibility
    responses = []
    
    # Use PipelineConfig to get ordered planning stages
    from functions import PipelineConfig
    planning_stages = ['planning', 'six_hats', 'dependency', 'code_structure', 'architecture', 'task_list', 'config']
    
    for stage_name in planning_stages:
        print("\n" + "="*60)
        print(f"🔄 STAGE: {stage_name.upper()}")
        print("="*60)
        
        # Get stage configuration from PipelineConfig
        schema_name = PipelineConfig.get_stage_schema(stage_name)
        prompt_func_name = PipelineConfig.get_stage_prompt_func(stage_name)
        
        # Import the required schema and prompt function dynamically
        import prompts
        schema = getattr(prompts, schema_name) if schema_name else None
        prompt_func = getattr(prompts, prompt_func_name)
        
        # Execute stage with whiteboard context
        response = pipeline.execute_stage(
            stage_name=stage_name,
            paper_content=paper_content,
            prompt_func=prompt_func,
            schema=schema
        )
        print_response(response)
        
        # Track responses for compatibility
        responses.append(response)
        
        # Parse and save structured data for compatibility
        response_content = response['choices'][0]['message']['content']
        structured_data = parse_structured_response(response_content)
        
        # Save traditional artifacts for compatibility
        artifact_manager.save_response(stage_name, response)
        artifact_manager.save_structured_data(stage_name, structured_data)
        
        # Save YAML config if this is the config stage
        if stage_name == 'config':
            artifact_manager.save_config_yaml(structured_data['config_yaml'])
        
        # Extract key data to whiteboard for coding phase - FIXED
        if stage_name == 'task_list':
            # Store task list in whiteboard for easy access
            task_list = structured_data.get('task_list', [])
            logic_analysis = structured_data.get('logic_analysis', [])
            updates = [
                f"knowledge.task_list={json.dumps(task_list)}",
                f"knowledge.logic_analysis={json.dumps(logic_analysis)}"
            ]
            whiteboard_manager.apply_updates(updates)
    
    # Save traditional artifacts for compatibility
    artifact_manager.save_trajectories(responses)
    artifact_manager.save_model_config(pipeline.reasoning_model, pipeline.coding_model)
    
    # Save combined structured responses for compatibility
    structured_responses = {}
    for response in responses:
        stage_name = response.get('stage', 'unknown')
        content = response['choices'][0]['message']['content']
        structured_responses[stage_name] = parse_structured_response(content)
    
    with open(f"{artifact_manager.output_dir}/all_structured_responses.json", 'w') as f:
        json.dump(structured_responses, f, indent=2)
    
    print("\n✅ Whiteboard-based planning phase completed successfully!")
    
    return whiteboard_manager.load_whiteboard()
    
def run_analysis_phase(paper_content: str, whiteboard_manager: WhiteboardManager, 
                       pipeline: WhiteboardPipeline, artifact_manager: ArtifactManager) -> None:
    """Run analysis phase using whiteboard-driven approach"""
    
    print("\n" + "="*60)
    print("🔍 STARTING WHITEBOARD-BASED ANALYSIS PHASE")
    print("="*60)
    print("   File-by-file analysis with accumulated whiteboard context")
    
    # Extract task list from whiteboard
    whiteboard = whiteboard_manager.load_whiteboard()
    knowledge = whiteboard.get('knowledge', {})
    
    # Handle case where knowledge might be a JSON string
    if isinstance(knowledge, str):
        try:
            knowledge = json.loads(knowledge)
        except json.JSONDecodeError:
            knowledge = {}
    
    task_list_json = knowledge.get('task_list', '[]')
    if isinstance(task_list_json, str):
        task_list = json.loads(task_list_json)
    else:
        task_list = task_list_json
    
    # Get development order if available (preferred for analysis)
    dev_order_json = knowledge.get('development_order', '[]')
    if isinstance(dev_order_json, str):
        try:
            development_order = json.loads(dev_order_json)
        except json.JSONDecodeError:
            development_order = []
    else:
        development_order = dev_order_json if isinstance(dev_order_json, list) else []
    
    # Use development order if available, otherwise use task list
    files_to_analyze = development_order if development_order else task_list
    
    # Get logic analysis from whiteboard
    logic_analysis_json = knowledge.get('logic_analysis', '[]')
    if isinstance(logic_analysis_json, str):
        logic_analysis = json.loads(logic_analysis_json)
    else:
        logic_analysis = logic_analysis_json
    
    # Convert logic analysis to dict
    logic_analysis_dict = {}
    for item in logic_analysis:
        if isinstance(item, list) and len(item) >= 2:
            logic_analysis_dict[item[0]] = item[1]
    
    if not files_to_analyze:
        print("❌ No files to analyze found in whiteboard")
        return
    
    print(f"📋 Task list contains {len(task_list)} total files")
    if development_order:
        print(f"📋 Development order contains {len(development_order)} files (using this for analysis)")
    print(f"📋 Analyzing {len(files_to_analyze)} files:")
    
    # Check if analysis is already complete for the files we need
    if artifact_manager.check_analysis_completion(files_to_analyze):
        print(f"📂 Analysis phase already completed for all {len(files_to_analyze)} target files")
        print("   Skipping analysis phase...")
        return
    
    for i, filename in enumerate(files_to_analyze, 1):
        print(f"   {i}. {filename}")
    
    # Process each file using whiteboard context
    for todo_file_name in tqdm(files_to_analyze, desc="Analyzing files"):
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
        
        # Execute analysis using whiteboard context
        response = pipeline.execute_stage(
            stage_name='analysis',
            paper_content=paper_content,
            prompt_func=get_analysis_prompt,
            schema=ANALYSIS_SCHEMA,
            todo_file_name=todo_file_name,
            todo_file_desc=todo_file_desc
        )
        print_response(response)
        
        # Parse and save structured data
        analysis_data = parse_structured_response(response['choices'][0]['message']['content'])
        
        # Save artifacts for this file
        artifact_manager.save_analysis_response(todo_file_name, response)
        artifact_manager.save_analysis_structured(todo_file_name, analysis_data)
        
        # Update whiteboard with analysis completion - FIXED
        analysis_updates = [
            f"analysis.files.{safe_filename}.completed=true",
            f"analysis.files.{safe_filename}.core_functionality={analysis_data.get('core_functionality', '')[:100]}"
        ]
        whiteboard_manager.apply_updates(analysis_updates)
    
    print(f"\n✅ Whiteboard-based analysis phase completed for {len([f for f in files_to_analyze if f != 'config.yaml'])} files")
    
def run_file_organization_phase(whiteboard_manager: WhiteboardManager, 
                               pipeline: WhiteboardPipeline, 
                               artifact_manager: ArtifactManager) -> Dict[str, Any]:
    """Run file organization phase using whiteboard-driven approach"""
    
    print("\n" + "="*60)
    print("📁 STARTING WHITEBOARD-BASED FILE ORGANIZATION PHASE")
    print("="*60)
    print("   Organizing files using accumulated whiteboard knowledge")
    
    # Check if already completed
    file_org_file = os.path.join(artifact_manager.output_dir, "file_organization_structured.json")
    if os.path.exists(file_org_file):
        print("📂 File organization already completed, loading existing data...")
        with open(file_org_file, 'r') as f:
            return json.load(f)
    
    # Get task list from whiteboard
    whiteboard = whiteboard_manager.load_whiteboard()
    task_list_json = whiteboard.get('knowledge', {}).get('task_list', '[]')
    if isinstance(task_list_json, str):
        task_list = json.loads(task_list_json)
    else:
        task_list = task_list_json
    
    print(f"📋 Organizing {len([f for f in task_list if f != 'config.yaml'])} files by development order")
    
    # Execute file organization using whiteboard context
    response = pipeline.execute_stage(
        stage_name='file_organization',
        paper_content='',  # No paper content needed
        prompt_func=get_file_organization_prompt,
        schema=FILE_ORGANIZATION_SCHEMA
    )
    print_response(response)
    
    # Parse and save structured data
    file_org_data = parse_structured_response(response['choices'][0]['message']['content'])
    
    # Save artifacts
    artifact_manager.save_file_organization_response(response)
    artifact_manager.save_file_organization_structured(file_org_data)
    
    # Update whiteboard with file organization
    development_order = file_org_data.get('development_order', [])
    org_updates = [
        f"knowledge.development_order={json.dumps(development_order)}",
        "pipeline.file_organization.completed=true"
    ]
    whiteboard_manager.apply_updates(org_updates)
    
    # Display development order
    print(f"\n📋 Development Order ({len(development_order)} files):")
    for i, filename in enumerate(development_order, 1):
        print(f"   {i}. {filename}")
    
    print("\n✅ Whiteboard-based file organization phase completed!")
    
    return file_org_data

def run_file_classification_phase(whiteboard_manager: WhiteboardManager, 
                                 pipeline: WhiteboardPipeline, 
                                 artifact_manager: ArtifactManager) -> Dict[str, Any]:
    """Classify all functionality into 5 simple files before code generation"""
    
    print("\n" + "="*60)
    print("📁 STARTING 5-FILE CLASSIFICATION PHASE")
    print("="*60)
    print("   Simplifying to: imports.py, constants.py, functions.py, main.py")
    
    # Check if already completed
    classification_file = os.path.join(artifact_manager.output_dir, "file_classification.json")
    if os.path.exists(classification_file):
        print("📂 File classification already completed, loading existing data...")
        with open(classification_file, 'r') as f:
            return json.load(f)
    
    # Execute classification using whiteboard context
    response = pipeline.execute_stage(
        stage_name='file_classification',
        paper_content='',  # No paper content needed
        prompt_func=get_file_classification_prompt,
        schema=FILE_CLASSIFICATION_SCHEMA
    )
    
    # Parse and save structured data
    classification_data = parse_structured_response(response['choices'][0]['message']['content'])
    
    # Save artifacts
    with open(classification_file, 'w') as f:
        json.dump(classification_data, f, indent=2)
    
    # Update whiteboard with classification
    updates = [
        "pipeline.file_classification.completed=true",
        f"knowledge.simplified_files.imports={classification_data.get('imports_content', 'Standard library imports')}",
        f"knowledge.simplified_files.constants={classification_data.get('constants_content', 'Configuration constants')}",
        f"knowledge.simplified_files.functions={classification_data.get('functions_content', 'Core utility functions')}",
        f"knowledge.simplified_files.main={classification_data.get('main_content', 'Main execution flow')}"
    ]
    whiteboard_manager.apply_updates(updates)
    
    print(f"\n✅ 5-file classification completed!")
    print(f"📋 Files to generate:")
    print(f"   1. imports.py - {classification_data.get('imports_content', 'N/A')[:60]}...")
    print(f"   2. constants.py - {classification_data.get('constants_content', 'N/A')[:60]}...")
    print(f"   3. functions.py - {classification_data.get('functions_content', 'N/A')[:60]}...")
    print(f"   4. main.py - {classification_data.get('main_content', 'N/A')[:60]}...")
    
    return classification_data

# prompts.py - Add these new schema and prompt

def run_coding_phase(paper_content: str, output_dir: str, output_repo_dir: str,
                    api_client: APIClient, coding_model: str, max_parallel: int,
                    whiteboard_manager: WhiteboardManager,
                    reasoning_model: str,  # Add this parameter
                    development_order: List[str] = None, 
                    max_context_tokens: int = 128000) -> List[Dict[str, Any]]:
    """Run the simplified 5-file coding phase with whiteboard-driven context building"""
    
    print("\n" + "="*60)
    print("💻 SIMPLIFIED 5-FILE CODING PHASE")
    print("="*60)
    print("   Generating exactly 5 files: imports.py, constants.py, functions.py, classes.py, main.py")
    
    # Run file classification first if not already done
    artifact_manager = ArtifactManager(output_dir)
    classification_data = run_file_classification_phase(whiteboard_manager, 
                                                       WhiteboardPipeline(reasoning_model, coding_model, api_client, whiteboard_manager),
                                                       artifact_manager)
    
    # NEW: STEP 1 - Create file mapping from original PDR files to consolidated 5-file structure
    original_task_list = whiteboard_manager.load_whiteboard().get('knowledge', {}).get('task_list', [])
    if isinstance(original_task_list, str):
        import json
        try:
            original_task_list = json.loads(original_task_list)
        except json.JSONDecodeError:
            original_task_list = []
    
    from functions import create_file_mapping  # Import the new function
    file_mapping = create_file_mapping(classification_data, original_task_list)
    
    # Store mapping in whiteboard for correction phase
    mapping_updates = [f"translation.file_mapping.{k.replace('.', '_').replace('/', '_')}={v}" 
                      for k, v in file_mapping.items()]
    whiteboard_manager.apply_updates(mapping_updates)
    
    print(f"\n📋 File Translation Mapping:")
    for orig, target in file_mapping.items():
        print(f"   {orig} → {target}")
    
    # Check dependencies
    if not os.path.exists(f'{output_dir}/planning_config.yaml'):
        print("❌ No config file found. Run planning phase first.")
        return []
    
    # Load config
    with open(f'{output_dir}/planning_config.yaml') as f:
        config_yaml = f.read()
    
    # Fixed list of 5 files to generate (no more complex file organization)
    files_to_generate = [
        ("imports.py", classification_data.get('imports_content', 'Library imports and dependencies')),
        ("constants.py", classification_data.get('constants_content', 'Configuration values and hyperparameters')),
        ("functions.py", classification_data.get('functions_content', 'Utility functions only')),
        ("classes.py", classification_data.get('classes_content', 'Class definitions and methods')),
        ("main.py", classification_data.get('main_content', 'High-level execution flow'))
    ]
    
    print(f"   Generating exactly 5 files: imports.py, constants.py, functions.py, classes.py, main.py")
    for i, (filename, description) in enumerate(files_to_generate, 1):
        print(f"   {i}. {filename}")
        print(f"      └─ {description[:60]}{'...' if len(description) > 60 else ''}")
    
    # Initialize simplified coding pipeline
    coding_pipeline = CodingPipeline(
        api_client=api_client,
        coding_model=coding_model,
        output_dir=output_dir,
        output_repo_dir=output_repo_dir,
        whiteboard_manager=whiteboard_manager
    )
    
    # Build shared context with whiteboard
    shared_context = {
        'paper_content': paper_content,
        'config_yaml': config_yaml,
        'whiteboard_yaml': whiteboard_manager.get_whiteboard_yaml(),
        'classification_data': classification_data,
        'file_mapping': file_mapping  # Add file mapping to shared context
    }
    
    # Process files sequentially (no parallel complexity needed for 5 files)
    results = coding_pipeline.process_files_sequential(files_to_generate, shared_context)
    
    # NEW: STEP 2 - Translate imports in main.py after generation
    from functions import translate_imports_in_main  # Import the new function
    
    for result in results:
        if result['filename'] == 'main.py' and result['success']:
            print(f"\n🔄 Translating imports in main.py...")
            print(f"   Original imports reference: {', '.join(file_mapping.keys())}")
            print(f"   Translating to: {', '.join(set(file_mapping.values()))}")
            
            translated_code = translate_imports_in_main(result['code'], file_mapping)
            
            # Write the translated version back to repo
            with open(f"{output_repo_dir}/main.py", 'w') as f:
                f.write(translated_code)
            
            # Update result with translated code
            result['code'] = translated_code
            result['translation_applied'] = True
            print(f"   ✅ Import translation completed")
            
            # Show what was translated
            import re
            original_imports = re.findall(r'from (\w+) import', result.get('original_code', ''))
            translated_imports = re.findall(r'from (\w+) import', translated_code)
            if original_imports != translated_imports:
                print(f"   📝 Translated imports:")
                for orig, trans in zip(original_imports, translated_imports):
                    if orig != trans:
                        print(f"      from {orig} import → from {trans} import")
    
    # Copy config file to output repo
    shutil.copy(f'{output_dir}/planning_config.yaml', f'{output_repo_dir}/config.yaml')
    
    # Update whiteboard with coding completion and translation info
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    translated = [r for r in results if r.get('translation_applied', False)]
    
    coding_updates = [
        f"coding.simplified.completed=true",
        f"coding.simplified.successful_files={len(successful)}",
        f"coding.simplified.failed_files={len(failed)}",
        f"coding.simplified.translated_files={len(translated)}",
        f"coding.simplified.completion_time={time.time()}"
    ]
    whiteboard_manager.apply_updates(coding_updates)
    
    # Generate enhanced summary
    print(f"\n" + "="*60)
    print("✅ Simplified 5-file coding phase completed!")
    print(f"📊 Results: {len(successful)} successful, {len(failed)} failed")
    if translated:
        print(f"🔄 Import Translation: {len(translated)} files had imports translated")
    print(f"📁 Repository: {output_repo_dir}")
    print(f"🎯 Whiteboard state: {output_dir}/whiteboard.json")
    
    if successful:
        print(f"\n✅ Generated simplified files:")
        for result in successful:
            status_icons = ["✨"]
            if result.get('translation_applied', False):
                status_icons.append("🔄")
            if result.get('whiteboard_updates', 0) > 0:
                status_icons.append("📝")
            
            print(f"   {''.join(status_icons)} {result['filename']}")
            if 'content_summary' in result:
                print(f"     └─ {result['content_summary'][:80]}{'...' if len(result['content_summary']) > 80 else ''}")
    
    if failed:
        print(f"\n❌ Failed files:")
        for result in failed:
            print(f"   - {result['filename']}: {result.get('error', 'Unknown error')}")
    
    # Generate results summary with translation info
    results_summary = {
        'total_files': len(files_to_generate),
        'successful': len(successful),
        'failed': len(failed),
        'translated': len(translated),
        'simplified_approach': True,
        'files_generated': [r['filename'] for r in successful],
        'file_mapping': file_mapping,
        'translation_applied': len(translated) > 0,
        'results': results
    }
    
    with open(f"{output_dir}/coding_results.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save file mapping for future reference
    with open(f"{output_dir}/file_translation_mapping.json", 'w') as f:
        json.dump(file_mapping, f, indent=2)
    
    print(f"\n📋 File mapping saved to: {output_dir}/file_translation_mapping.json")
    
    return results

def run_iterative_refinement_phase(
    paper_content: str,  # Add this
    output_dir: str, 
    output_repo_dir: str,
    api_client: APIClient, 
    reasoning_model: str,
    whiteboard_manager: WhiteboardManager
) -> Dict[str, Any]:
    """Execute snapshot-based iterative refinement with proper comparison."""
    
    # PHASE 1: Initial gap analysis and snapshot
    print("\n" + "="*60)
    print("🔍 INITIAL GAP ANALYSIS & SNAPSHOT")
    print("="*60)
    
    current_code = load_generated_code_files(output_repo_dir)
    initial_response = api_client.chat_completion(
        model=reasoning_model,
        messages=get_gap_analysis_prompt(
            current_code_files=current_code,
            whiteboard_yaml=whiteboard_manager.get_whiteboard_yaml()
        ),
        response_format=GAP_ANALYSIS_SCHEMA,
        stream=True
    )
    
    initial_gap_data = parse_structured_response(initial_response['choices'][0]['message']['content'])
    
    # Create snapshot of specific deficiency lists
    initial_snapshot = initial_gap_data['undefined_items']
    initial_total = sum(len(items) for items in initial_snapshot.values())
    
    # Save snapshot to whiteboard
    whiteboard_manager.apply_updates([
        f"refinement.initial_snapshot={json.dumps(initial_snapshot)}",
        f"refinement.initial_total={initial_total}"
    ])
    
    print(f"📸 Snapshot created: {initial_total} total deficiencies")
    for category, items in initial_snapshot.items():
        if items:
            print(f"   - {category}: {len(items)} items")

    if initial_total == 0:
        print("✅ No gaps found - code appears complete!")
        return {"success": True, "items_resolved": 0, "resolution_rate": 1.0}

    # PHASE 2: Category Implementation (stays the same)
    print("\n" + "="*60)
    print("🔧 IMPLEMENTATION PHASE")
    print("="*60)
    
    categories = {
        'functions': 'functions.py',
        'classes': 'classes.py', 
        'constants': 'constants.py',
        'imports': 'imports.py',
        'main': 'main.py',
        'config': 'config.yaml'
    }
    
    items_implemented = 0
    
    for category, target_file in categories.items():
        corrected_category = f"corrected_{category}"
        items = initial_snapshot.get(corrected_category, [])
        
        if not items:
            print(f"⏭️ Skipping {category} - no items to implement") 
            continue
            
        print(f"\n🛠️ Implementing {len(items)} {category} items...")
        
        # Generate corrections for this category
        response = api_client.chat_completion(
            model=reasoning_model,
            messages=get_category_implementation_prompt(
                category=corrected_category,
                items_list=items,
                whiteboard_yaml=whiteboard_manager.get_whiteboard_yaml(),
                paper_content=paper_content if paper_content else ""  # Trim for context window
            ),
            response_format=CATEGORY_IMPLEMENTATION_SCHEMA,
            stream=True
        )
        
        result = parse_structured_response(response['choices'][0]['message']['content'])
        implemented = len(result.get('items_completed', []))
        items_implemented += implemented
        
        # Save corrections to corrected_* file
        corrected_path = f"{output_repo_dir}/corrected_{target_file}"
        with open(corrected_path, 'w') as f:
            f.write(result['implementation'])
            
        print(f"✅ Saved {implemented} implementations to corrected_{target_file}")

    # PHASE 3: Direct Snapshot Comparison  
    print("\n" + "="*60)
    print("📊 SNAPSHOT COMPARISON")
    print("="*60)
    
    # Load corrected files
    corrected_files = load_generated_code_files(output_repo_dir, prefix="corrected_")
    
    # Ask LLM to directly compare against original snapshot
    comparison_prompt = f"""
    ## Original Deficiency Snapshot
    {json.dumps(initial_snapshot, indent=2)}
    
    ## Corrected Files  
    {corrected_files}
    
    ## Task
    Compare the corrected files against the original deficiency snapshot.
    Return ONLY the remaining deficiencies that were NOT properly addressed in the corrected files.
    
    Use the same category structure as the original snapshot.
    """
    
    comparison_response = api_client.chat_completion(
        model=reasoning_model,
        messages=[
            {"role": "system", "content": "You are analyzing code corrections against a deficiency snapshot."},
            {"role": "user", "content": comparison_prompt}
        ],
        response_format=GAP_ANALYSIS_SCHEMA,
        stream=True
    )
    
    final_gap_data = parse_structured_response(comparison_response['choices'][0]['message']['content'])
    remaining_deficiencies = final_gap_data['undefined_items']
    remaining_total = sum(len(items) for items in remaining_deficiencies.values())
    
    # Simple arithmetic
    items_resolved = initial_total - remaining_total
    resolution_rate = items_resolved / initial_total if initial_total > 0 else 1.0
    
    print(f"\n🎯 Refinement Results:")
    print(f" - Initial deficiencies: {initial_total}")
    print(f" - Items implemented: {items_implemented}")
    print(f" - Items resolved: {items_resolved}")
    print(f" - Remaining deficiencies: {remaining_total}")
    print(f" - Resolution rate: {resolution_rate:.1%}")
    
    # Show category breakdown
    print(f"\n📋 Category Breakdown:")
    for category in initial_snapshot.keys():
        initial_count = len(initial_snapshot.get(category, []))
        remaining_count = len(remaining_deficiencies.get(category, []))
        resolved_count = initial_count - remaining_count
        
        if initial_count > 0:
            print(f"   - {category}: {resolved_count}/{initial_count} resolved ({resolved_count/initial_count:.1%})")
    
    return {
        "success": True,
        "initial_total": initial_total,
        "items_implemented": items_implemented,
        "items_resolved": items_resolved,
        "remaining_total": remaining_total,
        "resolution_rate": resolution_rate,
        "initial_snapshot": initial_snapshot,
        "remaining_deficiencies": remaining_deficiencies,
        "files_updated": [f for f in os.listdir(output_repo_dir) if f.startswith('corrected_')]
    }

def main():
    """Main execution function with whiteboard-based pipeline"""
    
    # Parse arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Check if running AutoGen validation only
    if args.autogen_validation_only:
        print(f"\n🤖 Running AutoGen Validation Only")
        print(f"📁 Output directory: {args.output_dir}")
        print(f"📁 Repository directory: {args.output_repo_dir}")
        
        # Validate prerequisites
        if not validate_autogen_prerequisites(args.output_dir, args.output_repo_dir):
            return
        
        # Initialize API client and whiteboard
        api_client = APIClient(base_url=args.api_base_url, api_key=args.api_key, 
                              initial_seed=args.seed, default_timeout=args.timeout)
        whiteboard_manager = WhiteboardManager(args.output_dir)
        
        # Load paper content for context
        print(f"📄 Loading paper from: {args.paper_markdown_path}")
        paper_content = load_paper_content(args.paper_markdown_path)
        paper_requirements = paper_content[:2000]
        
        # Run AutoGen validation with whiteboard tracking
        try:
            from functions import run_autogen_validation_phase
            
            validation_result = run_autogen_validation_phase(
                output_repo_dir=args.output_repo_dir,
                output_dir=args.output_dir,
                api_client=api_client,
                coding_model=args.coding_model,
                paper_requirements=paper_requirements
            )
            
            if validation_result['success']:
                print("\n🎉 AutoGen validation completed successfully!")
                print("✅ Your application is now executable and validated!")
            else:
                print(f"\n⚠️  AutoGen validation failed: {validation_result.get('error', 'Unknown error')}")
                print("🔧 Check validation artifacts for details")
        
        except ImportError:
            print("❌ AutoGen validation module not found")
        except Exception as e:
            print(f"❌ AutoGen validation error: {e}")
        
        return
    
    # Initialize whiteboard manager
    whiteboard_manager = WhiteboardManager(args.output_dir)
    
    # Clear whiteboard if requested
    if args.clear_whiteboard:
        print("🗑️  Clearing whiteboard state...")
        whiteboard_manager.save_whiteboard({})
    
    # Check pipeline state
    pipeline_state = check_pipeline_state(args.output_dir)
    
    print(f"\n🚀 Starting whiteboard-based pipeline for: {args.paper_name}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🎯 Whiteboard file: {args.output_dir}/whiteboard.json")
    print(f"📊 Pipeline state:")
    print(f"   Planning: {'✅' if pipeline_state['planning_complete'] else '❌'}")
    print(f"   Analysis: {'✅' if pipeline_state['analysis_complete'] else '❌'}")  
    print(f"   File Org: {'✅' if pipeline_state['file_organization_complete'] else '❌'}")
    print(f"   Coding:   {'✅' if pipeline_state.get('coding_complete', False) else '🔄' if pipeline_state['coding_started'] else '❌'}")
    print(f"   Refinement: {'✅' if pipeline_state.get('refinement_complete', False) else '❌'}")
    print(f"   AutoGen:  {'✅' if pipeline_state.get('autogen_validation_complete', False) else '❌'}")
    
    # Enhanced resume logic with iterative refinement support
    skip_to_refinement = False
    skip_to_coding = False
    
    # Check for explicit refinement resume
    if getattr(args, 'resume_from_refinement', False):
        if not pipeline_state.get('coding_complete', False):
            print("❌ Cannot resume from refinement: Coding not complete")
            return
        print("🔄 Resuming from iterative refinement phase...")
        skip_to_refinement = True
    elif hasattr(args, 'resume_from_coding') and args.resume_from_coding:
        if not pipeline_state['planning_complete']:
            print("❌ Cannot resume from coding: Planning not complete")
            return
        if not pipeline_state['analysis_complete']:
            print("❌ Cannot resume from coding: Analysis not complete")
            return
        print("🔄 Resuming from coding phase...")
        skip_to_coding = True
    elif getattr(args, 'resume_from_analysis', False):
        if not pipeline_state['planning_complete']:
            print("❌ Cannot resume from analysis: Planning not complete")
            return
        print("🔄 Resuming from analysis phase...")
        skip_to_coding = False
    else:
        # Auto-detect resume point based on pipeline state
        if (pipeline_state.get('coding_complete', False) and 
            getattr(args, 'enable_iterative_refinement', False) and 
            not pipeline_state.get('refinement_complete', False)):
            print("🔄 Auto-resume: Coding complete, jumping to iterative refinement...")
            skip_to_refinement = True
        elif pipeline_state['planning_complete'] and pipeline_state['analysis_complete']:
            print("🔄 Auto-resume: Planning and analysis complete, resuming from coding...")
            skip_to_coding = True
        elif pipeline_state['planning_complete']:
            print("🔄 Auto-resume: Planning complete, resuming from analysis...")
            skip_to_coding = False
        else:
            print("🆕 Starting fresh whiteboard-based pipeline...")
            skip_to_coding = False
    
    # Initialize components
    api_client = APIClient(base_url=args.api_base_url, api_key=args.api_key, 
                          initial_seed=args.seed, default_timeout=args.timeout)
    pipeline = WhiteboardPipeline(args.reasoning_model, args.coding_model, api_client, whiteboard_manager)
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
    if skip_to_coding or skip_to_refinement:
        # Load existing whiteboard state
        whiteboard = whiteboard_manager.load_whiteboard()
        print(f"📂 Loaded existing whiteboard state")
        
        # Load or run file organization
        if pipeline_state['file_organization_complete']:
            with open(f"{args.output_dir}/file_organization_structured.json", 'r') as f:
                file_org_data = json.load(f)
            print("📂 Loaded existing file organization")
        else:
            print("🔄 Running file organization phase...")
            file_org_data = run_file_organization_phase(whiteboard_manager, pipeline, artifact_manager)
        
    else:
        # Run planning phase if needed
        if not pipeline_state['planning_complete']:
            whiteboard = run_planning_phase(paper_content, pipeline, artifact_manager, whiteboard_manager)
        else:
            whiteboard = whiteboard_manager.load_whiteboard()
            print(f"📂 Loaded existing whiteboard state from planning")
        
        # Run analysis phase if needed
        if not pipeline_state['analysis_complete']:
            run_analysis_phase(paper_content, whiteboard_manager, pipeline, artifact_manager)
        else:
            print("📂 Analysis already complete, skipping...")
        
        # Run file organization phase
        if pipeline_state['file_organization_complete']:
            with open(f"{args.output_dir}/file_organization_structured.json", 'r') as f:
                file_org_data = json.load(f)
            print("📂 Loaded existing file organization")
        else:
            file_org_data = run_file_organization_phase(whiteboard_manager, pipeline, artifact_manager)
    
    # Run coding phase with whiteboard integration (only if not already complete and not skipping to refinement)
    if not pipeline_state.get('coding_complete', False) and not skip_to_refinement:
        print(f"\n{'='*60}")
        print("💻 WHITEBOARD-DRIVEN CODING PHASE")
        print(f"{'='*60}")
        
        # Optional: Force regenerate
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
                whiteboard_manager=whiteboard_manager,
                reasoning_model=args.reasoning_model,
                development_order=file_org_data.get('development_order', [])
            )
            
            # Enhanced final summary
            successful = [r for r in results if r['success']]
            failed = [r for r in results if not r['success']]
            resumed = [r for r in results if r.get('resumed', False)]
            total_whiteboard_updates = sum(r.get('whiteboard_updates', 0) for r in successful)
            
            print(f"\n" + "="*60)
            print("✅ Complete whiteboard-based coding phase finished!")
            print(f"📊 Coding Results:")
            print(f"   Total files: {len(results)}")
            print(f"   Successful: {len(successful)}")
            print(f"   Failed: {len(failed)}")
            if resumed:
                print(f"   Resumed: {len(resumed)}")
            print(f"   Whiteboard updates: {total_whiteboard_updates}")
            print(f"📁 All artifacts saved to: {args.output_dir}")
            print(f"🎯 Whiteboard state: {args.output_dir}/whiteboard.json")
            
            if successful:
                print(f"\n✅ Generated files:")
                for result in successful:
                    status = "📂" if result.get('resumed', False) else "✨"
                    print(f"   {status} {result['filename']}")
            
            if failed:
                print(f"\n❌ Failed files:")
                for result in failed:
                    print(f"   - {result['filename']}: {result.get('error', 'Unknown error')}")
            
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Pipeline interrupted by user (Ctrl-C)")
            print(f"💡 To resume where you left off, run the exact same command:")
            print(f"   python main.py --paper_name {args.paper_name} --paper_markdown_path {args.paper_markdown_path} --output_dir {args.output_dir} --output_repo_dir {args.output_repo_dir}")
            print(f"   The whiteboard system will automatically detect completed work and skip it.")
            print(f"🎯 Current whiteboard state preserved in: {args.output_dir}/whiteboard.json")
            return
            
    elif skip_to_refinement:
        print(f"\n📂 Coding phase already completed, jumping to refinement...")
    else:
        print(f"\n📂 Coding phase already completed, skipping...")
    
    # Run iterative refinement phase (CONSOLIDATED - handles both skip_to_refinement and enable_iterative_refinement)
    if ((getattr(args, 'enable_iterative_refinement', False) or skip_to_refinement) and 
        not pipeline_state.get('refinement_complete', False)):
        
        print(f"\n{'='*60}")
        print("🔧 ITERATIVE REFINEMENT PHASE")
        print(f"{'='*60}")
        
        try:
            refinement_result = run_iterative_refinement_phase(
                paper_content=paper_content,  # Pass the paper content you already loaded
                output_dir=args.output_dir,
                output_repo_dir=args.output_repo_dir,
                api_client=api_client,
                reasoning_model=args.reasoning_model,
                whiteboard_manager=whiteboard_manager
            )
                                    
            if refinement_result['success']:
                resolution_rate = refinement_result.get('resolution_rate', 0.0)
                print(f"🎯 Refinement achieved {resolution_rate:.1%} resolution rate")
                print(f"   📊 {refinement_result.get('items_resolved', 0)}/{refinement_result.get('initial_total', 0)} deficiencies resolved")
                
                # Update pipeline state to mark refinement as complete
                whiteboard_manager.apply_updates([
                    "pipeline.refinement.completed=true",
                    f"pipeline.refinement.resolution_rate={resolution_rate}",
                    f"pipeline.refinement.completion_time={time.time()}"
                ])
            else:
                print(f"❌ Refinement failed: {refinement_result.get('error', 'Unknown error')}")
        
        except Exception as e:
            print(f"❌ Iterative refinement error: {e}")
    
    elif pipeline_state.get('refinement_complete', False):
        print(f"\n📂 Iterative refinement already completed")
    
    # AutoGen validation with whiteboard tracking
    if (getattr(args, 'enable_autogen_validation', False) and 
        not pipeline_state.get('autogen_validation_complete', False)):
        
        print(f"\n{'='*60}")
        print("🤖 AUTOGEN VALIDATION PHASE")
        print(f"{'='*60}")
        
        if not pipeline_state.get('coding_complete', False):
            print("⚠️  Skipping AutoGen validation - no successful code generation")
        else:
            try:
                from functions import run_autogen_validation_phase
                
                # Load paper content for context
                paper_requirements = paper_content[:2000]
                
                validation_result = run_autogen_validation_phase(
                    output_repo_dir=args.output_repo_dir,
                    output_dir=args.output_dir,
                    api_client=api_client,
                    coding_model=args.coding_model,
                    paper_requirements=paper_requirements
                )
                
                if validation_result['success']:
                    print("🎉 AutoGen validation completed - application is executable!")
                    # Update whiteboard with validation completion
                    whiteboard_manager.apply_updates([
                        "validation.autogen.completed=true",
                        f"validation.autogen.completion_time={time.time()}"
                    ])
                else:
                    print("⚠️  AutoGen validation had issues - check artifacts")
                    
            except ImportError:
                print("❌ AutoGen validation module not available")
            except Exception as e:
                print(f"❌ AutoGen validation error: {e}")
    
    elif pipeline_state.get('autogen_validation_complete', False):
        print(f"\n📂 AutoGen validation already completed")
    
    # Final pipeline summary
    print(f"\n{'='*60}")
    print("🎉 WHITEBOARD-BASED PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"💡 Whiteboard advantages:")
    print(f"   - Persistent knowledge accumulation across stages")
    print(f"   - Dynamic context building without hardcoded dependencies")
    print(f"   - Resume capability with complete state preservation")
    print(f"   - Incremental updates prevent information loss")
    print(f"   - TRIZ-based iterative refinement for code quality")
    print(f"\n📁 All artifacts saved to: {args.output_dir}")
    print(f"🎯 Final whiteboard state: {args.output_dir}/whiteboard.json")
    
    if pipeline_state.get('refinement_complete', False):
        print(f"📈 Code refinement: Completed")
        refined_markdown = f"{args.output_dir}/kumorfm_refined_implementation.md"
        if os.path.exists(refined_markdown):
            print(f"📄 Refined code: {refined_markdown}")
    
    if pipeline_state.get('coding_complete', False):
        print(f"💻 Generated repository: {args.output_repo_dir}")
        
    print(f"\n🚀 {args.paper_name} implementation pipeline complete!")

if __name__ == "__main__":
    main()