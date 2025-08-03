#!/usr/bin/env python3
"""
Comprehensive cleanup script for the whiteboard-based pipeline
Removes all generated artifacts and resets to clean state
"""

import os
import sys
import shutil
import glob
from pathlib import Path

def clean_pipeline(output_dir: str = "output", repo_dir: str = "repos"):
    """Clean all pipeline artifacts"""
    
    print("🗑️  CLEANING PIPELINE ARTIFACTS")
    print("=" * 50)
    
    # Clean output directories
    output_dirs = [f for f in glob.glob(f"{output_dir}/*") if os.path.isdir(f)]
    print(f"Found {len(output_dirs)} output directories to clean:")
    
    for output_path in output_dirs:
        project_name = os.path.basename(output_path)
        print(f"\n📁 Cleaning {project_name}...")
        
        # Remove all files but preserve directory structure
        try:
            for item in os.listdir(output_path):
                item_path = os.path.join(output_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                    print(f"   📄 Deleted file: {item}")
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"   📁 Deleted directory: {item}")
        except Exception as e:
            print(f"   ⚠️  Error cleaning {output_path}: {e}")
    
    # Clean repository directories  
    repo_dirs = [f for f in glob.glob(f"{repo_dir}/*") if os.path.isdir(f)]
    print(f"\nFound {len(repo_dirs)} repository directories to clean:")
    
    for repo_path in repo_dirs:
        project_name = os.path.basename(repo_path)
        print(f"\n📂 Cleaning repository {project_name}...")
        
        try:
            # Remove all files but preserve directory
            for item in os.listdir(repo_path):
                item_path = os.path.join(repo_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                    print(f"   📄 Deleted: {item}")
                # Don't remove directories to preserve structure
        except Exception as e:
            print(f"   ⚠️  Error cleaning {repo_path}: {e}")
    
    # Remove any lingering analysis files that might be in wrong places
    print("\n🔍 Removing stray analysis files...")
    stray_patterns = [
        "*_analysis_*.json",
        "*_simple_analysis_*.json", 
        "*/.*_analysis_*.json",  # Hidden files
        "*_response.json",
        "*_structured.json"
    ]
    
    for pattern in stray_patterns:
        files = glob.glob(f"{output_dir}/**/{pattern}", recursive=True)
        for file in files:
            try:
                os.remove(file)
                print(f"   🗑️  Removed stray file: {file}")
            except Exception as e:
                print(f"   ⚠️  Error removing {file}: {e}")
    
    print("\n✅ Pipeline cleanup complete!")
    print("   Use --clear_whiteboard flag for fresh start next run")

def clean_specific_project(project_name: str, output_dir: str = "output", repo_dir: str = "repos"):
    """Clean specific project artifacts"""
    
    print(f"🗑️  CLEANING PROJECT: {project_name}")
    print("=" * 50)
    
    # Clean specific output directory
    project_output = os.path.join(output_dir, project_name)
    if os.path.exists(project_output):
        print(f"📁 Cleaning output: {project_output}")
        try:
            for item in os.listdir(project_output):
                item_path = os.path.join(project_output, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            print("   ✅ Output directory cleaned")
        except Exception as e:
            print(f"   ⚠️  Error cleaning output: {e}")
    
    # Clean specific repo directory
    project_repo = os.path.join(repo_dir, project_name)
    if os.path.exists(project_repo):
        print(f"📂 Cleaning repository: {project_repo}")
        try:
            for item in os.listdir(project_repo):
                item_path = os.path.join(project_repo, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
            print("   ✅ Repository directory cleaned")
        except Exception as e:
            print(f"   ⚠️  Error cleaning repository: {e}")

def reset_whiteboard(output_dir: str = "output"):
    """Reset all whiteboard states"""
    
    print("🔄 RESETTING WHITEBOARD STATES")
    print("=" * 50)
    
    whiteboard_files = glob.glob(f"{output_dir}/*/whiteboard.json")
    print(f"Found {len(whiteboard_files)} whiteboard files to reset")
    
    for wb_file in whiteboard_files:
        try:
            os.remove(wb_file)
            print(f"   🗑️  Removed: {wb_file}")
        except Exception as e:
            print(f"   ⚠️  Error removing {wb_file}: {e}")
    
    print("✅ Whiteboard reset complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean pipeline artifacts")
    parser.add_argument("--project", help="Clean specific project only")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--repo-dir", default="repos", help="Repository directory") 
    parser.add_argument("--reset-whiteboard", action="store_true", help="Also reset whiteboard files")
    parser.add_argument("--all", action="store_true", help="Clean everything including whiteboards")
    
    args = parser.parse_args()
    
    if args.all:
        args.reset_whiteboard = True
        
    try:
        if args.project:
            clean_specific_project(args.project, args.output_dir, args.repo_dir)
        else:
            clean_pipeline(args.output_dir, args.repo_dir)
            
        if args.reset_whiteboard:
            reset_whiteboard(args.output_dir)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Cleanup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Cleanup failed: {e}")
        sys.exit(1)
        
    print("\n🎉 Cleanup completed successfully!")
