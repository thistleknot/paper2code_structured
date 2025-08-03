#!/bin/bash

# Files to regenerate (the problematic ones with repetition issues)
FILES_TO_DELETE=(
    "kumo/eval/reporting.py"
    "kumo/api/server.py"
    # Add other problematic files here as needed
)

# Delete final generated files
for file in "${FILES_TO_DELETE[@]}"; do
    echo "Deleting final file: repos/kumo/$file"
    rm -f "repos/kumo/$file"
done

# Delete related artifacts for each file
for file in "${FILES_TO_DELETE[@]}"; do
    # Convert filepath to safe filename (replace / with _)
    safe_name="${file//\//_}"
    
    echo "Deleting artifacts for: $file"
    
    # Delete coding artifacts
    rm -f "output/kumo/coding_artifacts/${safe_name}_coding.txt"
    rm -f "output/kumo/coding_artifacts/${safe_name}_deliberation.txt"
    
    # Delete structured responses
    rm -f "output/kumo/structured_code_responses/${safe_name}_structured.json"
    
    # Delete diffs
    rm -f "output/kumo/diffs/${safe_name}.diff"
done

echo "Cleanup complete. Rerun the pipeline to regenerate these files."
