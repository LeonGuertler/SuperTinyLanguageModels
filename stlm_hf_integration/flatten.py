"""BAD CODE lol
https://huggingface.co/docs/transformers/en/custom_models#using-a-model-with-custom-code"""

import os
import re

import networkx as nx
from networkx.algorithms.dag import topological_sort


def collect_python_files(directory, ignore_dirs):
    """Recursively collect all Python files in the given directory, excluding specified directories."""
    python_files = set()
    for root, _, files in os.walk(directory):
        if any(ignore_dir in root for ignore_dir in ignore_dirs):
            continue
        for file in files:
            if file.endswith(".py"):
                # relative_path = os.path.relpath(os.path.join(root, file), directory)
                # module_name = relative_path.replace(os.sep, ".")[
                #     :-3
                # ]  # Remove .py extension
                python_files.add(os.path.join(root, file))
    return python_files


def get_module_name(file_path, base_directory):
    """Get the module name from the file path relative to the base directory."""
    relative_path = os.path.relpath(file_path, base_directory)
    module_name = relative_path.replace(os.sep, ".")[:-3]  # Remove .py extension
    return module_name


def extract_imported_modules(content):
    """Extract imported module names from the file content."""
    imports = re.findall(r"(?:from\s+(\S+)\s+import\s+|\bimport\s+)(\S+)", content)
    modules = set()
    for imp in imports:
        right_imports = imp[1].split(",")
        for right_import in right_imports:
            imported_module = f"{imp[0]}.{right_import}" if imp[0] else right_import
            # Split by '.' to handle submodules and handle direct imports
            imported_modules = imported_module.split(".")
            for i in range(len(imported_modules)):
                module = ".".join(imported_modules[: i + 1])
                modules.add(module)
    return modules


def build_dependency_graph(file_paths, base_directory):
    """Build a dependency graph from the collected Python file paths."""
    graph = nx.DiGraph()

    # Map file paths to module names
    module_map = {
        file_path: get_module_name(file_path, base_directory)
        for file_path in file_paths
    }

    # Add nodes to the graph with their corresponding file paths
    for file_path, module_name in module_map.items():
        graph.add_node(file_path, module_name=module_name)
        with open(file_path, "r") as file:
            content = file.read()

        # Extract all imported modules
        imported_modules = extract_imported_modules(content)

        # Process each import
        for imported_module in imported_modules:
            if imported_module in module_map.values():
                # Find the file path of the imported module
                imported_file_path = next(
                    path for path, name in module_map.items() if name == imported_module
                )
                # Add edge from imported file to current file
                graph.add_edge(imported_file_path, file_path)
    return graph


def process_imports(file_content, imported_model_modules, all_imports):
    """Adjust import statements to work in a single-file context and inline content."""
    lines = file_content.split("\n")
    processed_lines = []

    in_multiline_import = False
    multiline_import = []

    for line in lines:
        stripped_line = line.strip()
        if in_multiline_import:
            multiline_import.append(line)
            if stripped_line.endswith(")"):
                full_import = "\n".join(multiline_import)
                in_multiline_import = False
                if full_import.startswith("from models") or full_import.startswith(
                    "import models"
                ):
                    continue
                all_imports.add(full_import)
            continue
        elif stripped_line.startswith(("from ", "import ")):
            if stripped_line.endswith("\\") or stripped_line.endswith("("):
                in_multiline_import = True
                multiline_import = [line]
                continue
            if stripped_line.startswith("from models") or stripped_line.startswith(
                "import models"
            ):
                continue
            all_imports.add(line)
            continue
        if "core_models" not in imported_model_modules:
            print(imported_model_modules)
            pass
        line = simplify_reference(line, imported_model_modules)
        processed_lines.append(line)

    return "\n".join(processed_lines)


def flatten_models_code(
    base_directory, output_file, ignore_dirs, include_subfolders=[]
):
    """Flatten the models code into a single file, excluding specified directories."""
    ignore_dirs = [os.path.join(base_directory, d) for d in ignore_dirs]
    include_subfolders = [os.path.join(base_directory, d) for d in include_subfolders]
    python_files = collect_python_files(base_directory, ignore_dirs)
    for include_subfolder in include_subfolders:
        include_files = collect_python_files(include_subfolder, [])
        python_files.update(include_files)
    dep_graph = build_dependency_graph(python_files, "")
    python_files = list(topological_sort(dep_graph))
    print(python_files)

    all_imports = set()
    # collect references to models in the imported modules
    imported_models_modules = []
    for file_path in python_files:
        imported_models_modules.extend(collect_models_imported_modules(file_path))

    # First pass: collect imports and file contents
    file_contents = []
    processed_files = set()
    for file_path in python_files:
        if file_path not in processed_files:
            with open(file_path, "r") as infile:
                file_content = infile.read()
                processed_files.add(file_path)
                processed_content = process_imports(
                    file_content,
                    imported_models_modules,
                    all_imports,
                )
                file_contents.append(f"# {file_path}\n{processed_content}\n\n")

    # These imports: from trainers.base_trainer import BaseTrainer
    # from trainers.dataloader import BaseDataloader
    # from trainers.utils import load_data
    # all need to be replaced with mock functions to make the code run, as they are not present in the models directory,
    # but also won't be called anyhow... (hopefully) (if it gets to that point maybe come up with a method that isn't flattening
    # our code base into a single file...) (or just remove them from the output file manually)
    all_imports.add("class BaseTrainer: pass")
    all_imports.add("class BaseDataloader: pass")
    all_imports.add("def load_data(): pass")
    all_imports.add("def to_absolute_path(*args, **kwargs): pass")
    all_imports.remove("from hydra.utils import to_absolute_path")
    all_imports.remove("from trainers.base_trainer import BaseTrainer")
    all_imports.remove("from trainers.dataloader import BaseDataloader")
    all_imports.remove("from trainers.utils import load_data")

    # Second pass: write imports and contents to output file
    with open(output_file, "w") as outfile:
        if all_imports:
            outfile.write("\n".join(sorted(all_imports)) + "\n\n")
        for content in file_contents:
            outfile.write(content)


def collect_models_imported_modules(file_path):
    """Collect the modules imported from models in a given file."""
    with open(file_path, "r") as infile:
        lines = infile.readlines()
    imported_models_modules = []
    for line in lines:
        if line.startswith("from models") or line.startswith("import models"):
            match = re.search(r"from models(.*) import (.+)", line)
            if match:
                imported_models_modules.append(match.group(2))
            match = re.search(r"import models(.*)", line)
            if match:
                imported_models_modules.append(match.group(1))
            for module in imported_models_modules:
                if "," in module:
                    imported_models_modules.extend(module.split(", "))
    return imported_models_modules


def simplify_reference(line, imported_models_modules):
    """If the line references something imported from models, simplify it."""
    # Create a regex pattern to match fully qualified names of imported modules
    pattern = (
        r"\b("
        + "|".join(re.escape(module) for module in imported_models_modules)
        + r")\.(\w+)\b"
    )

    def replacement(match):
        # The match group 2 is the class or function name
        return match.group(2)

    # Replace fully qualified names in the line with just the class/function names
    simplified_line = re.sub(pattern, replacement, line)

    return simplified_line


# Specify the models directory and the output file
models_directory = "models"
output_file = "stlm_hf_integration/stlm_modelling.py"

# ignore_dirs = ["experimental", "build_"]  # List of directories to ignore
ignore_dirs = []  # use this arg at your peril.. imports *will* break
include_subfolders = ["experimental/next_thought"]  # List of subfolders to include

flatten_models_code(models_directory, output_file, ignore_dirs, include_subfolders)

print(f"Flattened models code has been written to {output_file}")
