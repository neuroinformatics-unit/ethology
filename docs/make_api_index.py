"""Generate the API index page for all ``ethology`` modules."""

import os
from pathlib import Path

# Modules to exclude from the API index
exclude_modules = ["ethology.validators.json_schemas"]

# Set the current working directory to the directory of this script
script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)


def make_api_index():
    """Create a doctree of all ``ethology`` modules."""
    doctree = "\n"
    api_path = Path("../ethology")
    for path in sorted(api_path.rglob("*.py")):
        if path.name.startswith("_"):
            continue

        # Convert file path to module name
        rel_path = path.relative_to(api_path.parent)
        module_name = str(rel_path.with_suffix("")).replace(os.sep, ".")

        # Split the module name into parts
        # (e.g., ["ethology", "annotations", ...])
        module_parts = module_name.split(".")

        # Check against each excluded module's path
        exclude = False
        for excluded in exclude_modules:
            excluded_parts = excluded.split(".")

            # Only exclude if the module is a submodule of the excluded path
            if (
                len(module_parts) >= len(excluded_parts)
                and module_parts[: len(excluded_parts)] == excluded_parts
            ):
                exclude = True
                break
        if exclude:
            continue
        doctree += f"    {module_name}\n"

    # Get the header
    api_head_path = Path("source") / "_templates" / "api_index_head.rst"
    api_head = api_head_path.read_text()

    # Write api_index.rst with header + doctree
    output_path = Path("source") / "api_index.rst"
    with output_path.open("w") as f:
        f.write(api_head)
        f.write(doctree)


if __name__ == "__main__":
    make_api_index()
