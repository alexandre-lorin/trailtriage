from pathlib import Path


def safe_target(output: Path, target: Path) -> Path:
    output = output.absolute()
    target = target.absolute()
    if not target.is_relative_to(output):
        raise ValueError("Target outside output")
    if ".." in target.relative_to(output).parts:
        raise ValueError("Parent traversal in output path")
    current = target
    while current != output.parent:
        if current.is_symlink():
            raise ValueError(f"Refusing a symlink at managed path: {current}")
        if current.is_file() and current.stat().st_nlink > 1:
            raise ValueError(f"Refusing a hardlinked managed file: {current}")
        current = current.parent
    return target
