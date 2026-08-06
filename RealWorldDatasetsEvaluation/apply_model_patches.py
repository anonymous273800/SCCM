from __future__ import annotations
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
PATCH_ROOT = ROOT / "ModelPatches" / "Models"

FILES = [
    (PATCH_ROOT / "RLS" / "rls_family_common.py", PROJECT_ROOT / "Models" / "RLS" / "rls_family_common.py"),
    (PATCH_ROOT / "WidrowHoff" / "widrowhoff_family_common.py", PROJECT_ROOT / "Models" / "WidrowHoff" / "widrowhoff_family_common.py"),
    (PATCH_ROOT / "WidrowHoff" / "WidrowHoff_SCCM.py", PROJECT_ROOT / "Models" / "WidrowHoff" / "WidrowHoff_SCCM.py"),
]

def main():
    for source, target in FILES:
        if not target.exists():
            raise FileNotFoundError(f"Target model file not found: {target}")
        backup = target.with_suffix(target.suffix + ".before_realworld_fix")
        if not backup.exists():
            shutil.copy2(target, backup)
        shutil.copy2(source, target)
        print(f"Patched: {target}")
    print("Model patches applied. Original files were backed up once.")

if __name__ == "__main__":
    main()
