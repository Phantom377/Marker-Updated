import os
import re
import site

def fix_union_annotations():
    site_packages = site.getsitepackages()

    pattern_union = re.compile(r'(\w+)\s*\|\s*(\w+)')
    pattern_none = re.compile(r'(\w+)\s*\|\s*None')

    files_fixed = 0

    for sp in site_packages:
        for root, _, files in os.walk(sp):
            for f in files:
                if not f.endswith(".py"):
                    continue

                path = os.path.join(root, f)

                try:
                    with open(path, "r", encoding="utf-8") as file:
                        content = file.read()

                    if "|" not in content:
                        continue

                    new_content = content

                    # Fix X | None → Optional[X]
                    new_content = pattern_none.sub(r'Optional[\1]', new_content)

                    # Fix A | B → Union[A, B]
                    new_content = pattern_union.sub(r'Union[\1, \2]', new_content)

                    if new_content != content:
                        if "Optional[" in new_content or "Union[" in new_content:
                            if "from typing import" not in new_content:
                                new_content = "from typing import Optional, Union\n" + new_content

                        with open(path, "w", encoding="utf-8") as file:
                            file.write(new_content)

                        print(f"Fixed: {path}")
                        files_fixed += 1

                except Exception:
                    pass

    print(f"\nTotal files fixed: {files_fixed}")


fix_union_annotations()