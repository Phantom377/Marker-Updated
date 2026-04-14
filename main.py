import os
import re
import json
import subprocess
from pathlib import Path
from typing import Optional


class MarkerWrapper:
    def __init__(
        self,
        output_dir: str = "./conversion_results",
        gemini_api_key: str = None,
        claude_api_key: str = None,
        openai_api_key: str = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._models = None  # lazy-loaded once

        if gemini_api_key:
            os.environ["GOOGLE_API_KEY"] = gemini_api_key
        if claude_api_key:
            os.environ["ANTHROPIC_API_KEY"] = claude_api_key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key

    def _load_models(self):
        if self._models is None:
            print("Loading models (this runs once)...")
            import torch
            from marker.models import load_all_models
            self._models = load_all_models(device="cpu", dtype=torch.float32)
        return self._models

    # ── Post-processing: lightweight cleanup only ─────────────────────────
    def fix_latex(self, markdown: str) -> str:
        fixes_applied = []

        TRIPLE_DOLLAR = re.compile(r'\$\$\$+')
        BR_IN_CELL    = re.compile(r'(\S)<br>(\S)')
        BR_LONELY     = re.compile(r'<br>')
        BLOCK_ENV     = re.compile(
            r'(?<!\$)\$(?!\$)((?:[^$\n])*?\\begin\{'
            r'(?:array|cases|matrix|pmatrix|bmatrix|vmatrix|align|gather|multline|aligned)'
            r'\}(?:[^$\n])*?)(?<!\$)\$(?!\$)'
        )

        lines = markdown.split('\n')
        processed_lines = []
        in_block_math = False

        for line in lines:
            if line.count('$$') % 2 != 0:
                in_block_math = not in_block_math

            if '$' in line and not in_block_math:
                new_line = BLOCK_ENV.sub(
                    lambda m: (
                        fixes_applied.append("upgraded $ to $$ for block env") or
                        f"$$\n{m.group(1).strip()}\n$$"
                    ),
                    line
                )
                processed_lines.append(new_line)
            else:
                processed_lines.append(line)

        markdown = '\n'.join(processed_lines)

        count = len(TRIPLE_DOLLAR.findall(markdown))
        if count:
            fixes_applied.append(f"cleaned up $$$ ({count}x)")
        markdown = TRIPLE_DOLLAR.sub('$$', markdown)

        count = len(BR_IN_CELL.findall(markdown))
        if count:
            fixes_applied.append(f"fixed <br> in table cells ({count}x)")
        markdown = BR_IN_CELL.sub(r'\1 \2', markdown)
        markdown = BR_LONELY.sub(' ', markdown)

        if fixes_applied:
            print(f"  LaTeX fixes applied: {len(fixes_applied)}")
            for fix in fixes_applied:
                print(f"    - {fix}")
        else:
            print("  No LaTeX fixes needed")

        return markdown

    # ── Single file conversion ───────────────────────────────────────────
    def convert_single(
        self,
        filepath: str,
        force_ocr: bool = False,
        start_page: int = None,
        max_pages: int = None,
        paginate_output: bool = False,
        langs: Optional[list] = None,
    ) -> dict:
        from marker.convert import convert_single_pdf
        from marker.settings import settings

        if force_ocr:
            settings.OCR_ALL_PAGES = True
        if paginate_output:
            settings.PAGINATE_OUTPUT = True

        models = self._load_models()
        text, images, metadata = convert_single_pdf(
            filepath,
            models,
            start_page=start_page,
            max_pages=max_pages,
            langs=langs,
        )

        print("  Running LaTeX post-processing...")
        text = self.fix_latex(text)

        pdf_stem = Path(filepath).stem
        doc_output_dir = self.output_dir / pdf_stem
        doc_output_dir.mkdir(parents=True, exist_ok=True)

        out_file = doc_output_dir / f"{pdf_stem}.md"
        out_file.write_text(text, encoding="utf-8")
        print(f"Saved: {out_file}")

        meta_file = doc_output_dir / f"{pdf_stem}_meta.json"
        meta_file.write_text(json.dumps(metadata, indent=4), encoding="utf-8")

        saved_images = []
        if images:
            for img_name, img_data in images.items():
                img_path = doc_output_dir / img_name
                img_data.save(str(img_path))
                saved_images.append(str(img_path))
            print(f"Saved {len(saved_images)} images to: {doc_output_dir}")
        else:
            print("No images found in document")

        return {
            "text": text,
            "metadata": metadata,
            "images": images,
            "saved_images": saved_images,
            "output_dir": str(doc_output_dir),
        }

    # ── Convert all PDFs in a folder ─────────────────────────────────────
    def convert_folder(
        self,
        input_folder: str,
        force_ocr: bool = False,
    ) -> list:
        input_path = Path(input_folder)
        if not input_path.exists():
            raise FileNotFoundError(f"Input folder not found: {input_folder}")

        pdf_files = list(input_path.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in: {input_folder}")
            return []

        print(f"Found {len(pdf_files)} PDF(s) in {input_folder}")
        print(f"Output will be saved to: {self.output_dir}\n")

        results = []
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"[{i}/{len(pdf_files)}] Converting: {pdf_path.name}")
            try:
                result = self.convert_single(
                    filepath=str(pdf_path),
                    force_ocr=force_ocr,
                )
                result["filename"] = pdf_path.name
                result["status"] = "success"
                results.append(result)
                print(f"✓ Done: {pdf_path.name}\n")
            except Exception as e:
                print(f"✗ Failed: {pdf_path.name} — {e}\n")
                results.append({
                    "filename": pdf_path.name,
                    "status": "failed",
                    "error": str(e),
                })

        success = sum(1 for r in results if r["status"] == "success")
        failed  = sum(1 for r in results if r["status"] == "failed")
        print(f"\n── Conversion Summary ──")
        print(f"Total:   {len(pdf_files)}")
        print(f"Success: {success}")
        print(f"Failed:  {failed}")
        print(f"Output:  {self.output_dir}")

        return results

    # ── Batch conversion via marker CLI ──────────────────────────────────
    def convert_batch(
        self,
        input_folder: str,
        workers: int = None,
        force_ocr: bool = False,
    ):
        cmd = ["marker", input_folder, "--output_dir", str(self.output_dir)]
        if workers:
            cmd += ["--workers", str(workers)]
        if force_ocr:
            cmd += ["--force_ocr"]

        print(f"Running batch conversion on: {input_folder}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(f"Batch conversion failed:\n{result.stderr}")


# ── Usage ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    marker = MarkerWrapper(
        output_dir="./conversion_results",
    )

    results = marker.convert_folder(
        "./input_equation",
        force_ocr=False,
    )

    for r in results:
        if r["status"] == "success":
            print(f"\n── {r['filename']} ──")
            print(f"Images: {len(r['saved_images'])}")
