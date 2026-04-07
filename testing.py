import os
import re
import json
import subprocess
from pathlib import Path
from enum import Enum
from typing import Optional


# ── Enums ─────────────────────────────────────────────────────────────────
class OutputFormat(Enum):
    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"
    CHUNKS = "chunks"

class ConverterType(Enum):
    PDF = "marker.converters.pdf.PdfConverter"
    TABLE = "marker.converters.table.TableConverter"
    OCR = "marker.converters.ocr.OCRConverter"
    EXTRACTION = "marker.converters.extraction.ExtractionConverter"

class LLMService(Enum):
    GEMINI = "marker.services.gemini.GoogleGeminiService"
    VERTEX = "marker.services.vertex.GoogleVertexService"
    OLLAMA = "marker.services.ollama.OllamaService"
    CLAUDE = "marker.services.claude.ClaudeService"
    OPENAI = "marker.services.openai.OpenAIService"
    AZURE = "marker.services.azure_openai.AzureOpenAIService"

os.environ["OLLAMA_MODEL"] = "llama3"
# ── Main class ────────────────────────────────────────────────────────────
class MarkerWrapper:
    def validate_ollama(self, model="llama3"):
        import requests

        try:
            res = requests.get("http://localhost:11434/api/tags", timeout=3)
            if res.status_code != 200:
                raise RuntimeError("Ollama not responding")

            models = res.json().get("models", [])
            available = [m["name"] for m in models]

            if not any(model in m for m in available):
                raise RuntimeError(f"Model '{model}' not found in Ollama")

            print(f"✓ Ollama running with model: {model}")

        except Exception as e:
            raise RuntimeError(f"Ollama validation failed: {e}")
    def __init__(
        self,
        output_dir: str = "./conversion_results",
        gemini_api_key: str = None,
        claude_api_key: str = None,
        openai_api_key: str = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if gemini_api_key:
            os.environ["GOOGLE_API_KEY"] = gemini_api_key
        if claude_api_key:
            os.environ["ANTHROPIC_API_KEY"] = claude_api_key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key

    def is_spanning_row(self, row):
        filled = [c for c in row if str(c).strip()]
        return len(filled) == 1
    
    def normalize_table_schema(self, schema):
        headers = schema.get("headers", [])
        rows = schema.get("rows", [])

        new_rows = []

        for row in rows:
            if not isinstance(row, list):
                continue

            # 🔥 detect spanning row
            if self.is_spanning_row(row):
                text = next((c for c in row if str(c).strip()), "")

                # collapse into single logical row
                new_rows.append([text])
            else:
                new_rows.append(row)

        return {
            "headers": headers,
            "rows": new_rows
        }

    def call_ollama(self, prompt: str, model="llama3"):
        import requests

        try:
            res = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0}
                },
                timeout=60
            )

            if res.status_code != 200:
                print("  ⚠️ Ollama error:", res.text)
                return ""

            return res.json().get("response", "").strip()

        except Exception as e:
            print("  ⚠️ Ollama failed:", e)
            return ""

    def table_json_to_markdown_llm(self, table_schema, model="llama3"):
        import json

        # 🔥 normalize before sending
        table_schema = self.normalize_table_schema(table_schema)

        # 🔥 truncate for safety
        raw = json.dumps(table_schema, ensure_ascii=False)
        if len(raw) > 3000:
            print("  ⚠️ Table too large, truncating...")
            raw = raw[:3000]

        prompt = f"""
    You are reconstructing a table extracted from a PDF.

    CRITICAL RULES:
    - Output ONLY a valid markdown table
    - Use EXACTLY one header row
    - Use EXACTLY one alignment row
    - ALL columns MUST be center aligned using :--:
    - Do NOT hallucinate or invent data

    SPECIAL CASE:
    If a row contains content in only ONE cell and the rest are empty:
    → This represents a merged row spanning ALL columns
    → You MUST expand it logically across the table OR collapse appropriately

    BAD example:
    | text |   |   |

    GOOD handling:
    | text |
    OR
    | text | text | text |

    depending on context

    Ensure:
    - No duplicate headers
    - No repeated alignment rows
    - Consistent column count

    Table JSON:
    {raw}
    """

        return self.call_ollama(prompt, model=model)

    # ── Post-processing: lightweight cleanup only ─────────────────────────
    def fix_latex(self, markdown: str) -> str:
        """
        Lightweight post-processor for simple cleanup only.
        Complex equation fixes (Word fractions, broken LaTeX) are
        handled by the LLM during conversion — use use_llm=True.
        """
        fixes_applied = []

        # Pre-compile patterns
        TRIPLE_DOLLAR = re.compile(r'\$\$\$+')
        BR_IN_CELL    = re.compile(r'(\S)<br>(\S)')
        BR_LONELY     = re.compile(r'<br>')
        BLOCK_ENV     = re.compile(
            r'(?<!\$)\$(?!\$)((?:[^$\n])*?\\begin\{'
            r'(?:array|cases|matrix|pmatrix|bmatrix|vmatrix|align|gather|multline|aligned)'
            r'\}(?:[^$\n])*?)(?<!\$)\$(?!\$)'
        )

        # Fix 1 — upgrade inline $ to $$ for block environments
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

        # Fix 2 — clean up triple $$$
        count = len(TRIPLE_DOLLAR.findall(markdown))
        if count:
            fixes_applied.append(f"cleaned up $$$ ({count}x)")
        markdown = TRIPLE_DOLLAR.sub('$$', markdown)

        # Fix 3 — fix <br> inside table cells e.g. 101 <= X <=<br>500
        count = len(BR_IN_CELL.findall(markdown))
        if count:
            fixes_applied.append(f"fixed <br> in table cells ({count}x)")
        markdown = BR_IN_CELL.sub(r'\1 \2', markdown)
        markdown = BR_LONELY.sub(' ', markdown)

        # Report
        if fixes_applied:
            print(f"  LaTeX fixes applied: {len(fixes_applied)}")
            for fix in fixes_applied:
                print(f"    - {fix}")
        else:
            print("No LaTeX fixes needed")

        return markdown

    # ── Single file conversion ───────────────────────────────────────────
    def convert_single(
        self,
        filepath: str,
        output_format: OutputFormat = OutputFormat.MARKDOWN,
        converter: ConverterType = ConverterType.PDF,
        use_llm: bool = False,
        llm_service: LLMService = LLMService.OLLAMA,
        force_ocr: bool = False,
        page_range: str = None,
        paginate_output: bool = False,
        disable_image_extraction: bool = False,
        debug: bool = False,
    ) -> dict:
        from marker.converters.pdf import PdfConverter
        from marker.converters.table import TableConverter
        from marker.converters.ocr import OCRConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
        from marker.config.parser import ConfigParser

        config = {
            "output_format": output_format.value,
            "output_dir": str(self.output_dir),
        }
        if force_ocr:
            config["force_ocr"] = True
        if page_range:
            config["page_range"] = page_range
        if paginate_output:
            config["paginate_output"] = True
        if disable_image_extraction:
            config["disable_image_extraction"] = True
        if debug:
            config["debug"] = True
        if use_llm:
            config["use_llm"] = True
            config["llm_service"] = llm_service.value
        
        # Validate Ollama before running
        if use_llm and llm_service == LLMService.OLLAMA:
            self.validate_ollama(model=os.environ.get("OLLAMA_MODEL", "llama3"))

        config_parser = ConfigParser(config)
        models = create_model_dict()

        converter_cls = {
            ConverterType.PDF: PdfConverter,
            ConverterType.TABLE: TableConverter,
            ConverterType.OCR: OCRConverter,
        }.get(converter, PdfConverter)

        conv = converter_cls(
            config=config_parser.generate_config_dict(),
            artifact_dict=models,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service() if use_llm else None,
        )

        rendered = conv(filepath)
        text, metadata, images = text_from_rendered(rendered)

        # ── Post-processing: lightweight cleanup ──
        if output_format == OutputFormat.MARKDOWN:
            print("  Running LaTeX post-processing...")
            text = self.fix_latex(text)

        # ── Save markdown/json/html ──
        pdf_stem = Path(filepath).stem
        doc_output_dir = self.output_dir / pdf_stem
        doc_output_dir.mkdir(parents=True, exist_ok=True)

        out_file = doc_output_dir / f"{pdf_stem}.{output_format.value}"
        out_file.write_text(text, encoding="utf-8")
        print(f"Saved text: {out_file}")

        # ── Save images ──
        if images:
            images_dir = doc_output_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            saved_images = []
            for img_name, img_data in images.items():
                img_path = images_dir / f"{img_name}.png"
                img_data.save(str(img_path))
                saved_images.append(str(img_path))

            print(f"Saved {len(saved_images)} images to: {images_dir}")
        else:
            print("No images found in document")
            saved_images = []

        return {
            "text": text,
            "metadata": metadata,
            "images": images,
            "saved_images": saved_images,
            "output_dir": str(doc_output_dir)
        }

    # ── Convert all PDFs in a folder ─────────────────────────────────────
    def convert_folder(
        self,
        input_folder: str,
        output_format: OutputFormat = OutputFormat.MARKDOWN,
        use_llm: bool = False,
        llm_service: LLMService = LLMService.GEMINI,   # ← add this
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
                    output_format=output_format,
                    use_llm=use_llm,
                    llm_service=llm_service,   # ← add this
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
                    "error": str(e)
                })

        success = sum(1 for r in results if r["status"] == "success")
        failed  = sum(1 for r in results if r["status"] == "failed")
        print(f"\n── Conversion Summary ──")
        print(f"Total:   {len(pdf_files)}")
        print(f"Success: {success}")
        print(f"Failed:  {failed}")
        print(f"Output:  {self.output_dir}")

        return results

    # ── Batch conversion (marker CLI) ────────────────────────────────────
    def convert_batch(
        self,
        input_folder: str,
        output_format: OutputFormat = OutputFormat.MARKDOWN,
        workers: int = None,
        use_llm: bool = False,
        force_ocr: bool = False,
    ):
        cmd = ["marker", input_folder, "--output_dir", str(self.output_dir),
               "--output_format", output_format.value]
        if workers:
            cmd += ["--workers", str(workers)]
        if use_llm:
            cmd += ["--use_llm"]
        if force_ocr:
            cmd += ["--force_ocr"]

        print(f"Running batch conversion on: {input_folder}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(f"Batch conversion failed:\n{result.stderr}")

    # ── Multi-GPU batch conversion ────────────────────────────────────────
    def convert_chunk(
        self,
        input_folder: str,
        num_devices: int = 2,
        num_workers: int = 4,
    ):
        env = os.environ.copy()
        env["NUM_DEVICES"] = str(num_devices)
        env["NUM_WORKERS"] = str(num_workers)

        cmd = ["marker_chunk_convert", input_folder, str(self.output_dir)]
        print(f"Running multi-GPU conversion: {num_devices} GPUs, {num_workers} workers")
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        print(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(f"Chunk conversion failed:\n{result.stderr}")

    # ── API server ────────────────────────────────────────────────────────
    def start_server(self, port: int = 8001):
        cmd = ["marker_server", "--port", str(port)]
        print(f"Starting Marker API server on port {port}...")
        print(f"Docs available at: http://localhost:{port}/docs")
        subprocess.run(cmd)

    def call_server(self, filepath: str, port: int = 8001, **kwargs) -> dict:
        import requests
        data = {"filepath": filepath, **kwargs}
        res = requests.post(f"http://localhost:{port}/marker", data=json.dumps(data))
        res.raise_for_status()
        return res.json()

    # ── Structured extraction ─────────────────────────────────────────────
    def extract_structured(
        self,
        filepath: str,
        schema: dict,
        llm_service: LLMService = LLMService.GEMINI,
    ) -> dict:
        from marker.converters.extraction import ExtractionConverter
        from marker.models import create_model_dict
        from marker.config.parser import ConfigParser

        config_parser = ConfigParser({
            "page_schema": schema,
            "use_llm": True,
            "llm_service": llm_service.value,
        })

        conv = ExtractionConverter(
            artifact_dict=create_model_dict(),
            config=config_parser.generate_config_dict(),
            llm_service=config_parser.get_llm_service(),
        )

        rendered = conv(filepath)
        return rendered

    # ── Extract blocks ────────────────────────────────────────────────────
    def extract_blocks(self, filepath: str, block_type: str = "Table") -> list:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.schema import BlockTypes

        conv = PdfConverter(artifact_dict=create_model_dict())
        document = conv.build_document(filepath)

        bt = getattr(BlockTypes, block_type)
        blocks = document.contained_blocks((bt,))
        print(f"Found {len(blocks)} {block_type} blocks")
        return blocks


# ── Usage ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    marker = MarkerWrapper(
        output_dir="./conversion_results",
        # No API key needed — Ollama runs locally
    )
    results = marker.convert_folder(
        "./input_equation",
        use_llm=True,
        llm_service=LLMService.OLLAMA,   # ← use Ollama
        force_ocr=True,                  # ← enable force OCR
    )

    # Print a preview of each converted file
    for r in results:
        if r["status"] == "success":
            print(f"\n── {r['filename']} ──")
            print(f"Images: {len(r['saved_images'])}")