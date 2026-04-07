import os
import re
import json
import subprocess
from pathlib import Path
from enum import Enum
from typing import Optional
import requests
from typing import List, Dict, Any


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


# ── Main class ────────────────────────────────────────────────────────────
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

        if gemini_api_key:
            os.environ["GOOGLE_API_KEY"] = gemini_api_key
        if claude_api_key:
            os.environ["ANTHROPIC_API_KEY"] = claude_api_key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key

    # ── Post-processing: lightweight cleanup only ─────────────────────────
    def fix_latex(self, markdown: str) -> str:
        """
        Lightweight post-processor for simple cleanup only.
        Complex equation fixes (Word fractions, broken LaTeX) are
        handled by the LLM during conversion — use use_llm=True.
        """
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

    def fix_table_alignment(self, markdown: str) -> str:
        lines = markdown.split("\n")
        new_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]

            if line.strip().startswith("|") and i + 1 < len(lines):
                next_line = lines[i + 1]

                # only insert alignment if next line is NOT alignment
                if not re.match(r'^\|\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|$', next_line.strip()):
                    cols = line.count("|") - 1
                    align_row = "|" + "|".join([":--:" for _ in range(cols)]) + "|"

                    new_lines.append(line)
                    new_lines.append(align_row)
                    i += 1
                    continue

            new_lines.append(line)
            i += 1

        return "\n".join(new_lines)

    def call_ollama(self, prompt: str, model: str = "llama3:8b") -> str:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0
            }
        }

        res = requests.post(url, json=payload, timeout=180)
        res.raise_for_status()
        data = res.json()
        return data.get("response", "").strip()

    def extract_tables_from_json(self, json_text: str) -> List[Dict[str, Any]]:
        """
        Tries to extract table-like blocks from Marker JSON output.
        Assumes Marker JSON is valid JSON text.
        """
        try:
            data = json.loads(json_text)
            print("JSON length:", len(json_text))
        except Exception:
            return []

        tables = []

        def walk(node):
            if isinstance(node, dict):
                block_type = str(node.get("block_type", "")).lower()
                node_type = str(node.get("type", "")).lower()

                if "table" in block_type or node_type == "table":
                    tables.append(node)

                for v in node.values():
                    walk(v)

            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(data)
        return tables

    def table_json_to_minimal_schema(self, table_obj: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize Marker table JSON into a simpler schema for LLM.
        """
        simplified = {
            "headers": [],
            "rows": [],
            "raw": table_obj
        }

        children = table_obj.get("children") or []

        if isinstance(children, list):
            rows = []
            for child in children:
                if not isinstance(child, dict):
                    continue

                sub_children = child.get("children") or []
                if not isinstance(sub_children, list):
                    continue

                row_vals = []
                for sub in sub_children:
                    if not isinstance(sub, dict):
                        continue

                    text = sub.get("text") or sub.get("html") or sub.get("content") or ""

                    if not text:
                        nested = sub.get("children") or []
                        if isinstance(nested, list):
                            text = " ".join(
                                c.get("text", "")
                                for c in nested
                                if isinstance(c, dict)
                            ).strip()

                    row_vals.append(str(text).strip())

                if row_vals:
                    rows.append(row_vals)

            if rows:
                simplified["headers"] = rows[0]
                simplified["rows"] = rows[1:]
                return simplified

        # Fallback if direct cells/rows exist
        if "rows" in table_obj and isinstance(table_obj["rows"], list):
            rows = []
            for r in table_obj["rows"]:
                if isinstance(r, list):
                    rows.append([str(x).strip() for x in r])
                elif isinstance(r, dict) and "cells" in r:
                    rows.append([str(c).strip() for c in r["cells"]])

            if rows:
                simplified["headers"] = rows[0]
                simplified["rows"] = rows[1:]
                return simplified

        return simplified

    def table_json_to_markdown_llm(self, table_schema: Dict[str, Any], model: str = "llama3:8b") -> str:
        prompt = f"""
You are fixing a table extracted from a PDF.

Convert the following JSON table into a VALID markdown table.

Rules:
- Output ONLY the markdown table
- Use exactly one header row
- Use exactly one alignment row
- ALL columns must be center aligned using :--:
- Preserve the table content faithfully
- If merged cells from the PDF are unclear, make the most reasonable reconstruction
- Keep empty cells blank instead of inventing values
- Do not add explanations
- Do not wrap in triple backticks

JSON table:
{json.dumps(table_schema, ensure_ascii=False, indent=2)}
"""
        return self.call_ollama(prompt, model=model)

    def extract_markdown_tables(self, markdown: str) -> List[Dict[str, Any]]:
        """
        Extract existing markdown table blocks from the markdown output.
        """
        lines = markdown.splitlines()
        tables = []
        current = []
        start = None

        def looks_like_table_line(line: str) -> bool:
            s = line.strip()
            return s.startswith("|") and s.count("|") >= 2

        for i, line in enumerate(lines):
            if looks_like_table_line(line):
                if start is None:
                    start = i
                    current = [line]
                else:
                    current.append(line)
            else:
                if start is not None:
                    tables.append({
                        "start": start,
                        "end": i - 1,
                        "text": "\n".join(current)
                    })
                    start = None
                    current = []

        if start is not None:
            tables.append({
                "start": start,
                "end": len(lines) - 1,
                "text": "\n".join(current)
            })

        return tables

    def replace_markdown_tables(self, markdown: str, replacements: List[str]) -> str:
        """
        Replace tables in markdown in order.
        Assumes number/order of extracted markdown tables roughly matches JSON tables.
        """
        table_spans = self.extract_markdown_tables(markdown)
        if not table_spans:
            return markdown

        lines = markdown.splitlines()
        out = []
        cursor = 0

        for idx, span in enumerate(table_spans):
            out.extend(lines[cursor:span["start"]])

            if idx < len(replacements) and replacements[idx].strip():
                out.extend(replacements[idx].strip().splitlines())
            else:
                out.extend(lines[span["start"]:span["end"] + 1])

            cursor = span["end"] + 1

        out.extend(lines[cursor:])
        return "\n".join(out)

    def clean_tables_via_json_llm(
        self,
        markdown_text: str,
        json_text: str,
        model: str = "llama3:8b"
    ) -> str:
        tables = self.extract_tables_from_json(json_text)
        if not tables:
            print("  No JSON tables found for LLM cleaning")
            return markdown_text

        cleaned_tables = []

        for i, table_obj in enumerate(tables, 1):
            print(f"  Cleaning table {i} with Llama 3...")
            schema = self.table_json_to_minimal_schema(table_obj)
            cleaned_md = self.table_json_to_markdown_llm(schema, model=model)

            # basic sanity check
            if cleaned_md.strip().startswith("|"):
                cleaned_tables.append(cleaned_md.strip())
            else:
                cleaned_tables.append("")

        repaired_markdown = self.replace_markdown_tables(markdown_text, cleaned_tables)
        return repaired_markdown
    
    # ── Internal: convert to one format ──────────────────────────────────
    def _convert_one_format(
        self,
        filepath: str,
        output_format: OutputFormat,
        converter_cls,
        config_parser,
        models,
        use_llm: bool,
        doc_output_dir: Path,
    ) -> dict:
        from marker.output import text_from_rendered

        conv = converter_cls(
            config=config_parser.generate_config_dict(),
            artifact_dict=models,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service() if use_llm else None,
        )

        print(f"    Starting core conversion: {filepath}")
        rendered = conv(filepath)
        print(f"    Finished core conversion: {filepath}")
        text, metadata, images = text_from_rendered(rendered)

        # Post-processing for markdown only
        if output_format == OutputFormat.MARKDOWN:
            print("  Running post-processing...")
            text = self.fix_latex(text)
            text = self.fix_table_alignment(text)

        # Save output file
        pdf_stem = Path(filepath).stem
        out_file = doc_output_dir / f"{pdf_stem}.{output_format.value}"
        out_file.write_text(text, encoding="utf-8")
        print(f"  Saved {output_format.value}: {out_file}")

        return {"text": text, "metadata": metadata, "images": images}

    # ── Single file conversion ───────────────────────────────────────────
    def convert_single(
        self,
        filepath: str,
        output_formats: list = None,
        converter: ConverterType = ConverterType.PDF,
        use_llm: bool = False,
        llm_service: LLMService = LLMService.GEMINI,
        force_ocr: bool = False,
        page_range: str = None,
        paginate_output: bool = False,
        disable_image_extraction: bool = False,
        debug: bool = False,
        table_llm_model: str = "llama3:8b",
        clean_tables_with_llm: bool = True,
    ) -> dict:
        from marker.converters.pdf import PdfConverter
        from marker.converters.table import TableConverter
        from marker.converters.ocr import OCRConverter
        from marker.models import create_model_dict
        from marker.config.parser import ConfigParser

        if output_formats is None:
            output_formats = [OutputFormat.MARKDOWN, OutputFormat.JSON]

        # force both markdown and json if table cleaning is enabled
        if clean_tables_with_llm:
            if OutputFormat.MARKDOWN not in output_formats:
                output_formats.append(OutputFormat.MARKDOWN)
            if OutputFormat.JSON not in output_formats:
                output_formats.append(OutputFormat.JSON)

        pdf_stem = Path(filepath).stem
        doc_output_dir = self.output_dir / pdf_stem
        doc_output_dir.mkdir(parents=True, exist_ok=True)

        models = create_model_dict()
        converter_cls = {
            ConverterType.PDF: PdfConverter,
            ConverterType.TABLE: TableConverter,
            ConverterType.OCR: OCRConverter,
        }.get(converter, PdfConverter)

        all_results = {}
        saved_images = []

        for output_format in output_formats:
            print(f"  Converting to {output_format.value}...")

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

            config_parser = ConfigParser(config)

            result = self._convert_one_format(
                filepath=filepath,
                output_format=output_format,
                converter_cls=converter_cls,
                config_parser=config_parser,
                models=models,
                use_llm=use_llm,
                doc_output_dir=doc_output_dir,
            )
            all_results[output_format.value] = result

        first_result = next(iter(all_results.values()))
        images = first_result.get("images", {})

        if images:
            images_dir = doc_output_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            for img_name, img_data in images.items():
                img_path = images_dir / f"{img_name}.png"
                img_data.save(str(img_path))
                saved_images.append(str(img_path))
            print(f"  Saved {len(saved_images)} images to: {images_dir}")
        else:
            print("  No images found in document")

        markdown_text = all_results.get("markdown", {}).get("text", "")
        json_text = all_results.get("json", {}).get("text", "")

        if clean_tables_with_llm and markdown_text and json_text:
            print("  Rebuilding markdown tables from JSON with Llama 3...")
            try:
                markdown_text = self.clean_tables_via_json_llm(
                    markdown_text=markdown_text,
                    json_text=json_text,
                    model=table_llm_model,
                )

                repaired_file = doc_output_dir / f"{pdf_stem}.tables_cleaned.markdown"
                repaired_file.write_text(markdown_text, encoding="utf-8")
                print(f"  Saved repaired markdown: {repaired_file}")
            except Exception as e:
                print(f"  Table LLM cleaning failed: {e}")

        return {
            "markdown": markdown_text,
            "json": json_text,
            "metadata": first_result.get("metadata", {}),
            "images": images,
            "saved_images": saved_images,
            "output_dir": str(doc_output_dir),
        }

    # ── Convert all PDFs in a folder ─────────────────────────────────────
    def convert_folder(
        self,
        input_folder: str,
        output_formats: list = None,
        use_llm: bool = False,
        force_ocr: bool = False,
        clean_tables_with_llm: bool = True,
        table_llm_model: str = "llama3:8b",
    ) -> list:
        input_path = Path(input_folder)
        if not input_path.exists():
            raise FileNotFoundError(f"Input folder not found: {input_folder}")

        pdf_files = list(input_path.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in: {input_folder}")
            return []

        if output_formats is None:
            output_formats = [OutputFormat.MARKDOWN, OutputFormat.JSON]

        print(f"Found {len(pdf_files)} PDF(s) in {input_folder}")
        print(f"Output formats: {[f.value for f in output_formats]}")
        print(f"Output will be saved to: {self.output_dir}\n")

        results = []
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"[{i}/{len(pdf_files)}] Converting: {pdf_path.name}")
            try:
                result = self.convert_single(
                    filepath=str(pdf_path),
                    output_formats=output_formats,
                    use_llm=use_llm,
                    force_ocr=force_ocr,
                    clean_tables_with_llm=clean_tables_with_llm,
                    table_llm_model=table_llm_model,
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
        failed = sum(1 for r in results if r["status"] == "failed")
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
    marker = MarkerWrapper(output_dir="./conversion_results")

    results = marker.convert_folder(
        "./input_equation",
        output_formats=[OutputFormat.MARKDOWN, OutputFormat.JSON],
        use_llm=False,              # keep Marker LLM off
        force_ocr=False,
        clean_tables_with_llm=True, # use Llama 3 only for table repair
        table_llm_model="llama3:8b"
    )

    for r in results:
        if r["status"] == "success":
            print(f"\n── {r['filename']} ──")
            print(f"Images:  {len(r['saved_images'])}")
            print(f"Output:  {r['output_dir']}")

    # ── Print summary ──

    # ── Other format options ──
    # Markdown only:  output_formats=[OutputFormat.MARKDOWN]
    # JSON only:      output_formats=[OutputFormat.JSON]
    # All formats:    output_formats=[OutputFormat.MARKDOWN, OutputFormat.JSON, OutputFormat.HTML]