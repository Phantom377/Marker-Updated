import os
import re
import json
import subprocess
from pathlib import Path
from enum import Enum
from typing import Optional, List, Dict, Any


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


class MarkerWrapper:
    def __init__(
        self,
        output_dir="./conversion_results",
        gemini_api_key=None,
        claude_api_key=None,
        openai_api_key=None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if gemini_api_key:
            os.environ["GOOGLE_API_KEY"] = gemini_api_key
        if claude_api_key:
            os.environ["ANTHROPIC_API_KEY"] = claude_api_key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key

    def fix_latex(self, markdown):
        fixes_applied = []

        triple_dollar = re.compile(r"\$\$\$+")
        br_in_cell = re.compile(r"(\S)<br>(\S)")
        br_lonely = re.compile(r"<br>")
        block_env = re.compile(
            r"(?<!\$)\$(?!\$)((?:[^$\n])*?\\begin\{"
            r"(?:array|cases|matrix|pmatrix|bmatrix|vmatrix|align|gather|multline|aligned)"
            r"\}(?:[^$\n])*?)(?<!\$)\$(?!\$)"
        )

        lines = markdown.split("\n")
        processed_lines = []
        in_block_math = False

        for line in lines:
            if line.count("$$") % 2 != 0:
                in_block_math = not in_block_math

            if "$" in line and not in_block_math:
                new_line = block_env.sub(
                    lambda m: (
                        fixes_applied.append("upgraded $ to $$ for block env") or
                        "$$\n{0}\n$$".format(m.group(1).strip())
                    ),
                    line,
                )
                processed_lines.append(new_line)
            else:
                processed_lines.append(line)

        markdown = "\n".join(processed_lines)

        count = len(triple_dollar.findall(markdown))
        if count:
            fixes_applied.append("cleaned up $$$ ({0}x)".format(count))
        markdown = triple_dollar.sub("$$", markdown)

        count = len(br_in_cell.findall(markdown))
        if count:
            fixes_applied.append("fixed <br> in table cells ({0}x)".format(count))
        markdown = br_in_cell.sub(r"\1 \2", markdown)
        markdown = br_lonely.sub(" ", markdown)

        if fixes_applied:
            print("  LaTeX fixes applied: {0}".format(len(fixes_applied)))
            for fix in fixes_applied:
                print("    - {0}".format(fix))
        else:
            print("  No LaTeX fixes needed")

        return markdown

    def convert_single(
        self,
        filepath,
        output_format=OutputFormat.MARKDOWN,
        converter=ConverterType.PDF,
        use_llm=False,
        llm_service=LLMService.GEMINI,
        force_ocr=False,
        page_range=None,
        paginate_output=False,
        disable_image_extraction=False,
        debug=False,
    ):
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

        if output_format == OutputFormat.MARKDOWN:
            print("  Running LaTeX post-processing...")
            text = self.fix_latex(text)

        pdf_stem = Path(filepath).stem
        doc_output_dir = self.output_dir / pdf_stem
        doc_output_dir.mkdir(parents=True, exist_ok=True)

        out_file = doc_output_dir / "{0}.{1}".format(pdf_stem, output_format.value)
        out_file.write_text(text, encoding="utf-8")
        print("Saved text: {0}".format(out_file))

        if images:
            images_dir = doc_output_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            saved_images = []
            for img_name, img_data in images.items():
                img_path = images_dir / "{0}.png".format(img_name)
                img_data.save(str(img_path))
                saved_images.append(str(img_path))

            print("Saved {0} images to: {1}".format(len(saved_images), images_dir))
        else:
            print("No images found in document")
            saved_images = []

        return {
            "text": text,
            "metadata": metadata,
            "images": images,
            "saved_images": saved_images,
            "output_dir": str(doc_output_dir),
        }

    def convert_folder(
        self,
        input_folder,
        output_format=OutputFormat.MARKDOWN,
        use_llm=False,
        force_ocr=False,
    ):
        input_path = Path(input_folder)
        if not input_path.exists():
            raise FileNotFoundError("Input folder not found: {0}".format(input_folder))

        pdf_files = list(input_path.glob("*.pdf"))
        if not pdf_files:
            print("No PDF files found in: {0}".format(input_folder))
            return []

        print("Found {0} PDF(s) in {1}".format(len(pdf_files), input_folder))
        print("Output will be saved to: {0}\n".format(self.output_dir))

        results = []
        for i, pdf_path in enumerate(pdf_files, 1):
            print("[{0}/{1}] Converting: {2}".format(i, len(pdf_files), pdf_path.name))
            try:
                result = self.convert_single(
                    filepath=str(pdf_path),
                    output_format=output_format,
                    use_llm=use_llm,
                    force_ocr=force_ocr,
                )
                result["filename"] = pdf_path.name
                result["status"] = "success"
                results.append(result)
                print("✓ Done: {0}\n".format(pdf_path.name))
            except Exception as e:
                print("✗ Failed: {0} — {1}\n".format(pdf_path.name, e))
                results.append({
                    "filename": pdf_path.name,
                    "status": "failed",
                    "error": str(e),
                })

        success = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] == "failed")
        print("\n── Conversion Summary ──")
        print("Total:   {0}".format(len(pdf_files)))
        print("Success: {0}".format(success))
        print("Failed:  {0}".format(failed))
        print("Output:  {0}".format(self.output_dir))

        return results

    def convert_batch(
        self,
        input_folder,
        output_format=OutputFormat.MARKDOWN,
        workers=None,
        use_llm=False,
        force_ocr=False,
    ):
        cmd = [
            "marker",
            input_folder,
            "--output_dir",
            str(self.output_dir),
            "--output_format",
            output_format.value,
        ]
        if workers:
            cmd += ["--workers", str(workers)]
        if use_llm:
            cmd += ["--use_llm"]
        if force_ocr:
            cmd += ["--force_ocr"]

        print("Running batch conversion on: {0}".format(input_folder))
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            raise RuntimeError("Batch conversion failed:\n{0}".format(result.stderr))

    def convert_chunk(self, input_folder, num_devices=2, num_workers=4):
        env = os.environ.copy()
        env["NUM_DEVICES"] = str(num_devices)
        env["NUM_WORKERS"] = str(num_workers)

        cmd = ["marker_chunk_convert", input_folder, str(self.output_dir)]
        print(
            "Running multi-GPU conversion: {0} GPUs, {1} workers".format(
                num_devices, num_workers
            )
        )
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        print(result.stdout)
        if result.returncode != 0:
            raise RuntimeError("Chunk conversion failed:\n{0}".format(result.stderr))

    def start_server(self, port=8001):
        cmd = ["marker_server", "--port", str(port)]
        print("Starting Marker API server on port {0}...".format(port))
        print("Docs available at: http://localhost:{0}/docs".format(port))
        subprocess.run(cmd)

    def call_server(self, filepath, port=8001, **kwargs):
        import requests

        data = {"filepath": filepath}
        data.update(kwargs)

        res = requests.post(
            "http://localhost:{0}/marker".format(port),
            data=json.dumps(data),
        )
        res.raise_for_status()
        return res.json()

    def extract_structured(self, filepath, schema, llm_service=LLMService.GEMINI):
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

    def extract_blocks(self, filepath, block_type="Table"):
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.schema import BlockTypes

        conv = PdfConverter(artifact_dict=create_model_dict())
        document = conv.build_document(filepath)

        bt = getattr(BlockTypes, block_type)
        blocks = document.contained_blocks((bt,))
        print("Found {0} {1} blocks".format(len(blocks), block_type))
        return blocks


if __name__ == "__main__":
    marker = MarkerWrapper(
        output_dir="./conversion_results",
        gemini_api_key="",
    )

    results = marker.convert_folder(
        "./input_equation",
        use_llm=False,
        force_ocr=False,
    )

    for r in results:
        if r["status"] == "success":
            print("\n── {0} ──".format(r["filename"]))
            print("Images: {0}".format(len(r["saved_images"])))