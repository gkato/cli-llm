import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "pdf_to_gemma_curl.sh"


class PdfToUnlimitedScriptTests(unittest.TestCase):
    def test_multi_page_request_and_cleaned_output(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            fake_bin = temp_path / "bin"
            fake_bin.mkdir()
            pdf = temp_path / "sample.pdf"
            pdf.write_bytes(b"%PDF-test")
            output = temp_path / "output"

            pdfium = fake_bin / "pypdfium2"
            pdfium.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env python3
                    import sys
                    from pathlib import Path

                    if sys.argv[1] == "pdfinfo":
                        print("Page Count: 2")
                    elif sys.argv[1] == "render":
                        output = Path(sys.argv[sys.argv.index("--output") + 1])
                        output.mkdir(parents=True, exist_ok=True)
                        (output / "page1.png").write_bytes(b"page-one")
                        (output / "page2.png").write_bytes(b"page-two")
                    else:
                        raise SystemExit(2)
                    """
                )
            )
            pdfium.chmod(0o755)

            curl = fake_bin / "curl"
            curl.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env python3
                    import json
                    import sys
                    from pathlib import Path

                    args = sys.argv[1:]
                    request = args[args.index("--data-binary") + 1]
                    response = Path(args[args.index("--output") + 1])
                    payload = json.loads(Path(request.removeprefix("@")).read_text())
                    content = payload["messages"][0]["content"]
                    assert payload["model"] == "baidu/Unlimited-OCR"
                    assert content[0]["text"] == "<image>Multi page parsing."
                    assert len(content) == 3
                    assert all(item["image_url"]["url"].startswith("data:image/png;base64,") for item in content[1:])
                    assert payload["skip_special_tokens"] is False
                    assert payload["vllm_xargs"] == {"ngram_size": 35, "window_size": 1024}
                    response.write_text(json.dumps({
                        "choices": [{"message": {"content":
                            "<|ref|># Heading<|/ref|>\\n<|det|>text [0,0,1,1]<|/det|>\\nBody"
                        }}]
                    }))
                    """
                )
            )
            curl.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{fake_bin}:{env['PATH']}",
                    "PDFIUM_BIN": str(pdfium),
                    "UNLIMITED_API_KEY": "test-key",
                }
            )
            result = subprocess.run(
                [str(SCRIPT), str(pdf), str(output)],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                (output / "transcription.md").read_text(),
                "# Heading\n\nBody\n",
            )
            self.assertTrue((output / "response.json").is_file())
            self.assertTrue((output / "transcription.raw.txt").is_file())


if __name__ == "__main__":
    unittest.main()
