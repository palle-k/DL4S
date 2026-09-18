#
#  docc_landing_page.py
#  DL4S
#
#  Created by Palle Klewitz on 18.09.26.
#  Copyright (c) 2026 - Palle Klewitz
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

"""
Rewrites README.md to be compatible with DocC.
"""

import re
import sys
from pathlib import Path


class LandingPageWriter:
    heading_pattern = re.compile(r"^#{1,6} (.+?)\s*#*$")
    anchor_link_pattern = re.compile(r"\]\(#([^)]+)\)")

    def __init__(self, module_name: str, readme: str) -> None:
        self.module_name = module_name
        self.readme_lines = readme.splitlines()

    def render(self) -> str:
        anchors = self.docc_anchors()
        body = "\n".join(self.rewrite_anchor_links(line, anchors) for line in self.readme_lines)
        return f"# ``{self.module_name}``\n\n{body}\n"

    def docc_anchors(self) -> dict[str, str]:
        """Maps GitHub-flavored markdown headings to DocC anchor."""
        anchors: dict[str, str] = {}
        in_code_block = False
        for line in self.readme_lines:
            if line.startswith("```"):
                in_code_block = not in_code_block
                continue
            match = self.heading_pattern.match(line)
            if in_code_block or match is None:
                continue
            anchor = self.docc_anchor(match.group(1))
            anchors.setdefault(anchor.lower(), anchor)
        return anchors

    @staticmethod
    def docc_anchor(heading: str) -> str:
        return "".join("-" if character == " " else character for character in heading if character == " " or character.isalnum() or character in "-_")

    def rewrite_anchor_links(self, line: str, anchors: dict[str, str]) -> str:
        return self.anchor_link_pattern.sub(lambda match: f"](#{anchors.get(match.group(1).lower(), match.group(1))})", line)


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(f"usage: {sys.argv[0]} <module name> <README.md> <landing page.md>")
    module_name, readme_path, output_path = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(LandingPageWriter(module_name, readme_path.read_text()).render())


if __name__ == "__main__":
    main()
