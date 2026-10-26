"""docs/report.md → docs/report.docx (한글 글꼴·표 테두리·그림 폭 정리).

    pip install pypandoc_binary python-docx
    python docs/make_docx.py
"""

import re
import subprocess
import tempfile
from pathlib import Path

import pypandoc
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

DOCS = Path(__file__).resolve().parent
KO_FONT, MONO_FONT = "Malgun Gothic", "Consolas"


def main() -> None:
    pandoc = pypandoc.get_pandoc_path()
    tmp = Path(tempfile.mkdtemp())
    subprocess.run([pandoc, "-o", tmp / "ref.docx", "--print-default-data-file", "reference.docx"], check=True)
    ref = Document(tmp / "ref.docx")
    for st in ref.styles:
        rpr = st.element.get_or_add_rPr()
        rf = rpr.find(qn("w:rFonts"))
        if rf is None:
            rf = rpr.makeelement(qn("w:rFonts"), {})
            rpr.append(rf)
        mono = st.name in ("Source Code", "Verbatim Char")
        for a in ("w:ascii", "w:hAnsi"):
            rf.set(qn(a), MONO_FONT if mono else KO_FONT)
        rf.set(qn("w:eastAsia"), KO_FONT)
    ref.styles["Normal"].font.size = Pt(10.5)
    ref.save(tmp / "ref_ko.docx")

    # 그림 바로 아래의 기울임 캡션 줄은 pandoc 이 alt 텍스트로 캡션을 만들므로 지운다.
    md = re.sub(r"\n\*그림 \d+\.[^\n]*\*\n", "\n", (DOCS / "report.md").read_text())
    (tmp / "report.md").write_text(md)
    subprocess.run([pandoc, tmp / "report.md", "-f", "gfm", "-o", tmp / "raw.docx",
                    "--reference-doc", tmp / "ref_ko.docx", "--resource-path", DOCS], check=True)

    d = Document(tmp / "raw.docx")
    for t in d.tables:
        borders = t._tbl.tblPr.makeelement(qn("w:tblBorders"), {})
        for e in ("top", "left", "bottom", "right", "insideH", "insideV"):
            borders.append(borders.makeelement(qn("w:" + e), {qn("w:val"): "single", qn("w:sz"): "4",
                                                               qn("w:space"): "0", qn("w:color"): "999999"}))
        t._tbl.tblPr.append(borders)
        for row in t.rows:
            for c in row.cells:
                for p in c.paragraphs:
                    for r in p.runs:
                        r.font.size = Pt(9)
    for s in d.inline_shapes:
        if s.width > Inches(6.3):
            ratio = Inches(6.3) / s.width
            s.width, s.height = Inches(6.3), int(s.height * ratio)
    d.core_properties.title = "온디바이스 다중 신호 사진 인증과 하이브리드 추천을 적용한 여행 미션 게이미피케이션 앱"
    d.core_properties.author = "김가연, 강규린"
    d.save(DOCS / "report.docx")
    print("저장:", DOCS / "report.docx")


if __name__ == "__main__":
    main()
