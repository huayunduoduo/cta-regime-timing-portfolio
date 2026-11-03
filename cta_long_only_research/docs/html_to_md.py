# -*- coding: utf-8 -*-
"""把 docs/ 下两份导出的 HTML 完整转换为 Markdown（保留全部可见内容）。

用法：在 cta_long_only_research/ 目录下
    python docs/html_to_md.py
输出：
    docs/股债金量化项目研究讨论.md
    docs/report.md
转换是结构化的（标题/段落/表格/代码/折叠区），不是抓正文截断。
"""
from pathlib import Path
from bs4 import BeautifulSoup, NavigableString, Tag

HERE = Path(__file__).parent

JOBS = [
    ("股债金量化项目研究讨论.html", "股债金量化项目研究讨论.md"),
    ("report_wechat.html", "report.md"),
]


def inline(node) -> str:
    """行内元素 → Markdown 行内文本。"""
    if isinstance(node, NavigableString):
        return str(node)
    if not isinstance(node, Tag):
        return ""
    name = node.name
    if name in ("strong", "b"):
        inner = "".join(inline(c) for c in node.children).strip()
        return f"**{inner}**" if inner else ""
    if name in ("em", "i"):
        inner = "".join(inline(c) for c in node.children).strip()
        return f"*{inner}*" if inner else ""
    if name == "code":
        return f"`{node.get_text()}`"
    if name == "br":
        return "  \n"
    if name == "a":
        href = node.get("href", "")
        txt = node.get_text().strip()
        return f"[{txt}]({href})" if txt else ""
    return "".join(inline(c) for c in node.children)


def block(node, out: list, depth_h: int = 2):
    """块级元素 → Markdown 块。depth_h: 标题起始层级。"""
    if isinstance(node, NavigableString):
        t = str(node).strip()
        if t:
            out.append(t)
        return
    if not isinstance(node, Tag):
        return
    name = node.name

    if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        level = int(name[1]) + depth_h - 2
        level = max(2, min(level, 6))
        txt = node.get_text(" ", strip=True)
        if txt:
            out.append("\n" + "#" * level + " " + txt + "\n")
        return

    if name == "p":
        txt = "".join(inline(c) for c in node.children).strip()
        if txt:
            out.append("\n" + txt + "\n")
        return

    if name == "pre":
        code = node.get_text().strip("\n")
        lang = "text"
        cls = (node.code.get("class") if node.code else None) or node.get("class") or []
        for c in cls:
            if c.startswith("language-"):
                lang = c[9:]
        out.append(f"\n```{lang}\n{code}\n```\n")
        return

    if name == "table":
        rows = []
        for tr in node.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            rows.append(["".join(inline(c) for c in cell.children).strip().replace("|", "\\|")
                         for cell in cells])
        if rows:
            ncol = max(len(r) for r in rows)
            rows = [r + [""] * (ncol - len(r)) for r in rows]
            out.append("")
            out.append("| " + " | ".join(rows[0]) + " |")
            out.append("|" + "---|" * ncol)
            for r in rows[1:]:
                out.append("| " + " | ".join(r) + " |")
            out.append("")
        return

    if name in ("ul", "ol"):
        for i, li in enumerate(node.find_all("li", recursive=False), 1):
            txt = "".join(inline(c) for c in li.children).strip()
            bullet = f"{i}." if name == "ol" else "-"
            if txt:
                out.append(f"{bullet} {txt}")
        out.append("")
        return

    if name == "details":
        summary = node.find("summary")
        stxt = summary.get_text(" ", strip=True) if summary else "展开内容"
        out.append("\n**▶ " + stxt + "**\n")
        for c in node.children:
            if isinstance(c, Tag) and c.name == "summary":
                continue
            block(c, out, depth_h)
        return

    if name in ("script", "style", "head", "summary"):
        return

    # div/main/section/body 等容器：递归
    for c in node.children:
        block(c, out, depth_h)


def convert(src: Path, dst: Path, title_fallback: str):
    soup = BeautifulSoup(src.read_text(encoding="utf-8"), "html.parser")
    out = []
    block(soup.body or soup, out)
    md = "\n".join(out)
    # 压缩 3+ 连续空行
    import re
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
    md = f"<!-- 由 {src.name} 经 docs/html_to_md.py 结构化转换，内容未删改 -->\n\n# {title_fallback}\n\n" + md
    dst.write_text(md, encoding="utf-8")
    print(f"{src.name} -> {dst.name}  {len(md):,} 字符")


if __name__ == "__main__":
    for s, d in JOBS:
        convert(HERE / s, HERE / d, d[:-3])
