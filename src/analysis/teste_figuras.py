"""Verificação das figuras e tabelas do texto — Fase 6, itens 4 e 10.

    python -m src.analysis.teste_figuras

Se aparecer "Figuras e tabelas funcionando corretamente.", os formatadores, as
tabelas LaTeX e as figuras estão verificados. Gera tudo em pasta temporária a
partir de `data/`; se houver `pdflatex`, compila um documento de teste que inclui
todas as tabelas (para pegar erro de sintaxe); sem `pdflatex`, esse passo é pulado.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

from src.analysis import figuras as F
from src.analysis import tabelas as T


def teste_formatadores():
    assert T.num(0.5) == "0,50" and T.num(3.14159, 1) == "3,1"
    assert T.num(0.123, 1, sinal=True) == "+0,1" and T.num(-0.04, 2, sinal=True) == "-0,04"
    assert T.esc("a_b & 50%") == r"a\_b \& 50\%"
    t = T.tabular("lr", ["A", "B"], [["1", "2"], ["3", "4"]], separadores={0})
    assert t.count("\\midrule") == 2 and t.startswith("\\begin{tabular}{lr}")
    assert t.count("\\\\") == 3 and "\\bottomrule" in t
    # separador depois da última linha é ignorado
    assert T.tabular("l", ["A"], [["1"]], separadores={0}).count("\\midrule") == 1


def teste_tabelas_dos_dados():
    with tempfile.TemporaryDirectory() as d:
        arquivos = T.gerar(T.PASTA_DATA, Path(d))
        nomes = sorted(p.name for p in arquivos)
        assert nomes == sorted(["tab_validacao.tex", "tab_veredito.tex", "tab_amplificacao.tex",
                                "tab_concorrencia_efetiva.tex", "tab_excesso_p90.tex",
                                "tab_excesso_p95.tex", "tab_excesso_p99.tex"]), nomes
        for p in arquivos:
            txt = p.read_text(encoding="utf-8")
            assert txt.startswith("\\begin{tabular}") and txt.rstrip().endswith("\\end{tabular}")
            assert "nan" not in txt.lower() and "." not in txt.replace("\\.", "")  # vírgula decimal
        v = (Path(d) / "tab_veredito.tex").read_text(encoding="utf-8")
        assert "68/80" in v and "85,0\\%" in v and v.count("sim") + v.count("não") == 9
        # regeneração idêntica (determinística)
        antes = {p.name: p.read_bytes() for p in arquivos}
        T.gerar(T.PASTA_DATA, Path(d))
        assert antes == {p.name: p.read_bytes() for p in arquivos}


def teste_figuras_dos_dados():
    with tempfile.TemporaryDirectory() as d:
        saidas = F.gerar(F.PASTA_DATA, Path(d))
        assert len(saidas) == 12  # 6 figuras x (pdf, png)
        for p in saidas:
            assert p.exists() and p.stat().st_size > 5000, p
        assert (Path(d) / "fig_violacao_vs_c_p99.pdf").read_bytes()[:4] == b"%PDF"


def teste_compila_latex():
    if shutil.which("pdflatex") is None:
        print("  (pdflatex ausente: compilação pulada)")
        return
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        T.gerar(T.PASTA_DATA, d)
        F.gerar(F.PASTA_DATA, d)
        corpo = "\n".join(
            f"\\begin{{table}}[h]\\centering\\small\\input{{{p.stem}}}\\end{{table}}"
            for p in sorted(d.glob("tab_*.tex"))
        )
        figs = "\n".join(f"\\includegraphics[width=\\textwidth]{{{p.stem}}}" for p in sorted(d.glob("fig_*.pdf")))
        (d / "t.tex").write_text(
            "\\documentclass{article}\\usepackage[utf8]{inputenc}\\usepackage[T1]{fontenc}"
            "\\usepackage{booktabs}\\usepackage{graphicx}"
            "\\usepackage[margin=1.5cm]{geometry}\\begin{document}\n" + corpo + "\n" + figs + "\n\\end{document}\n",
            encoding="utf-8")
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "t.tex"],
                           cwd=d, capture_output=True, text=True, timeout=150)
        assert r.returncode == 0, r.stdout[-1500:]
        assert (d / "t.pdf").stat().st_size > 10000


def main():
    for f in (teste_formatadores, teste_tabelas_dos_dados, teste_figuras_dos_dados, teste_compila_latex):
        f()
        print(f"  ok: {f.__name__}")
    print("Figuras e tabelas funcionando corretamente.")


if __name__ == "__main__":
    main()
