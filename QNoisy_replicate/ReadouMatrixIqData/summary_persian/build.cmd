@echo off
REM Build Persian report (requires XeLaTeX + -shell-escape for minted code snippets)
cd /d "%~dp0"
xelatex -shell-escape -interaction=nonstopmode report_algorithm.tex
xelatex -shell-escape -interaction=nonstopmode report_algorithm.tex
xelatex -shell-escape -interaction=nonstopmode report_algorithm.tex
echo.
echo Done: report_algorithm.pdf
