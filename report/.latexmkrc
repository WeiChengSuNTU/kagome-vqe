# Build: latexmk report.tex  (xelatex + biber; output in output/)
@default_files = ('report.tex');
$pdf_mode = 5;
$bibtex_use = 2;
$out_dir = 'output';
$aux_dir = 'output';
