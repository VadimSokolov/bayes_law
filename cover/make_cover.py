"""Build the KDP wraparound cover and the front cover for Bayes and Law.
Layout follows the Bayesian Causality cover: title on one line in white block
type at the top, the painting as a full width band across the front, author
names in white block type below the band, no series strip.
Usage: python3 make_cover.py PAINTING.jpg PAGES"""
import sys, subprocess, pathlib
from PIL import Image

painting = sys.argv[1] if len(sys.argv) > 1 else "prisoners_band.jpg"
pages = int(sys.argv[2]) if len(sys.argv) > 2 else 436
spine = round(pages * 0.002252, 4)            # KDP white paper, inches
bleed, trim_w, trim_h = 0.125, 7.0, 10.0      # standing KDP trim 7 x 10 in
W = 2 * bleed + 2 * trim_w + spine
H = 2 * bleed + trim_h

# full width band across the front: trim width plus outer bleed
band_w, band_h = trim_w + bleed, round(4.40 * trim_w / 6.0, 3)
band_top = 2.75                                # inches from top edge of the wrap
im = Image.open(painting).convert("RGB")
target = band_w / band_h
crop_h = int(im.width / target)
top = int(0.262 * im.height)                   # keep the windows, the courtyard and the ring of prisoners
top = min(top, im.height - crop_h)
band = im.crop((0, top, im.width, top + crop_h))
band = band.resize((int(band_w * 300), int(band_h * 300)), Image.LANCZOS)
band.save("band.jpg", quality=95, dpi=(300, 300))

tex = r"""\documentclass{article}
\usepackage{fontspec}
\setmainfont{texgyreheros}[Extension=.otf, UprightFont=*-regular, BoldFont=*-bold, ItalicFont=*-italic, BoldItalicFont=*-bolditalic]
\usepackage[paperwidth=@W@in,paperheight=@H@in,margin=0in]{geometry}
\usepackage{tikz}
\usepackage{graphicx}
\pagestyle{empty}
\hyphenpenalty=10000
\exhyphenpenalty=10000
\definecolor{bg}{RGB}{48,50,104}
\begin{document}
\noindent\begin{tikzpicture}[x=1in,y=1in]
\fill[bg] (0,0) rectangle (@W@,@H@);
% front panel starts at the spine fold
\node[anchor=north west, inner sep=0pt] at (@FX@,@H@-@BT@) {\includegraphics[width=@BW@in,height=@BH@in]{band.jpg}};
% title on one line, white block type
\node[white, anchor=north, font=\fontsize{52}{56}\selectfont\bfseries] at (@FX@+@CX@,@H@-0.70) {Bayes and Law};
\node[white, anchor=north, font=\fontsize{15}{18}\selectfont] at (@FX@+@CX@,@H@-1.75) {Evidence, Expected Utility and Machines in the Courtroom};
% author names side by side, white block type
\node[white, anchor=north, font=\fontsize{22}{26}\selectfont\bfseries] at (@FX@+@CX@,@H@-@NAMES@) {Nick Polson \hspace{0.55in} Vadim Sokolov};
% spine
\node[white, rotate=-90, font=\fontsize{18}{20}\selectfont\bfseries] at (@SX@+@HS@,@H@/2+1.2) {Bayes and Law};
\node[white, rotate=-90, font=\fontsize{11}{13}\selectfont\bfseries] at (@SX@+@HS@,1.9) {Polson \quad Sokolov};
% back cover
\node[white, anchor=north west, text width=5.5in, align=flush left, font=\fontsize{11}{15}\selectfont] at (@BX@,@H@-1.0) {%
Every trial is an exercise in reasoning under uncertainty. A jury weighs a smear of DNA against an alibi; a commercial judge asks what an insurer would have done had it been told the truth; an overloaded court must decide which of millions of cases to hear first. Probability is the language of the first problem and expected utility the language of the second. Bayes' rule joins them.\par\medskip
\emph{Bayes and Law} is a casebook for lawyers, statisticians and policy makers. It works through the classic problems, from the island problem and the O.~J. Simpson battering argument to \emph{People v Collins}, \emph{R v Adams} and Sally Clark; reads English tort, misrepresentation and contract law as Bayesian constraints, with the collapse of Greensill Capital as a case study; turns stories and pleadings into generative models; and draws on the work of Lindley, Fienberg, Kadane, Dawid, Mortera, Smith, Balding and Donnelly. It compares English and French law, tests the legal origins thesis in Africa, models India's court backlog as a queue, follows Gould into Corsica and Gambetta into Sicily, and applies Becker and Murphy to costs and deterrence.\par\medskip
The final part asks Lindley's question: should machines decide? It explains how Lean style verification, explainable AI and Bayesian standards of calibration can make legal decisions checkable, and why a court that weighs odds rather than opinions serves justice better.};
\node[white, anchor=north west, text width=5.5in, align=flush left, font=\fontsize{10}{13}\selectfont] at (@BX@,2.75) {%
\textbf{Nick Polson} is the Robert Law Jr.\ Professor of Econometrics and Statistics at the University of Chicago Booth School of Business.\par\smallskip
\textbf{Vadim Sokolov} is a member of the Department of Systems Engineering and Operations Research at George Mason University.};
\end{tikzpicture}
\end{document}
"""
fx = bleed + trim_w + spine
sx = bleed + trim_w
names_top = band_top + band_h + 0.70
subs = {"@W@": f"{W:.4f}", "@H@": f"{H:.4f}", "@FX@": f"{fx:.4f}", "@SX@": f"{sx:.4f}",
        "@HS@": f"{spine / 2:.4f}", "@BX@": f"{bleed + 0.65:.4f}", "@BT@": f"{band_top:.4f}",
        "@BW@": f"{band_w:.4f}", "@BH@": f"{band_h:.4f}", "@NAMES@": f"{names_top:.4f}", "@CX@": f"{trim_w / 2:.4f}"}
for k, v in subs.items():
    tex = tex.replace(k, v)
pathlib.Path("cover_wrap.tex").write_text(tex)
subprocess.run(["xelatex", "-interaction=nonstopmode", "cover_wrap.tex"], check=False, capture_output=True)
subprocess.run(["pdftoppm", "-r", "300", "-png", "cover_wrap.pdf", "cover_wrap_300dpi"], check=True)
full = Image.open("cover_wrap_300dpi-1.png")
full.crop((int(round(fx * 300)), 0, full.width, full.height)).save("front_cover.png", dpi=(300, 300))
full.convert("RGB").save("cover_wrap_300dpi.jpg", quality=95, dpi=(300, 300))
pathlib.Path("cover_wrap_300dpi-1.png").unlink()
print(f"spine {spine} in, wrap {W:.4f} x {H:.4f} in, band {band_w} x {band_h} in")
