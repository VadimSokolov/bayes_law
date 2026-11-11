# Appendix B. Code for the Main Calculations

<!-- cite:begin -->
From *Bayes and Law: Evidence, Expected Utility and Machines in the Courtroom* by Nick Polson and Vadim Sokolov (2026), appendix B. Cite as Polson and Sokolov (2026), *Bayes and Law: Evidence, Expected Utility and Machines in the Courtroom*, edition 1, https://doi.org/10.5281/zenodo.24138404. Book page: https://vsokolov.org/books/bayes_law/. Source: https://github.com/VadimSokolov/bayes_law.
<!-- cite:end -->

This appendix lists a short Python program that reproduces the principal calculations in the book, using only NumPy and SciPy, together with the output it prints and notes on each function. The functions cover the Simpson calculation of the probability that a battered woman is killed by her partner given that she was battered and killed, the island problem with a uniform prior and the Balding and Donnelly sum, decision thresholds, the Sellke, Bayarri and Berger bound on the Bayes factor for a p value, the Bayes factor of Lindley's paradox, the Castaneda v Partida grand jury calculation, Merton's default probability and the precision weighted combination of two normal signals in the solvency example, and the proportionate payment under Schedule 1 of the Insurance Act 2015. The appendix closes with suggested projects, such as beta binomial Bayes factors and a Bayesian network for the Sacco case. Readers change the inputs and see how conclusions move, and use the code to check the worked solutions.

## From the literature

the island problem (Balding and Donnelly 1995); Simpson and the Dershowitz fallacy; the Sellke, Bayarri and Berger (2001) calibration of p values; Lindley's paradox (Lindley 1957); Castaneda v Partida, 430 U.S. 482 (1977); Merton's (1974) structural model of default; Insurance Act 2015 Schedule 1.

## New results

<!-- results:begin -->
No new results of this appendix are catalogued.
<!-- results:end -->

## Key terms

Python, NumPy, SciPy, code, reproducible calculations, simpson function, island formula, threshold, Bayes factor bound, Lindley paradox, Castaneda, Merton model, probability of default, proportionate payment, Insurance Act 2015, Sacco Bayesian network
