# Chapter 19. The Pleadings Game and Generative Bayesian Computation

<!-- cite:begin -->
From *Bayes and Law: Evidence, Expected Utility and Machines in the Courtroom* by Nick Polson and Vadim Sokolov (2026), chapter 19. Cite as Polson and Sokolov (2026), *Bayes and Law: Evidence, Expected Utility and Machines in the Courtroom*, edition 1, https://doi.org/10.5281/zenodo.24138404. Book page: https://vsokolov.org/books/bayes_law/. Source: https://github.com/VadimSokolov/bayes_law.
<!-- cite:end -->

How can pleadings, issue formation and the plausibility standard be modelled probabilistically, and can a neural network be trained to assess a pleaded claim? The chapter starts from Thomas Gordon's Pleadings Game, a dialogue game that turns assertions, denials and defeasible arguments into a set of issues, and adds probabilities by treating each issue as resolved for the claimant or the defendant and the claim as succeeding when the issues satisfy the elements of the cause of action. It reads the US plausibility standard of Bell Atlantic v Twombly and Ashcroft v Iqbal, in contrast with Conley v Gibson, as an implicit posterior threshold, and justifies a threshold by the cost of discovery, p greater than c over V. It compares English strike out and summary judgment under the Civil Procedure Rules (Swain v Hillman, Three Rivers District Council v Bank of England (No 3)). The central method is generative Bayesian computation: simulate disputes from a generative model, train a quantile neural network on parameter and data pairs, and obtain posterior quantiles for a new case by amortised, likelihood free inference. A Hedley Byrne negligent misstatement example illustrates the pleading exchange, and the chapter discusses case law as training data, selection bias, and the limits of a machine that cannot decide what the law is.

## From the literature

The Pleadings Game and procedural justice (Gordon 1995). Plausibility pleading: Bell Atlantic Corp v Twombly (2007) and Ashcroft v Iqbal (2009), replacing the Conley v Gibson (1957) no set of facts formula. English strike out and summary judgment: Swain v Hillman (2001); Three Rivers (No 3) (2001). Hedley Byrne negligent misstatement. Approximate Bayesian computation and likelihood free inference, and quantile regression by minimising the quantile loss. The authors' own proposals are the Pleadings Game paper (Polson and Sokolov 2026) and the generative Bayesian computation method (Polson and Sokolov 2025), which the chapter presents as its method.

## New results

<!-- results:begin -->
No new results of this chapter are catalogued.
<!-- results:end -->

## Key terms

Pleadings Game, Thomas Gordon, dialogue game, issues, admissions, denials, defeasible rules, Rule 8, Twombly, Iqbal, Conley v Gibson, plausibility pleading, motion to dismiss, strike out, summary judgment, real prospect of success, Three Rivers, BCCI, misfeasance in public office, cost of discovery, posterior threshold, generative Bayesian computation, GBC, quantile neural network, quantile loss, approximate Bayesian computation, likelihood free inference, amortisation, simulator, litigation analytics, case law as training data, selection bias, Hedley Byrne, negligent misstatement, procedural justice
