# delay_intervening_events_studies
 Preregistrations, data analysis, and online supplements, for the manuscript "The Roles of Delays and Intervening Events in Causal Learning"

## Links to Preregistrations and Notes about Deviations from Preregistrations
### General Comments
 - The comparisons between the short vs. long timeframe (Exp. 1a 1b, and Exp 2a vs. 2b) were not preregistered.
 - In the preregistrations we did not explain how we would handle multiple comparisons. For all the studies we used the Benjamini-Hochberg procedure to control for the false discovery rate. Raw p-values are also available in Online Supplement 1. 
 - For all experiments, we conducted both parametric and non-parametric versions of the analyses. We pre-registered in Experiment 1b, 2a and 2b that we would run non-parametric tests to test the influence of delays and intervening events. However, it turned out the results were very similar in both parametric and non-parametric tests. To keep it consistent with other regression analysis and Bayesian analysis which both are parametric, we decided to report the p values from the parametric results in the main text and put all non-parametric results Online Supplement 1.
### Exp. 1a
 - This experiment was not preregistered
### [Registration for Exp. 1b](http://osf.io/yde9v)
 - Participants were also asked to select which cause they saw at 9 AM, 11 AM, and 1 PM. This question was designed as an attention check as we thought it would be easy. However, it became apparent that this question was very hard for participants, and we did not use it as an attention check.
 - In the preregistration, we intended to analyze the veridical and non-veridical relationships separately. However, we decided to use difference scores to capture participants’ ability to discriminate between veridical and non-veridical relations, which cut down the number of analyses in half. The separate analysis for the veridical and non-veridical relationships were reported on Github in the exp1b_analysis.Rmd file, under the "Other analyses in registration (not included in the manuscript)" section.
 - The registration also mentioned an analysis of variability of delay. Because we did not find a main simle effect of delay, we did not think it made sense to further investigate a more subtle way that delay could impact learning, so we did not run this analysis.
### [Registration for Exp. 2a](https://osf.io/3jbdp)
 - In the preregistration we said that we would use difference (veridical minus non-veridical) scores because we had not realized that we had to analyze them separately at that time.
 - Related to the prior point, for the test of successful learning, the analysis was changed to test the judgments against 0 instead of comparing the veridical vs. nonveridical judgments.
 - For the analysis of the impact of delay, we said that we would first calculate the actual average delay rather than the delays calculated from the order events (delays of 1, 3, 9, or 11 events). Because we said that we would run Spearman correlations, only taking the ordinal reltions between these events would result in the same analysis either way.
### [Registration for Exp. 2b](https://osf.io/q6upm)
 - For the tests of successful learning, in addition to comparing the veridical judgments against 0, we also said tht we would compare veridical minus non-veridical. We dropped this second analysis for the same reason as in Exp. 2a.
 - We preregistered that we would run Wilcoxon tests to compare delays of 1 vs. 3, 3 vs. 9, and 9 vs. 11 timesteps at each time point, but we decided to report Spearman correlations instead to cut down the number of tests. The Wilcoxon tests are available in Online Supplement 1. None of the 48 tests were significant after adjusting for multiple comparisons.


## Other studies that were run on this line of research but not reported in this manuscript
- There was a different version of Exp. 2a that we [preregistered](https://osf.io/xdyrm) and ran. Like the reported studies, there was no impact of intervening events or delay. However, this study was very similar to experiment 1a, and we came to be concerned that this study had a number of methodological weaknesses, which are discussed in the manuscript in the introduction to Exp. 2a. We decided to design a better version. Thus, we did not end up reporting the results of this other version of Exp. 2a as it did not add anything.
- We also ran another study about how intervening events may differ when learning about common effect structures (3 causes and 1 effect) vs. common cause structures (1 cause and 3 effects). The findings from this study agree with the studies we reported in that no impact of the number of intervening events was found. However, because the main motivations and findings are about the role of causal structure, this study ended up being too far away from the other studies that were included in the manuscript. The various parts of this study are available as follows:
   -  Here is the registration: (https://osf.io/q9mpt)
   -  The results are reported here: Zhang, Y. (2025). [The Roles of Delays and Intervening Events in Causal Learning](https://pitt.idm.oclc.org/login?url=https://www.proquest.com/dissertations-theses/roles-delays-intervening-events-causal-learning/docview/3201334141/se-2) (Order No. 31848591). Available from Dissertations & Theses @ University of Pittsburgh; ProQuest Dissertations & Theses Global. (3201334141). 
   -  data and code: [Github Repository](https://github.com/yiwenzhang1997/dissertation-experiemnt3-analysis-public.git)


## Acknowledgements
This work was supported by [NSF grant 1651330](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1651330). 
