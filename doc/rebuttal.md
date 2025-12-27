# Response to Reviewers

We thank the reviewers for their insightful and constructive comments, which have helped us significantly improve the quality and clarity of our manuscript. Below, we provide a point-by-point response to each comment.

---

## Reviewer #1

> **Comment 1:** It would be good if the authors analyze their train/validation/test separation by the novelty of the binding pocket. For example, the PLINDER approach designed to understand memorization vs generalization of AF type approaches for ligand binding, classifies pockets by novelty, and can be used by the authors. It would be interesting to see whether Holo4k and Coach420 contain entirely novel pockets compared with the rest of the dataset, and how well the method performs on those. It could happen that the performance will drop, as is true for most existing methods, but it would still be informative to report.

**Response:**
We thank the reviewer for this excellent suggestion proposed to disentangle memorization from generalization. We have fundamentally shifted our training and evaluation pipeline to the **Plinder dataset (v2024-06)** and conducted the suggested novelty analysis:

1.  **Holo4k Analysis**: We rigorously screened the 4,543 systems in the Holo4k dataset against the Plinder splits. Based on this, we constructed a specific test set (`test340`) containing 340 systems, comprising:
    *   **300 systems** that are completely absent from the Plinder dataset (train/val/test), representing "novel" pockets relative to our training distribution.
    *   **40 systems** that overlap with the Plinder test set, included for cross-validation consistency.
    As shown in **Figure 2c**, YuelPocket maintains high performance (Top-1 success rate ~55%) on this Holo4k set. Since 88% of this test set consists of completely novel systems, these results provide strong evidence of the model's generalization capability rather than simple memorization.

2.  **Coach420 Analysis**: Our overlap analysis revealed that the vast majority of systems in the Coach420 dataset were already present in the Plinder *training* split. After strictly filtering out these overlapping examples to avoid testing on "memorized" data, the remaining number of unique systems was insufficient to constitute a statistically significant benchmark. Consequently, we excluded Coach420 from this study to ensuring our reported metrics reflect true generalization performance.

> **Comment 2:** It would also be useful to understand how sensitive the method (both the residue prediction part and the pocket prediction using FFT and multiple probes) is to the input structural coordinates. I would suggest exploring both experimentally determined crystal structures and predicted models (e.g., from AlphaFold-type methods). I fully understand this analysis might be limited in size (for separately crystallized structures), but it could be insightful. For AlphaFold-type models, it would be better to choose post-2023 structures not similar to known ones, based on PLINDER as mentioned above.

**Response:**
We strongly agree with the reviewer on the importance of evaluating robustness on predicted structures. We have added a comprehensive new section, **"Robustness on AlphaFold Models"** (Figure 4), where we evaluated YuelPocket on AlphaFold-generated structures for the 1,036 systems in our test set.

Our results demonstrate that:
1.  **Residue-Level Accuracy**: YuelPocket achieves a Top-10 success rate exceeding 90% on AlphaFold structures (Figure 4a).
2.  **Comparison**: YuelPocket consistently surpasses P2Rank on AlphaFold structures in both DCA (Distance to Closest Atom) and DCC (Distance Center-to-Center) metrics (Figure 4b-c).
3.  **Resilience**: Performance remains stable even as the RMSD between the predicted and experimental backbone increases (up to 4 Å), indicating high tolerance to structural noise (Figure 4d-e).
4.  **Consistency**: There is a high degree of overlap (~61%) between successful predictions on ground truth and AlphaFold structures (Figure 4f).

---

## Reviewer #2

**General Response:** We thank the reviewer for their critical assessment. We have addressed the concerns regarding metrics, comparisons, and reproducibility by retraining on the Plinder dataset, adopting standard distance-based metrics (DCA, DCC), and performing direct comparisons with P2Rank.

> **Comment 1:** The authors assume that the use of virtual joint node connecting all protein residues and small molecule atoms to capture long-range interactions. They eve go as far as to claim that their study design “enables the model to capture the synergistic nature of molecular recognition”. The argument about computational efficiency of YuelPocket is well taken but it remains unclear how this representation truly captures "the synergistic nature of molecular recognition". If this is only because the virtual point connects all protein and ligand atoms, by itself it is not a proof of synergy or rigor. Notably, the claim of capturing long-range interactions appears as a statement in the abstract only, with no subsequent description of studies proving this point. The authors should either demonstrate explicitly that they actually capture long range interactions (which is a well-defined concept in molecular modeling) or remove this unsubstantiated claim.

**Response:**
We accept the reviewer’s feedback. We have revised the manuscript to focus on the **computational efficiency** provided by the virtual node (reducing complexity from quadratic to linear) and its role as a global information aggregator. We have removed speculative claims about "synergistic molecular recognition" that were not explicitly validated. The text now emphasizes that the architecture enables global connectivity, which is empirically shown to improve pocket prediction performance.

> **Comment 2:** Even bigger issue is the claim of “exceptional performance” of the algorithm that is apparently based on the reported AUC-ROC values of 0.85 and 0.89 on two benchmark datasets as well as “exceptional specificity (0.85) and NPV (0.95)” [NB: for what external set?]. Notably, these values relate, in the authors’ own words, to “the ability of YuelPocket to accurately identify non-binding regions while minimizing false positives”. *Non*-binding regions, *not* binding pockets! This is a prime example of mis-representing the meaning of the reported statistics in a study with the prime objective (and claims) to predict accurately the minor class of binding pocket residues, not non-binding regions.

**Response:**
We fully accept the reviewer’s criticism regarding the potential for AUC-ROC and specificity to be misleading in highly imbalanced datasets. In the revised manuscript, we have addressed this by fundamentally changing our evaluation strategy:

1.  **Adoption of Success Rate Metrics**: We have shifted our primary evaluation metric to **Success Rate based on spatial distance**. Unlike AUC-ROC, which can be inflated by the vast number of non-binding residues (true negatives), Success Rate directly measures the practical utility of the model: does the top-ranked prediction fall within the binding pocket? This metric is robust to class imbalance and provides a transparent assessment of the model's ability to identify the minority class.

2.  **Alignment with Field Standards**: We adopted this distance-based evaluation specifically because it is the standard method used by leading tools in the field, including **P2Rank**. By strictly following P2Rank's evaluation protocols—specifically Distance to Closest Atom (DCA) and Distance Center-to-Center (DCC)—we ensure our results are directly comparable and scientifically rigorous.

3.  **Dual-Mode Evaluation**: Our revised manuscript explicitly distinguishes between two prediction modes, both evaluated rigorously:
    *   **Residue-Level Mode**: Evaluated by checking if the top-ranked residues are within 4 Å of the ligand.
    *   **Coordinate-Level Mode**: Evaluated using the **exact same** DCA and DCC metrics as P2Rank. As shown in **Figure 3**, YuelPocket (Coordinate Mode) achieves ~10% higher success rates than P2Rank on these identical metrics, demonstrating superior performance in precise binding site localization without relying on misleading statistics.

> **Comment 3:** More to the same point of mis-representing the meaning of the observed statistical model
accuracy, the authors report but de-emphasize the importance of PR AUC values of 0.49
and 0.46, which is the only accuracy they should have reported for this highly imbalanced
dataset where the goal is to accurately predict the minor class. It appears from the author’s
assessment of the PR AUC baseline (called by the authors for some reason “random
guessing”) of 0.053 that the imbalance in the training set between binding pocket residues
and those outside of the binding pockets is about 1:20. It is well known from the standard
statistical literature (see, for instance, https://arize.com/blog/what-is-pr-auc/) that AUC
ROC metrics show inflated values for the highly imbalanced datasets and that PR AUC
metrics must be used to assess the model performance. In this regard, PR AUC values of
under 0.5 are good but far from great (and again, in this case of predicting pocket residues,

not the rest of the protein these are the only ones that matter) as these simply mean that
YuelPocket recovers slightly less than 50% of pocket residues correctly and of those it
predicts as positives less than a half are true positives. Of note, in an apparent attempt to
inflate the perception of the model performance, the authors compare the PR AUC values
to the “random guessing” of 0.053 but chose not to compare their AUC ROC values to the
similarly defined random guessing of non-pocket residues of 0.947 (meaning that their,
notably irrelevant to the point of the paper, prediction of the larger class is close to random
expectation).

**Response:**
We appreciate the reviewer's rigorous statistical insight. We fully agree that for highly imbalanced datasets like binding site prediction (~1:20 ratio), PR-AUC is a far more meaningful metric than AUC-ROC, and that comparisons to "random guessing" on ROC curves can differ from practical utility.

In response, we have made the following major revisions:
1.  **Removed Misleading Comparisons**: We have deleted the comparisons to "random guessing" baselines for ROC-AUC to avoid creating any false impression of inflated performance.
2.  **Transparent Reporting**: We report the PR-AUC values (0.49/0.46) openly. We acknowledge these figures reflect the inherent difficulty of per-residue classification on the entire protein surface.
3.  **Focus on Practical Utility (Top-N)**: While PR-AUC captures the trade-off across the *entire* dataset, practical users are primarily interested in the **top-ranked predictions**. A model with a moderate PR-AUC can still be exceptionally useful if it reliably ranks binding residues at the very top. Indeed, our results show that:
    *   **Residue Mode**: The **Top-10 Success Rate** (likelihood that at least one of the top 10 predicted residues is a binding site) exceeds **90%** on the test set.
    *   **Coordinate Mode**: Our Top-1/Top-3 center predictions outperform P2Rank.
    This demonstrates that while the model may not perfectly classify every ambiguous boundary residue (lowering PR-AUC), it is **highly effective** at its core task: reliably pointing researchers to the correct binding site.

> **Comment 4:** The discussion in the next to last paragraph before the “Binding Site Clustering and Center
Prediction” section about actual precision accuracy of the model with and without
assuming that all predicted binding residues are actual binding residues is very confusing
appearing as a good example of providing a circular argument. Putative (predicted) biding
sites cannot be regarded as true positives so whereas it is accurate to count such residues
as false positives, one cannot state that such residues artificially reduce precision metrics
because they must be regarded as true positives. This paragraph needs to be revised to
avoid positioning wishful thinking as ground truth.

**Response:**
We fully accept the reviewer's criticism. We agree that labeling predicted sites as "potential true positives" without experimental validation constitutes circular reasoning and "wishful thinking" that has no place in rigorous quantitative evaluation.

In the revised manuscript, we have:
1.  **Deleted the Problematic Paragraph**: We have completely removed the discussion that attempted to justify false positives as potential uncharacterized binding sites.
2.  **Strict Ground Truth Adherence**: All reported metrics (Precision, Recall, Success Rates) are now calculated **strictly** against the experimental ground truth (ligands defined in the dataset). We treat any prediction not matching the ground truth as a False Positive, ensuring our evaluation is conservative and scientifically sound.

> **Comment 5:** The discussion in the next paragraph should be revised as well. For as long as we agree
that the only objective of YuelPocket is the prediction of binding pocket residues and
therefore for this imbalanced dataset, precision and recall are the only meaningful metrics,
any discussion of metrics such as NPV is irrelevant. Further, the authors base their
conclusion about relative performance of the two threshold strategies by comparing their
performance using external sets. Strictly speaking, this approach is not rigorous as the
expectations for the external sets should be made strictly using training sets and then
assessed (but not stated) using test sets. The authors should provide comparative
performance of the two strategies for the training set and then see if their observations hold
true.

**Response:**
We agree with the reviewer that NPV is not a primary metric of interest for this imbalanced problem and that optimizing thresholds on the test set is methodologically unsound.

To address this, we have streamlined the manuscript by:
1.  **Removing Irrelevant Metrics**: We have removed NPV and Specificity from our primary performance claims, focusing instead on Precision, Recall, and Success Rates.
2.  **Removing Threshold Strategy Discussion**: We have excised the entire section comparing "Fixed vs. Adaptive Thresholds." As the reviewer correctly implied, binary classification thresholds can be arbitrary.
3.  **Adopting Threshold-Independent Evaluation**: Instead of debating the optimal cut-off, we now quantify performance using **Top-N Ranking metrics** (e.g., Top-10 Residue Success Rate, Top-N Pocket Centers). This approach evaluates the model's ability to correctly *prioritize* residues/pockets, which is the most critical factor for real-world applications and avoids the pitfalls of threshold tuning.

> **Comment 6:** First sentence in the section “Binding Site Clustering and Center Prediction”: “While our
previous results demonstrate that YuelPocket accurately predicts individual residues within
6Å of ligand atoms”: what is the ground for this statement about YuelPocket accuracy?

**Response:**
We have revised this sentence to be grounded in specific quantitative data. We no longer make vague claims of accuracy. Instead, we explicitly reference our **Top-10 Residue Success Rate**, which exceeds **90%** on the test set. This metric provides a concrete, empirically verifiable basis for stating that the model effectively identifies residue-level binding signals, which then serve as the foundation for the subsequent coordinate-level clustering.

> **Comment 7:** “In conclusion, YuelPocket demonstrates effective small molecule binding site prediction
with AUC-ROC values of 0.85-0.89 on benchmark datasets”: this statement is inaccurate
as explained above. This statistics is interpreted by the authors as "indicating the ability of
YuelPocket to accurately identify non-binding regions while minimizing false positives".
The challenge of predicting non-binding residues is very different from that of predicting
binding site residues: this performance is characterized by PR-ROC, which is actually low,
less than 0.5. Please be accurate in interpreting your own data and in using consistent
language.

**Response:**
We fully agree with the reviewer that referencing AUC-ROC in this context was misleading. We have completely rewritten the Conclusion to ensure accurate interpretation:
1.  **Removed AUC-ROC Claims**: We no longer cite AUC-ROC values as primary evidence of "effective binding site prediction."
2.  **Focus on Relevant Metrics**: The conclusion now summarizes YuelPocket's performance using **Distance-based Success Rates (DCA/DCC)** and **Top-N Accuracies**. These metrics directly measure the model's success in finding binding sites (the minority class) and are not inflated by the large number of non-binding residues.
3.  **Balanced Tone**: We present a balanced view, acknowledging the challenges of high-precision residue classification (as reflected in PR-AUC) while highlighting the model's strong practical utility in ranking and locating pockets.

> **Comment 8:** There is some confusion about how the authors formed the protein graph based on specific
distance threshold. They state that “edges between protein residues are established based
on spatial proximity, typically connecting residues within 8 Å of each other” (section
Graph Construction). Yet, in a different place, there is a different definition of contacts
based on 4.1 A threshold of the distance between Ca: so how the distance of 8A between
residues is calculated in this case?

**Response:**
We apologize for the confusion. We have revised the Methods section to explicitly distinguish between these two parameters, which serve completely different purposes:

1.  **Graph Construction Threshold (8.0 Å)**: This defines the *connectivity* of the protein graph. An edge is created between two residue nodes if the distance between their **$C\alpha$ atoms** is less than 8.0 Å. This ensures that the GNN captures the local spatial environment and geometry of the protein surface.
2.  **Label Definition Threshold (6.0 Å)**: This defines the *supervised learning targets*. A residue is labeled as a "binding site residue" (positive class) if *any* of its heavy atoms are within 6.0 Å of minimal distance to any ligand heavy atom.

These are distinct thresholds: one determines network topology (inputs), and the other determines ground-truth classification (outputs).

> **Comment 9:** Table S2: please include a comment on which benchmark dataset this performance is
reported for. Following the same format as Table 3 in P2Rank paper cited in your
manuscript would be extremely helpful. The authors should also provide an honest

statement about the comparative performance of YuelPocket using similar metrics on the
same external sets, which is not currently reported. Indeed, Table 3 in the P2Rank paper
reports method performance at 4A threshold whereas Table S2 shows YuelPocket data for
5A threshold. Furthermore, sources of data reported for alternative methods are not
provided but the numbers reported for Fpocket in Table S2 are substantially lower than
those reported in Table 3 of the P2Rank paper. Finally, Table 3 in the latter paper provides
accuracy metrics for binding site prediction for 7 different tools, and the best performing
one (P2Rank) shows statistics exceeding that of YuelPocket (72% vs 40.6% for Top N,
respectively, and 78.3% vs 66% for Top N+2, respectively, all for 4A threshold and for
COACH420 dataset).

**Response:**
We thank the reviewer for holding us to a high standard of benchmarking fairness. We fully acknowledge that our previous comparison (different thresholds, disparate datasets) was flawed.

To rectify this, we have:
1.  **Discarded Table S2**: We have removed the flawed table entirely.
2.  **Implemented Direct Benchmarking (Figure 3)**: We performed a new, rigorous side-by-side evaluation of YuelPocket vs. P2Rank on the **same datasets** (Plinder Test, 1,036 systems; Holo4k, 340 systems). Note that comparisons on COACH420 were omitted as we excluded that dataset due to training overlap (see Response to Comment 1).
3.  **Standardized Protocols**: Both models were evaluated using the **exact same metrics** (DCA and DCC) at the **same strict threshold (4.0 Å)**.
4.  **Honest Outcome**: Under these strictly identical conditions, YuelPocket outperforms P2Rank, achieving ~10% higher Top-1 Center Success Rates (DCC < 4Å) on both datasets (Figure 3).

This new comparison removes all ambiguity and demonstrates YuelPocket's superiority in a fair, apples-to-apples contest.

> **Comment 10:** As a final comment, the authors build their model using MOAD database. This is an
unfortunate choice because, as should have been acknowledged by the authors, this dataset
was sunset in 2023-24 (https://www.nature.com/articles/s41598-023-29996-w) and is no
longer available making the results non-reproducible by any reader. The authors must
provide the full download of the dataset that they used for model building.

**Response:**
We fully agree that relying on a sunset database is a risk to long-term reproducibility. To address this, we have **completely migrated our training and evaluation pipeline to the Plinder dataset (v2024-06)**.

By shifting to Plinder—a modern, actively maintained, and high-quality dataset—we ensure that our work is built on a stable foundation. We have made our full codebase and data processing scripts publicly available, ensuring that the entire pipeline (from raw Plinder data to final model weights) is fully reproducible by the community.
