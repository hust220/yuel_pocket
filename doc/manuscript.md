*^1^Department of Neurology and Neuroscience, University of Virginia,
School of Medicine, Charlottesville, VA, United States.*

*^\*^Corresponding author: Nikolay V. Dokholyan, E-mail:
<dokh@virginia.edu>*

Abstract

Predicting small molecule binding sites on proteins remains a critical challenge in structure-based drug discovery, especially with the increasing reliance on predicted protein structures. While recent advances like AlphaFold have transformed structure prediction, accurately identifying functional ligand-binding pockets on these models remains distinct and challenging. Here, we present YuelPocket, a geometric deep learning framework that treats protein-ligand interactions as a global graph connected via a virtual node, enabling the capture of long-range dependencies with linear complexity. YuelPocket operates in two complementary modes: residue-level prediction for identifying contact residues and coordinate-level prediction for pinpointing precise pocket centers. Trained on the large-scale Plinder dataset, YuelPocket outperforms state-of-the-art methods like P2Rank on independent benchmarks (Plinder and Holo4k), achieving ~10% higher success rates in both Distance to Closest Atom (DCA) and Center-to-Center (DCC) metrics. Crucially, YuelPocket demonstrates high robustness on AlphaFold-predicted structures, maintaining high accuracy even for targets with notable backbone deviations from experimental structures. By combining high precision, dual-mode prediction, and resilience to structural noise, YuelPocket offers a robust solution for binding site discovery in the era of predicted protein structures.

# INTRODUCTION

Small molecule binding site prediction^1^ on proteins represents one of
the most critical challenges in computational biology and drug
discovery, with far-reaching implications for understanding protein
function, rational drug design^2,3^, and structure-based drug discovery
pipelines^4--7^. The accurate identification of binding pockets is
fundamental to virtually every aspect of modern drug development, from
target validation^8^ to lead optimization and clinical candidate
selection.

The pharmaceutical industry faces unprecedented challenges in drug
discovery, with development costs exceeding \$2.6 billion per approved
drug and failure rates nearing 90% in clinical trials^9^. A major factor
contributing to these failures is the difficulty in accurately
predicting protein-small molecule interactions. Traditional drug
discovery methods often depend heavily on experimental techniques, which
are costly and time-consuming. The introduction of AlphaFold2^10^ has
transformed protein structure prediction, achieving high accuracy
in determining three-dimensional protein structures from amino acid
sequences. The release of AlphaFold3^11^ has further enabled researchers
to directly predict protein-small molecule complexes, potentially
significantly speeding up drug discovery. However, despite its
impressive capabilities, AlphaFold3 still encounters notable challenges
in drug discovery. Shen et al.^12^ found that while AlphaFold3 reliably
predicted the overall structure of receptors, its precision in
positioning small molecule ligands was variable and frequently
inaccurate. This inconsistency was especially evident with allosteric
modulators, highlighting a key obstacle in reliably identifying binding
sites.

Existing pocket prediction algorithms can be broadly categorized into
three main approaches, each with significant limitations. Traditional
geometric approaches such as Fpocket^1^, SiteHound^13^, and CASTp^14^
rely on geometric descriptors including surface curvature, solvent
accessibility, and cavity detection algorithms. These methods identify
potential binding sites based on physical properties such as surface
concavity, pocket depth, solvent accessibility, and geometric
complementarity to spherical probes. While these methods offer fast
computation and interpretable results without requiring training data,
they suffer from limited accuracy and poor performance on flat binding
sites or allosteric sites.

More recent machine learning approaches employ various techniques to
improve binding site prediction accuracy. P2Rank^15^ uses a Random
Forest classifier with geometric and evolutionary features to predict
ligand binding sites on a protein's solvent-accessible surface, trained
on the CHEN11^16^ dataset containing 251 proteins with 476 ligands.
DeepSite^17^ utilizes 3D convolutional neural networks that process 16
Å³ protein subgrids with eight feature channels, trained on the scPDB
v.2013 database with 7,622 binding sites. DeepPocket^18^ employs 3D
Convolutional Neural Networks for binding site detection and
segmentation, trained on the scPDB^19^ v.2017 database with 17,594
binding sites from 16,612 proteins. PUResNet^20^ combines U-Net^21^ and
ResNet^22^ architectures to predict binding site probabilities for each
voxel in 3D protein structures. While these methods offer improved
accuracy over geometric approaches, they are limited by computational
intensity and difficulty in capturing long-range interactions.

Emerging graph neural network (GNN)^23^-based approaches provide more
sophisticated modeling of protein structures, but suffer from
fundamental limitations in both architecture and training data scale.
PocketMiner^24^ uses a geometric vector perceptron graph neural network
to predict cryptic pockets, though it was trained on only 37 proteins
from molecular dynamics simulations, severely limiting its
generalization capabilities. SiteRadar^25^ represents protein structures
as graphs where nodes are heavy atoms and edges are interatomic
distances, predicting binding sites on a grid by classifying each grid
point as pocket or non-pocket. LigBind^26^ employs a relation-aware
graph neural network to predict ligand-specific binding residues,
representing each residue and its surrounding structural context (within
15 Å radius) as a graph, with a two-phase transfer learning approach
trained on over 1,000 ligands. While these methods represent advances
over traditional approaches, they share critical architectural
limitations: they construct only local graphs that capture limited
spatial relationships between neighboring elements, fundamentally
missing the global interaction patterns and synergistic binding effects
where distant residues cooperatively contribute to ligand recognition.
Particularly, SiteRadar creates graphs between grid points and their
surrounding protein atoms, while LigBind builds graphs between residues
and their neighboring protein residues within a fixed radius.
Furthermore, these methods do not explicitly incorporate ligand
information during the prediction process, limiting their ability to
predict ligand-specific binding sites, and their training datasets
remain relatively small compared to the diversity of protein-small
molecule interactions in nature.

Here, we introduce YuelPocket, a geometric deep learning framework designed to overcome these limitations. YuelPocket treats protein-ligand interactions as a global graph connected via a virtual node, ensuring efficient capture of long-range dependencies with linear complexity. To address the data scarcity issue inherent in previous GNN approaches, we train YuelPocket on the massive Plinder dataset, which provides a diverse and comprehensive landscape of protein-ligand interactions.

YuelPocket provides a holistic solution by operating in two complementary prediction modes:
1.  **Residue-Level Prediction Mode**: Identifies specific protein residues involved in binding, providing granular contact information essential for understanding interaction mechanisms.
2.  **Coordinate-Level Prediction Mode**: Directly predicts the 3D coordinates of binding pocket centers by ranking Solvent Accessible Surface (SAS) probes, offering precise starting points for downstream tasks like molecular docking.

We rigorously evaluated YuelPocket on the Plinder and Holo4k benchmarks, demonstrating superior performance over state-of-the-art methods like P2Rank. Crucially, recognizing the paradigm shift in structural biology, we assessed YuelPocket's robustness on AlphaFold-predicted structures. Our results show that YuelPocket maintains high accuracy even on targets with significant backbone deviations from experimental structures, establishing it as a robust and versatile tool for large-scale drug discovery in the era of predicted protein families. We further demonstrate its utility through a minimal probe set approach, enabling the comprehensive exploration of binding space across diverse targets.

# Results

## Binding Site Prediction Performance on Benchmark Datasets {#binding-site-prediction-performance-on-benchmark-datasets .Heading2}

We assessed YuelPocket's performance on two rigorous benchmark datasets: a subset of 1,036 systems from the **Plinder** test split and an independent **Holo4k** dataset containing 340 complexes. Instead of relying solely on residue-level classification metrics which can be biased by class imbalance, we adopted a more practically relevant metric: **Success Rate** based on spatial distance (Figure 2a). A prediction is considered successful if the top-N ranked residues (Top-1, Top-3, Top-10) fall within a specified distance threshold (4 Å to 10 Å) of any ligand atom.

On the Plinder test set (Figure 2b), YuelPocket demonstrates high precision. At the strict 4 Å threshold, the Top-1 success rate is approximately 48%, increasing to over 70% for Top-3 and surpassing 90% for Top-10. This indicates that even when the single highest-scoring residue is slightly offset, the model consistently places high-confidence predictions in the immediate vicinity of the binding site.

Performance on the external Holo4k dataset (Figure 2c) further confirms the model's generalization capability. YuelPocket achieves a Top-1 success rate of ~55% at 4 Å, with Top-3 and Top-10 rates reaching ~75% and ~88%, respectively. The consistency of these results across datasets highlights the robustness of our graph-based approach.

## Coordinate-Level Prediction Performance and Comparison with State-of-the-Art {#coordinate-level-prediction-performance-and-comparison-with-state-of-the-art .Heading2}

To benchmark the effectiveness of our coordinate-level prediction mode (`pos_sc3`), we compared YuelPocket against P2Rank, a widely used and highly accurate template-free pocket prediction tool. We evaluated both methods on the Plinder (1,036 systems) and Holo4k (340 systems) datasets using two standard metrics: Distance to Closest Atom (DCA) and Distance Center-to-Center (DCC). A prediction is considered successful if the distance falls below a specified cutoff.

YuelPocket consistently outperforms P2Rank across both datasets and metrics (Figure 3a-d). On the Plinder test set, YuelPocket achieves a Top-1 DCA success rate of ~62% at a 4 Å cutoff (Figure 3a), significantly higher than P2Rank's ~55%. This performance gap is even more pronounced in the DCC metric (Figure 3b), where YuelPocket's Top-1 success rate is ~45% compared to P2Rank's ~40%, indicating that our predicted centers are geometrically closer to the true ligand centroids. Similar trends are observed on the Holo4k dataset (Figure 3c-d), confirming the generalization capability of our approach. Notably, YuelPocket's Top-3 performance (dashed blue lines) often exceeds P2Rank's Top-3 performance, demonstrating a higher probability of capturing the correct site within the top ranks.

The visual comparison on the protein 7T8F (Figure 3e-h) illustrates the qualitative difference between the methods. YuelPocket generates a dense, high-probability point cloud (Figure 3e) that, after clustering (Figure 3f), precisely locates the binding site with the primary cluster (red) and correctly identifies a secondary site (orange). mainly covering the true ligand position. In contrast, P2Rank's predictions for this system are more sparse (Figure 3g-h), identifying the pocket but with less definition and lower confidence. This highlight YuelPocket's ability to provide sharper and more confident binding site localizations.

## Robustness on AlphaFold Models {#robustness-on-alphafold-models .Heading2}

Ideally, pocket prediction methods should be robust not only on high-quality crystal structures but also on predicted structures, which are increasingly common in modern drug discovery pipelines. We evaluated YuelPocket on AlphaFold-generated models for the systems in our test set.

At the residue level (Figure 4a), YuelPocket maintains high accuracy on AlphaFold structures, with a Top-10 success rate exceeding 90% across various distance thresholds. When benchmarked against P2Rank using coordinate-based metrics (DCA and DCC), YuelPocket (blue lines) consistently outperforms P2Rank (red lines) (Figure 4b-c). For instance, in the DCA metric (Figure 4b), YuelPocket's Top-1 performance is approximately 10 percentage points higher than P2Rank's, highlighting our model's superior ability to pinpoint binding sites even on predicted backbones.

We further analyzed the impact of structural quality. The RMSD distribution (Figure 4d) shows that while most AlphaFold models are accurate (< 2 Å RMSD), a significant subset exhibits larger deviations. Crucially, YuelPocket's performance is remarkably stable; the success rate decreases only slightly as RMSD increases from 0 to 4 Å and then plateaus (Figure 4e), indicating resilience to structural noise. This robustness is confirmed by the high consistency between predictions on experimental (Ground Truth, GT) and AlphaFold (AF) structures (Figure 4f), where ~61% of cases are successful on both, and cases where the model fails on AF despite succeeding on GT are relatively rare (~13%).

A visual example of protein 4I4V illustrates this robustness (Figure 4g-i). YuelPocket correctly identifies the binding site on both the experimental structure (Figure 4g) and the AlphaFold model (Figure 4h). Superimposing the structures (Figure 4i) reveals that despite noticeable backbone deviations in the loop regions (shown in gray vs pink), the model's prediction remains focused on the correct pocket residues.

## Comprehensive Binding Site Discovery with Minimal Probe Sets {#comprehensive-binding-site-discovery-with-minimal-probe-sets .Heading2}

For some proteins, it is challenging to develop drugs targeting the
known active site due to structural constraints, functional
requirements, or drug resistance mechanisms. In such cases, identifying
allosteric sites^32^ or novel binding pockets becomes crucial for drug
discovery. Since YuelPocket predicts ligand-specific binding sites, we
developed a minimal probe set approach (Methods, Figure 5a) to
comprehensively identify all potential binding sites across diverse
protein targets. A minimal probe set represents an optimized collection
of ligands that can collectively cover the majority of binding sites
across the entire protein dataset while minimizing redundancy and
computational cost.

We implemented a greedy algorithm to construct the minimal probe set,
which operates by iteratively selecting ligands that provide the maximum
incremental coverage of true pocket residues across the protein dataset.
The algorithm begins with an empty probe set and systematically
evaluates each candidate ligand based on its ability to identify
previously uncovered binding site residues. At each iteration, the
ligand that contributes the most new coverage is added to the set, and
the process continues until a predefined coverage threshold is achieved.
This approach resulted in a minimal probe set of 15 ligands (Top 3 shown
in Figure 5c; All shown in Table S3) that provides comprehensive
coverage of binding sites across diverse protein families. As the number
of probes increases, a larger fraction of proteins achieve recall values
above a given threshold for pocket residue identification, demonstrating
the effectiveness of the probe set in capturing diverse binding sites
(Figure 5b).

Compounds in the minimal probe set usually achieve higher coverage of
pocket residues than the other compounds. For example, in protein 1D1V,
the known ligand H4B primarily identifies residues around its own
binding site, missing the pocket for ligand PTU entirely (Figure 5d).
However, when using probe KI2 from our minimal set, the model
successfully identifies both the H4B and PTU binding sites,
precisely aligning with its two ligand binding sites (Figure 5e). While the prediction probabilities
for the H4B site decrease slightly when using KI2, this trade-off is
acceptable given the significant gain in overall binding site coverage.

## Correlation between Pocket Probability and Binding Affinity {#correlation-between-pocket-probability-and-binding-affinity .Heading2}

We also explored whether the probability of finding a binding pocket for
a specific ligand on a protein correlates with the binding affinity
between that ligand and protein. This question is particularly
significant because YuelPocket was trained exclusively on structural
information (protein 3D coordinates and ligand 2D structures) without
any exposure to binding affinity data during training. We investigated
this relationship using the PDBBind dataset, which contains 5,314
protein-ligand complex structures along with their corresponding Kd and
Ki values. For each complex, we computed pocket probabilities using
YuelPocket. Since YuelPocket outputs probability scores for individual
residues, we employed two aggregation methods to assess correlation with
binding affinity: selecting the maximum probability across all residues
and calculating the mean probability across all residues.

Our analysis revealed significant correlations between pocket prediction
probabilities and binding affinities, despite the model never having
been trained on affinity data (Figure S3). For maximum probability
analysis, we observed Pearson correlation of 0.391 and Spearman
correlation of 0.423 across all samples (Table S4). Mean probability
analysis yielded even stronger correlations, with a Pearson correlation
of 0.429 and a Spearman correlation of 0.415 across all samples. These
results demonstrate that YuelPocket\'s pocket prediction probabilities
exhibit moderate to strong correlations with experimental binding
affinities, particularly when using mean probability aggregation. This
correlation is remarkable because it emerges purely from structural
learning without any explicit affinity information during training.

The emergence of binding affinity correlations from purely structural
training data has profound implications for our understanding of
protein-ligand interactions. It suggests that the structural features
that determine binding site formation are inherently linked to the
energetic factors that govern binding strength. This finding further
corroborates the fact that the interaction energy must overcome a
significant entropic loss due to the binding of a small molecule. Our
virtual joint node architecture, by learning to represent the binding
pocket as a distinct molecular environment, appears to capture these
underlying physical principles that connect structure to function.

# Discussion

The introduction of the virtual joint node addresses the computational
complexity challenge in protein-ligand interaction modeling. An
intuitive approach to construct a unified graph is to directly connect
all protein residues to all ligand atoms, but it will create a quadratic
scaling problem that becomes computationally intractable for large
proteins or complex ligands. The edge count in such approach is C + m×n,
where *C* represents the summation of the number of protein backbones
(Figure 1a), residue contacts (Figure 1a), and compound bonds (Figure
1b), m represents the number of protein residues, and n represents the
number of ligand atoms. The m×n term creates quadratic complexity that
grows rapidly with molecular size, limiting the applicability of these
methods to small proteins and simple ligands. We introduce the virtual
joint node as an intermediary that dramatically reduces computational
complexity while maintaining information flow between protein and ligand
components. In YuelPocket, the edge count is C + m + n, resulting in
linear complexity that scales efficiently with molecular size. This
reduction enables the model to handle large proteins and complex ligands
while maintaining the ability to capture the essential interactions that
define binding specificity.


An important consideration in evaluating pocket prediction methods is the completeness of ground truth annotations. Our analysis suggests that many proteins contain uncharacterized binding sites that are counted as false positives when predicted by our model. The precision improvement from 0.2 to 0.4 when considering all ligands for a protein supports this hypothesis, indicating that our model may be identifying real but experimentally unvalidated binding sites.

The robustness of YuelPocket on AlphaFold-generated structures represents a significant advancement for practical drug discovery pipelines. While previous methods often suffer performance degradation when applied to predicted structures due to their reliance on precise local geometry, our results show that YuelPocket maintains high accuracy even in the presence of structural noise. This resilience is likely attributable to our global graph architecture, which captures broader geometric and physicochemical binding patterns rather than overfitting to specific local atomic arrangements. By effectively identifying binding sites on predicted models, YuelPocket bridges the gap between massive protein structure databases and structure-based drug design.

Furthermore, the strong performance of our coordinate-level prediction mode (`pos_sc3`) validates the synergy between granular residue scoring and explicit pocket center prediction. Unlike traditional methods that rely solely on geometric cavity detection, our contrastive learning approach generates candidate centers informed by the global protein-ligand compatibility. This allows YuelPocket to not only identify the general location of binding sites with high precision—surpassing established tools like P2Rank—but also to rank them effectively, providing reliable starting points for downstream molecular docking campaigns.

In conclusion, YuelPocket demonstrates effective small molecule binding
site prediction with AUC-ROC values of 0.85-0.89 on benchmark datasets.
The method shows ligand-specific prediction capabilities and reveals
correlations between pocket probabilities and binding affinities despite
being trained only on structural data. The adaptive threshold strategy
provides better precision than fixed thresholds, while clustering
analysis enables comparison with traditional pocket prediction methods.

# Methods {#methods-1}

## Raw Data Collection {#raw-data-collection .Heading2}

We utilize the Plinder dataset (v2024-06) for model training and primary evaluation. Plinder provides a comprehensive and high-quality collection of protein-ligand complexes with curated splits. We trained our model using the Plinder training split and evaluated it on a subset of 1,036 systems from the Plinder test split.

To evaluate the generalization capability and robustness of our model, we included two additional test sets:
1.  **Holo4k**: An independent benchmark dataset. We rigorously screened the 4,543 systems in the original Holo4k dataset against the Plinder splits. To create a challenging test set (`test340`), we selected 340 systems: 40 that overlap with the Plinder test set (for cross-validation consistency) and 300 that are completely absent from the Plinder dataset (train/val/test), representing entirely novel pockets to test strict generalization.
2.  **AlphaFold**: A dataset consisting of AlphaFold-predicted structures for the 1,036 systems in the Plinder test set, allowing us to evaluate performance on predicted protein structures.

Note that we excluded the widely used **COACH420** dataset from this study. Our overlap analysis revealed that the vast majority of COACH420 systems were already present in the Plinder training data. After filtering for redundancy, the remaining number of unique systems was insufficient to support a statistically meaningful evaluation.

We employ a multi-stage approach to transform three-dimensional structural information from these datasets into graph representations suitable for deep learning. For each protein-ligand complex, we extract atomic coordinates, residue types, and chemical connectivity patterns through specialized parsing functions.

PDB files are processed to extract residue-level information, where each
residue is represented by its Cα atom coordinates
$\mathbf{x}_{i} \in \mathbb{R}^{3}$ and one-hot encoded amino acid type
$\mathbf{h}_{i} \in \{ 0,1\}^{N_{residue}}$. The backbone connectivity
is established through spatial proximity analysis, where consecutive
residues $i$ and $j$ are connected if their Cα atoms satisfy the
distance constraint
$||\mathbf{x}_{i} - \mathbf{x}_{j}||_{2} < 4.1\text{ Å}$.

## Pocket Detection and Ground Truth Generation {#pocket-detection-and-ground-truth-generation .Heading2}

We employ a distance-based approach to identify protein residues that
form the binding interface with the ligand. For each protein residue
$i$, we calculate the minimum distance to any ligand atom:

$$d_{\min,i} = \min_{j \in \text{ligand}}||\mathbf{x}_{protein,i} - \mathbf{x}_{ligand,j}||_{2}$$

A residue is classified as part of the binding pocket if
$d_{\min,i} \leq \tau_{pocket}$, where $\tau_{pocket} = 6.0$ Å
represents the interaction threshold. This generates binary pocket
labels $\mathbf{y} \in \{ 0,1\}^{N_{residues}}$ that serve as ground
truth for supervised learning:

$$y_{i} = \left\{ \begin{matrix}
1 & \text{if }d_{\min,i} \leq \tau_{pocket} \\
0 & \text{otherwise}
\end{matrix} \right.\ $$

## Graph Construction {#graph-construction .Heading2}

We represent the protein-ligand system as a unified heterogeneous graph that explicitly models both local atomic interactions and global molecular context. The graph construction process involves three distinct components: the protein subgraph, the ligand subgraph, and a set of virtual "global" nodes that facilitate information exchange.

**Protein Representation.** We parse the protein structure into a dual-node representation for each residue to capture both backbone and side-chain geometry. For each residue, we create:
1.  A Backbone (BB) node located at the Cα atom, representing the peptide backbone.
2.  A Side-Chain (SC) node located at the geometric center of the side-chain atoms, representing the functional part of the residue.
Edges are established between protein nodes based on spatial proximity (distance $< 8.0$ Å) to capture the local structural environment. Additionally, we introduce a **Protein Virtual Node** that connects to all structural nodes (BB and SC) in the protein, serving as a global aggregator of protein information.

**Ligand Representation.** Small molecules are represented as molecular graphs where atoms serve as nodes and chemical bonds serve as edges. Similar to the protein, we introduce a **Ligand Virtual Node** for each ligand molecule that connects to all its constituent atoms. This virtual node aggregates the holistic chemical features of the ligand.

**Unified Interaction Graph.** The model processes a batch consisting of the protein graph and multiple ligand graphs (one true ligand and several negative/decoy ligands). The graph connectivity is defined as follows:
*   **Intra-Protein Edges**: Spatial contacts between protein nodes.
*   **Intra-Ligand Edges**: Chemical bonds between ligand atoms.
*   **Global Aggregation Edges**: Connections between every normal node and its corresponding virtual node (Protein Nodes $\leftrightarrow$ Protein Virtual Node; Ligand Atoms $\leftrightarrow$ Ligand Virtual Node).

## Node and Edge Feature Encoding {#node-and-edge-feature-encoding .Heading2}

**Node Features.** Each node $v_i$ is initialized with a feature vector $h_i \in \mathbb{R}^{d_{in}}$ composed of:
1.  **Chemical/Residue Type**: One-hot encoding of the amino acid type (for protein nodes) or atom type (for ligand nodes). A special 'BB' type is used for backbone nodes.
2.  **Structural Masks**: A 3-dimensional binary mask vector identifying the node type: $[mask_{protein}, mask_{ligand}, mask_{virtual}]$.

**Edge Features.** Edges $e_{ij}$ carry a 5-dimensional feature vector encoding the interaction type:
$$e_{ij} = [d_{ij}, I_{contact}, I_{bond}, I_{p\_global}, I_{l\_global}]$$
where $d_{ij}$ is the Euclidean distance (0 for non-spatial edges), and the remaining terms are binary indicators for protein contacts, chemical bonds, protein-to-virtual connections, and ligand-to-virtual connections, respectively.

## Model Architecture {#model-architecture .Heading2}

The YuelPocket architecture is built upon a deep Equivariant Graph Neural Network (EGNN) framework, configured with 16 layers and a hidden dimension of 128. The network processes the unified graph to update node embeddings, allowing information to flow between local geometric neighborhoods and global virtual hubs.

$$ H^{(l+1)}, X^{(l+1)} = \text{EGNNLayer}(H^{(l)}, X^{(l)}, E) $$

After 16 layers of message passing, the updated node representations are used to perform two distinct but related tasks via specialized prediction heads:

**1. Global Pairing Prediction (Contrastive Learning).**
We explicitly model the compatibility between the protein and the ligand at a global level. This is achieved by interacting the learned representation of the Protein Virtual Node ($h_{P\_Virt}$) with the Ligand Virtual Node ($h_{L\_Virt}$) via an element-wise product, followed by an MLP:
$$ \text{Score}_{pairing} = \text{MLP}_{pair}(h_{P\_Virt} \odot h_{L\_Virt}) $$
During training, we employ a contrastive loss (InfoNCE) to maximize the score of the true protein-ligand pair while minimizing the scores of decoy ligands.

**2. Residue-Level Pocket Prediction.**
To identify the specific binding residues, we compute the interaction between each normal protein node (BB or SC) and the virtual node of the target ligand. For a protein node $i$ with feature $h_i$ and the true ligand virtual node with feature $h_{L\_True}$, the pocket probability is computed as:
$$ P(pocket_i) = \sigma(\text{MLP}_{pocket}(h_i \odot h_{L\_True})) $$
This design allows the model to predict binding sites conditionally based on the specific chemical nature of the input ligand.

## Loss Function {#loss-function .Heading2}

The model is trained using a multi-objective loss function combining the global pairing objective and the local pocket prediction objective:
$$ \mathcal{L}_{total} = \mathcal{L}_{pairing} + \mathcal{L}_{pocket} $$

*   **Pairing Loss**: Cross-entropy loss identifying the true ligand among 50 random decoys.
*   **Pocket Loss**: A combination of Weighted Binary Cross-Entropy (BCE) to handle class imbalance and Dice Loss to optimize the overlap between predicted and ground-truth pocket regions. The Dice loss is dynamically enabled when the training F1 score exceeds 0.2 to refine the segmentation quality.

## Coordinate-Level Pocket Prediction (Pos_SC3) {#coordinate-level-pocket-prediction .Heading2}

To complement the residue-level predictions, YuelPocket includes a specialized mode (`pos_sc3`) designed to directly predict the spatial coordinates of binding pocket centers. This approach reformulates the pocket prediction task as a ranking problem over a set of candidate points generated on the protein surface.

**Probe Generation.** We utilize the Solvent Accessible Surface (SAS) of the protein to generate candidate pocket centers. We uniformly sample points on the SAS and treat each point as a potential "probe" or hypothesis for a binding site center. This discretization allows us to cover the entire potential binding surface of the protein.

**Graph Construction with Probes.** For each candidate probe $p_k$, we construct a local interaction graph $G_k$ that models the compatibility between the local protein environment around the probe and the target ligand.
*   **Nodes**: The graph consists of protein residues within a spatial cutoff radius (e.g., 10 Å) of the probe, all atoms of the target ligand, and a single **Probe Node** representing the candidate center itself.
*   **Edges**:
    *   **Protein-Probe Edges**: Connect all local protein residues to the Probe Node.
    *   **Ligand-Probe Edges**: Connect all ligand atoms to the Probe Node (effectively making the probe a "virtual hub" similar to the residue-level model).
    *   **Internal Edges**: Standard intra-protein contacts and intra-ligand bonds.
*   **Masks**: Explicit masks distinguish between protein residues, ligand atoms, and the probe node.

This construction enables the model to focus on the specific local geometry and physicochemical properties of the protein surface patch defined by the probe.

**Contrastive Learning Strategy.** We train the model using a contrastive learning framework (InfoNCE loss) to distinguish the true binding pocket from various types of decoys. For a given protein-ligand complex:
*   **Positive Sample**: A probe generated near the ground-truth ligand center ($distance < 4.0$ Å).
*   **Negative Samples (Decoys)**:
    1.  **Spatial Decoys**: Probes generated on the same protein surface but far from the binding site ($distance \ge 4.0$ Å).
    2.  **Ligand Decoys**: The true pocket probe paired with a wrong (randomly selected) ligand.

The model scores each probe-ligand pair, and the objective is to maximize the score of the Positive Sample while minimizing the scores of all Decoys. At inference time, we score all SAS probes for a target protein and ligand. The highest-scoring probes represent the predicted binding pocket centers. This approach naturally handles multiple binding sites and provides precise 3D coordinates for downstream applications like docking.

The probe selection algorithm is designed to efficiently identify a
minimal set of ligands (probes) that collectively maximize the coverage
of true binding pocket residues across a diverse set of protein targets.
The process operates in a batch-wise manner, iteratively selecting
ligands that contribute the most new coverage of pocket residues for
each batch of proteins. For each batch, the algorithm first determines
the set of true pocket residues for the proteins in the batch. It then
initializes the current coverage using predictions from the already
selected probe set. If the current coverage already exceeds a predefined
threshold (e.g., 80 %), the batch is skipped. Otherwise, the algorithm
evaluates each candidate ligand by predicting its pocket coverage across
the batch and quantifies how many new, previously uncovered, true pocket
residues it identifies. Ligands that contribute new coverage are added
to the probe set, and the process continues until the coverage threshold
is met or all ligands are exhausted. This greedy, coverage-driven
approach ensures that the selected probe set is both efficient and
effective in representing the diversity of binding pockets in the
dataset.

## Metrics {#metrics .Heading2}

We evaluate YuelPocket's performance using standard classification
metrics, including ROC curves (AUC-ROC), precision-recall curves
(AUC-PR), precision $P = \frac{TP}{TP + FP}$, recall
$R = \frac{TP}{TP + FN}$, specificity $S = \frac{TN}{TN + FP}$, negative
predictive value $NPV = \frac{TN}{TN + FN}$, Matthews correlation
coefficient
$MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$,
and F1-score $F1 = \frac{2 \times P \times R}{P + R}$. For comparison
with traditional pocket prediction methods, we use distance
center-to-center $DCC = ||\mathbf{x}_{pred} - \mathbf{x}_{ligand}||_{2}$
and success rates considering top N and N+2 ranked pocket centers, where
N represents the number of true binding sites per protein. To assess
correlations between pocket probabilities and binding affinities, we
employ Pearson correlation and Spearman correlation coefficients with
p-values for statistical significance testing.

# ACKNOWLEDGMENTS

We acknowledge support from the National Institutes of Health 1R35
GM134864 and the National Science Foundation grant 2210963.

# DATA AND SOFTWARE AVAILABILITY

Source codes and test data are deposited at:
https://github.com/hust220/yuel_pocket.git.

# SUPPORTING INFORMATION AVAILABLE

The supporting information provides Figure S1- S3, and Tables S1-S4.

# DECLARATION OF INTERESTS

The authors declare no competing financial interest.

# AUTHOR CONTRIBUTIONS

Jian Wang contributed to the conceptualization, methodology, model
development, data analysis, writing of the original draft, and
visualization of the study. Nikolay V. Dokholyan provided supervision,
resources, writing review and editing, project administration, and
funding acquisition.

# REFERENCES {#references-1}

1\. Le Guilloux, V., Schmidtke, P. & Tuffery, P. Fpocket: an open source
platform for ligand pocket detection. *BMC bioinformatics* **10**, 1--11
(2009).

2\. Wang, J. & Dokholyan, N. V. A Diffusion-Based Framework for
Designing Molecules in Flexible Protein Pockets. *bioRxiv* 2025--05
(2025).

3\. Wang, J. & Dokholyan, N. V. Multimodal Bonds Reconstruction Towards
Generative Molecular Design. *bioRxiv* 2025--05 (2025).

4\. Wang, J. & Dokholyan, N. V. MedusaDock 2.0: Efficient and Accurate
Protein--Ligand Docking With Constraints. *Journal of Chemical
Information and Modeling* **59**, 2509--2515 (2019).

5\. Ding, F. & Dokholyan, N. V. Incorporating backbone flexibility in
MedusaDock improves ligand-binding pose prediction in the CSAR2011
docking benchmark. *Journal of chemical information and modeling*
**53**, 1871--9 (2013).

6\. Wang, J. & Dokholyan, N. V. Yuel: Improving the Generalizability of
Structure-Free Compound-Protein Interaction Prediction. *Journal of
chemical information and modeling* **62**, 463--471 (2022).

7\. Wang, J. & Dokholyan, N. V. Leveraging Transfer Learning for
Predicting Protein--Small-Molecule Interaction Predictions. *J. Chem.
Inf. Model.* **65**, 3262--3269 (2025).

8\. Chirasani, V. R. *et al.* Whole proteome mapping of compound-protein
interactions. *Current Research in Chemical Biology* **2**, 100035
(2022).

9\. DiMasi, J. A., Grabowski, H. G. & Hansen, R. W. Innovation in the
pharmaceutical industry: New estimates of R&D costs. *Journal of Health
Economics* **47**, 20--33 (2016).

10\. Jumper, J. *et al.* Highly accurate protein structure prediction
with AlphaFold. *Nature* **596**, 583--589 (2021).

11\. Abramson, J. *et al.* Accurate structure prediction of biomolecular
interactions with AlphaFold 3. *Nature* **630**, 493--500 (2024).

12\. Shen, S. *et al.* An update for AlphaFold3 versus experimental
structures: assessing the precision of small molecule binding in GPCRs.
*Acta Pharmacol Sin* 1--10 (2025) doi:10.1038/s41401-025-01617-4.

13\. Hernandez, M., Ghersi, D. & Sanchez, R. SITEHOUND-web: a server for
ligand binding site identification in protein structures. *Nucleic acids
research* **37**, W413--W416 (2009).

14\. Binkowski, T. A., Naghibzadeh, S. & Liang, J. CASTp: computed atlas
of surface topography of proteins. *Nucleic acids research* **31**,
3352--3355 (2003).

15\. Krivák, R. & Hoksza, D. P2Rank: machine learning based tool for
rapid and accurate prediction of ligand binding sites from protein
structure. *J Cheminform* **10**, 39 (2018).

16\. Chen, K., Mizianty, M. J., Gao, J. & Kurgan, L. A Critical
Comparative Assessment of Predictions of Protein-Binding Sites for
Biologically Relevant Organic Compounds. *Structure* **19**, 613--621
(2011).

17\. Jiménez, J., Doerr, S., Martínez-Rosell, G., Rose, A. S. & De
Fabritiis, G. DeepSite: protein-binding site predictor using
3D-convolutional neural networks. *Bioinformatics* **33**, 3036--3042
(2017).

18\. Aggarwal, R., Gupta, A., Chelur, V., Jawahar, C. V. & Priyakumar,
U. D. DeepPocket: Ligand Binding Site Detection and Segmentation using
3D Convolutional Neural Networks. *J. Chem. Inf. Model.* **62**,
5069--5079 (2022).

19\. Burley, S. K. *et al.* Protein Data Bank (PDB): The Single Global
Macromolecular Structure Archive. in *Protein Crystallography* (eds.
Wlodawer, A., Dauter, Z. & Jaskolski, M.) vol. 1607 627--641 (Springer
New York, New York, NY, 2017).

20\. Kandel, J., Tayara, H. & Chong, K. T. PUResNet: prediction of
protein-ligand binding sites using deep residual neural network. *J
Cheminform* **13**, 65 (2021).

21\. Ronneberger, O., Fischer, P. & Brox, T. U-Net: Convolutional
Networks for Biomedical Image Segmentation. in *Medical Image Computing
and Computer-Assisted Intervention -- MICCAI 2015* (eds. Navab, N.,
Hornegger, J., Wells, W. M. & Frangi, A. F.) vol. 9351 234--241
(Springer International Publishing, Cham, 2015).

22\. He, K., Zhang, X., Ren, S. & Sun, J. Deep residual learning for
image recognition. in *Proceedings of the IEEE Computer Society
Conference on Computer Vision and Pattern Recognition* vols 2016-Decem
(2016).

23\. Scarselli, F., Gori, M., Tsoi, A. C., Hagenbuchner, M. &
Monfardini, G. The graph neural network model. *IEEE Transactions on
Neural Networks* **20**, 61--80 (2008).

24\. Meller, A. *et al.* Predicting the locations of cryptic pockets
from single protein structures using the PocketMiner graph neural
network. *Biophysical journal* **122**, 445a (2023).

25\. Evteev, S. A., Ereshchenko, A. V. & Ivanenkov, Y. A. SiteRadar:
Utilizing Graph Machine Learning for Precise Mapping of
Protein--Ligand-Binding Sites. *J. Chem. Inf. Model.* **63**, 1124--1132
(2023).

26\. Xia, Y., Pan, X. & Shen, H.-B. LigBind: Identifying Binding
Residues for Over 1000 Ligands with Relation-Aware Graph Neural
Networks. *Journal of Molecular Biology* **435**, 168091 (2023).

27\. Hu, L., Benson, M. L., Smith, R. D., Lerner, M. G. & Carlson, H. A.
Binding MOAD (Mother Of All Databases). *Proteins* **60**, 333--340
(2005).

28\. Schrodinger, L. L. C. The PyMOL molecular graphics system.
*Version* **1**, 0--0 (2010).

29\. Yin, S., Ding, F. & Dokholyan, N. V. Eris: An automated estimator
of protein stability \[2\]. *Nature Methods* **4**, 466--467 (2007).

30\. Brigham, E. O. *The Fast Fourier Transform and Its Applications*.
(Prentice-Hall, Inc., USA, 1988).

31\. Hahsler, M., Piekenbrock, M. & Doran, D. dbscan: Fast density-based
clustering with R. *Journal of Statistical Software* **91**, 1--30
(2019).

32\. Wang, J. *et al.* Mapping allosteric communications within
individual proteins. *Nature communications* **11**, 1--13 (2020).

33\. Weininger, D. SMILES, a chemical language and information
system. 1. Introduction to methodology and encoding rules. *Journal of
chemical information and computer sciences* **28**, 31--36 (1988).

34\. Kipf, T. N. & Welling, M. Semi-supervised classification with graph
convolutional networks. *arXiv preprint arXiv:1609.02907* (2016).

35\. Landrum, G. RDKit: A software suite for cheminformatics,
computational chemistry, and predictive modeling. (2013).

# FIGURES

![](media/image2.svg){width="6.5in" height="4.429861111111111in"}

Figure 1. Structural and topological representation of protein-compound
interactions in YuelPocket\'s graph neural network framework.​​

\(a\) ​Protein graph: Red nodes depict Cα atoms (spheres) with edges
(black lines) representing backbone connectivity (labeled \"Backbone\")
and inter-residue contacts (\"Contacts\").

(b)​ Compound graph: Blue nodes show heavy atoms (C, N labeled) with bond
edges (\"Bonds\"); stick model overlay illustrates chemical structure.

(c)​ Joint connection topology: Virtual joint node (labeled \"Joint\")
mediates protein-compound interactions via two edge types:
\"Protein-Joint Edges\" and \"Joint-Compound Edges\", avoiding
all-to-all residue-atom connections.

(d)​ GNN architecture: Graph input (left) passes through message-passing
layers (center) with learned edge weight updates, culminating in pocket
probability predictions (right).

\(e\) ​Spatial clustering: Euclidean embedding of protein residues
(colored by cluster ID) reveals pocket localization patterns.

![](media/image4.svg){width="6.5in" height="4.280555555555556in"}

Figure 2. Evaluation metric and performance of YuelPocket on benchmark datasets.

(a) Schematic definition of a successful prediction. A prediction is considered successful if the top-ranked residue falls within a defined distance (e.g., 4 Å) of any ligand atom.

(b) Success rates on the PLINDER test set (1,036 systems) at varying distance thresholds (4–10 Å) for Top-1, Top-3, and Top-10 ranked residues.

(c) Success rates on the Holo4K test set (340 systems) under the same evaluation criteria, demonstrating robust generalization.

(d) Visualization of predicted binding probabilities for a monomeric protein (PDB: 1GZF). Residues are colored by probability (red: high, white: low), showing accurate pocket localization.

(e) Visualization for a dimeric protein complex (PDB: 1B5D), highlighting the model's ability to identify binding sites in multimeric structures.

![](media/image6.svg){width="6.5in" height="5.549305555555556in"}



![](media/image8.svg){width="6.5in" height="3.7819444444444446in"}

Figure 4. Robustness of YuelPocket on AlphaFold-predicted structures.

(a) Residue-level success rates on the AlphaFold dataset show high accuracy (Top-10 > 90%).

(b-c) Success rates on the AlphaFold dataset using DCA (b) and DCC (c) metrics. YuelPocket (blue) consistently achieves higher success rates than P2Rank (red).

(d) Distribution of Cα RMSD between AlphaFold models and experimental crystal structures for the 1,036 test systems.

(e) P2Rank and YuelPocket success rates stabilize after an initial drop as RMSD increases, demonstrating robustness.

(f) Consistency analysis showing high overlap (blue) between successful predictions on Ground Truth (GT) and AlphaFold (AF) structures.

(g-i) Visual comparison of protein 4I4V. The model correctly identifies the binding site on both the Experimental structure (g) and the AlphaFold model (h), despite backbone deviations visible in the superposition (i).

![](media/image10.svg){width="6.5in" height="6.7243055555555555in"}

Figure 5. Minimal probe set construction and evaluation for
comprehensive binding site mapping.​​

(a)​ Schematic of the definition of minimal probe set.

(b)​ Fraction of proteins with the recall of pocket residues greater than
the recall threshold versus the recall threshold. The 15-probe set (red
curve) achieves superior coverage compared to single probes.

(c)​ Chemical structures of top 3 ranked probes: ​KI2​ (from PDB 1NH0), ​PTY​
(from PDB 3AR4), and ​FER​ (from PDB 3CBG).

\(d\) Native ligand ​H4B​ only detects its own binding site, missing the
​PTU​ pocket.

\(e\) Probe KI2 identifies both H4B and PTU sites, demonstrating pan-pocket coverage.

Figure 3. Performance comparison between YuelPocket and P2Rank.

(a-b) Success rates on the PLINDER test set using (a) Distance to Closest Atom (DCA) and (b) Distance Center-to-Center (DCC) metrics. YuelPocket (blue) consistently outperforms P2Rank (red) in both Top-1 (solid) and Top-3 (dashed) ranks.

(c-d) Success rates on the Holo4K test set for (c) DCA and (d) DCC metrics, showing similar superior performance for YuelPocket.

(e-h) Visual comparison on protein 7T8F. (e) Raw SAS probe scores from YuelPocket and (f) resulting top 2 clusters. (g) P2Rank predictions and (h) resulting clusters. YuelPocket provides a clearer and more accurate definition of the binding pocket.
