# Draft 1 — Lit Review: ML for MD & Conformation Sampling (A–D)

## 1. Introduction
Molecular dynamics (MD) provides atomistic trajectories by integrating Newton’s equations under an assumed potential energy surface (PES). In practice, two constraints dominate: **force-field error** (limited accuracy/transferability) and the **rare-event timescale gap** (folding, binding, nucleation, self-assembly). Recent progress in machine learning has produced three complementary responses: (i) learned force fields that approximate ab initio PESs at MD cost, (ii) learned collective variables (CVs) that accelerate exploration of rare events, and (iii) kinetics/workflow frameworks that validate and orchestrate simulation campaigns. This review organizes the literature into four buckets: **(A) ML interatomic potentials (MLIPs/NNPs) as force fields**, **(B) hybrid ML+MD for biomolecules**, **(C) ML-enhanced sampling and learned CVs**, and **(D) kinetic/state models and adaptive workflows**.

## 2. ML force fields (Bucket A): from descriptor kernels to equivariant networks and systematic expansions

### 2.1 Early MLIPs and the “local environment” paradigm
A foundational idea in ML force fields is that total energy can be decomposed into a sum of atomic contributions that depend on local environments. Early work such as **GAP** (Bartók 2009) established that symmetry-preserving representations of neighborhoods combined with nonparametric regression can reproduce near-DFT accuracy while enabling long MD, and it clarified the practical tradeoff between locality (cutoff) and attainable error (“error floors”). This line influenced later descriptor-based models (SOAP/GAP lineage) and provided a conceptual template: **encode local symmetry + learn a smooth mapping to energy/forces**.

### 2.2 End-to-end learned force fields: message passing and physical augmentation
Deep neural models replaced hand-crafted descriptors with learned representations. **SchNet** (Schütt 2018) demonstrated continuous-filter message passing as a general architecture for predicting energies and forces directly from coordinates. A central limitation of purely local message passing is long-range physics. **PhysNet** (Unke 2019) addressed this by embedding explicit **electrostatics via learned charges** (with charge conservation) and adding dispersion, arguing that correct asymptotics and reactive pathways require long-range structure that local models otherwise struggle to represent.

### 2.3 Equivariance as a data-efficiency and stability lever
A major modern development is enforcing geometric equivariance. **NequIP** (Batzner 2022) showed that E(3)-equivariant representations significantly improve accuracy and data efficiency for force learning, and sparked a wave of scalable equivariant MLIPs. **MACE** (Batatia 2023) extends equivariant message passing with higher-order features to obtain strong accuracy–speed tradeoffs. **Learning local equivariant representations** (Musaelian 2023) advances large-scale atomistic dynamics with architectures optimized for scalability, while **TorchMD-Net** (Thölke 2022) and its follow-on **TorchMD-Net 2.0** (Pelaez 2024) emphasize production-ready implementation choices aimed at practical simulation throughput. In parallel, the review by **Deringer 2019** frames these methods as an emerging toolbox for materials modeling, shifting the focus from fitting metrics to reliability under simulation conditions.

### 2.4 Systematic expansions: ACE and “beyond RMSE” evaluation
A complementary perspective treats MLIPs as systematic body-ordered expansions. **ACE** provides a completeness-oriented framework that unifies and generalizes descriptor families. **Drautz 2020** extends ACE to vectorial and tensorial properties and to additional atomic DOFs (magnetism, charge transfer), strengthening the case for hybrid/component models where learned short-range contributions coexist with explicit long-range terms. Importantly, **Kovács 2021** argues that force-field evaluation must go *beyond MAE/RMSE*, proposing stress tests including normal modes, high-temperature stability (“holes”), torsional profiles, and bond breaking. Their results show that regularization and physically meaningful 1-body references can dominate extrapolation behavior, implying that “good test RMSE” can still yield unusable MD.

### 2.5 Data generation and reliability: concurrent learning and symmetry-driven efficiency
Because force fields are only reliable within the distribution they have seen, dataset generation is central. **DeePMD** (Zhang 2017) established a scalable NN PES form used widely in production settings. **DP-GEN** (Zhang 2020) operationalized *concurrent learning*—explore with an ensemble, detect uncertain regions via model deviation, label selectively with DFT/QM, and retrain—turning force-field fitting into an automated HPC workflow. At the other end of the spectrum, **sGDML** (Chmiela 2018) and its software implementation (Chmiela 2019) demonstrate extreme data efficiency for single molecules by learning conservative force fields directly in the gradient domain while discovering relevant permutation symmetries from data, enabling highly accurate MD/PIMD for fixed chemical systems but with limited transferability.

### 2.6 Summary of Bucket A
Across A, the field has moved from descriptor kernels to deep message passing, then to equivariant architectures and systematic expansions. The open reliability challenge is not expressivity alone, but the combination of **coverage (active learning), long-range physics, and robustness under MD**.

## 3. ML-enhanced sampling and learned collective variables (Bucket C): discovering slow modes and reaction coordinates

### 3.1 Tooling foundations: biasing engines and dynamical reconstruction
Enhanced sampling requires robust implementation support. **PLUMED** (Tribello 2013) is foundational infrastructure enabling common biasing methods and facilitating reproducible CV/bias definitions across MD engines. On the methodological side, **Tiwary 2013** provides a key bridge from biasing to dynamics by describing conditions under which biased simulations can be used to recover kinetic information, an issue that becomes critical once ML-derived CVs are used to drive sampling.

### 3.2 Representation learning for slow coordinates: from time-lagged objectives to information bottlenecks
Several approaches learn low-dimensional representations aligned with slow processes:
- **VDE** (Hernández 2018) proposes a time-lagged variational autoencoder to encode nonlinear dynamics into a low-dimensional latent space, with interpretability tools to identify salient features.
- **TAE** (Wehmeyer & Noé 2018) replaces reconstruction with time-lagged prediction and links the linear limit to TICA/TCCA, providing an early, clean bridge between deep learning and transfer-operator-based slow modes.
- **Deep-TICA** (Bonati 2021) proposes a practical strategy to learn transfer-operator slow modes from biased simulations via scaled-time reweighting and then bias the learned coordinate with OPES, explicitly closing the loop from “learn CV” to “use CV for rare-event sampling.”
- **SPIB** (Wang & Tiwary 2021) frames RC learning as an information bottleneck where the representation is optimized to predict future states, creating a direct relationship between RCs and kinetic separability. **GNN-SPIB** (Zou 2024) extends this idea by learning representations from coordinates via graph neural networks, reducing dependence on hand-engineered descriptors.
- **RAVE** (Ribeiro 2018) emphasizes iterative refinement: learn an RC, bias sampling, reweight, and update, creating a general recipe for self-improving exploration.

### 3.3 ML-guided exploration beyond bias potentials
Not all ML acceleration operates as an explicit bias. **Machine-guided path sampling** (Jung 2023) uses ML to identify and sample mechanistic pathways in self-organization, highlighting a broader “ML-guided exploration” paradigm. In biomolecular contexts, **AlphaFold2-RAVE** (Vani 2023) illustrates a combined workflow where ML structure prediction provides plausible starting states while ML-guided enhanced sampling refines the thermodynamic ensemble.

### 3.4 Summary of Bucket C
Bucket C methods are most convincing when they (i) define objectives aligned with slow dynamics, (ii) provide a reweighting/validation story (especially when trained on biased data), and (iii) demonstrate improved kinetics/thermodynamics rather than only better latent visualizations.

## 4. Kinetics models and adaptive workflows (Bucket D): variational theory, software stacks, and campaign orchestration

### 4.1 Theory backbone: VAMP for general Markov processes
Learning representations for kinetics is grounded by variational principles. **Wu & Noé 2019** generalize VAC/TICA beyond reversible equilibrium systems via Koopman SVD, yielding the **VAMP** framework and cross-validation-friendly scores (e.g., VAMP-E). This provides a principled way to select features and model capacity by kinetic fidelity, and it underpins later deep kinetics models.

### 4.2 MSM software stacks: reproducible pipelines and validation
Two major toolkits make MSM-based kinetics workflows practical:
- **PyEMMA 2** (Scherer 2015) provides an end-to-end pipeline (featurization → TICA/kinetic map → clustering → MSM/HMSM → CK validation → analysis via TPT/MFPT/committors) and emphasizes uncertainty and coarse-graining via HMM/HMSM models.
- **MSMBuilder** (Harrigan 2017) offers a scikit-learn-style pipeline and stresses principled hyperparameter selection via cross-validated variational scores (e.g., GMRQ), improving reproducibility and reducing heuristic tuning.

### 4.3 Adaptive sampling at campaign scale
When the limiting factor is exploration efficiency rather than force evaluation, workflow-level control becomes important. **DeepDriveMD** (Lee 2019) implements a simulate → learn → query → respawn loop, using a CVAE latent space over contact maps to identify outliers and spawn new trajectories. This illustrates how ML can sit “above” MD to allocate compute resources across an ensemble, particularly on HPC systems.

### 4.4 Summary of Bucket D
Bucket D provides the validation and orchestration layer: variational scores (VAMP/GMRQ), MSM/HMSM pipelines, CK tests, and campaign-level adaptive loops that quantify and improve effective sampling.

## 5. Synthesis toward hybrid ML+MD for biomolecules (Bucket B): integration requirements and open problems
Although the papers span different goals, they imply a consistent “hybrid stack” for biomolecular simulation:

1) **Accurate forces where needed (from A):** equivariant MLIPs (NequIP/MACE/TorchMD-family) or structured expansions (ACE) supply improved local accuracy; physics augmentation (PhysNet electrostatics/dispersion) addresses long-range behavior; active learning (DP-GEN) provides error control by expanding coverage adaptively.
2) **Accelerated exploration (from C):** learned RCs/slow modes (TAE/VDE/Deep-TICA/SPIB/RAVE) can bias simulations (via PLUMED-compatible engines) or guide which states to resample.
3) **Kinetic/thermodynamic validation (from D):** VAMP-style objectives and MSM toolchains (PyEMMA/MSMBuilder) supply principled diagnostics to prevent “good-looking” embeddings that fail to preserve kinetics.

From a biomolecular hybrid perspective, the critical unresolved issues are (i) **long-range electrostatics and polarization** in local ML models, (ii) robust **out-of-distribution detection** and correction to avoid holes, (iii) consistent **coupling/partitioning interfaces** between ML and classical components, and (iv) reproducible end-to-end workflows that join MLIPs, enhanced sampling, and kinetics validation in standard simulation stacks.

## 6. Conclusion
Taken together, these works indicate that progress toward reliable ML-enhanced biomolecular simulation requires more than improved architectures. The most durable advances come from (i) symmetry- and physics-aware force models with controlled extrapolation, (ii) learned CV objectives tied explicitly to slow dynamics and rare-event sampling, and (iii) rigorous kinetics validation and workflow orchestration. Hybrid ML+MD for biomolecules is therefore best framed as a systems problem: integrating accurate local models, long-range physics, sampling acceleration, and kinetic validation into a single, reproducible simulation stack.
