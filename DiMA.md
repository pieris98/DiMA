## Diffusion on Language Model Encodings for Protein Sequence Generation

Viacheslav Meshchaninov * 1 Pavel Strashnov * 2 Andrey Shevtsov * 2 Fedor Nikolaev 2 Nikita Ivanisenko 2 Olga Kardymon 2 Dmitry Vetrov 1

## Abstract

Protein sequence design has seen significant advances through discrete diffusion and autoregressive approaches, yet the potential of continuous diffusion remains underexplored. Here, we present DiMA, a latent diffusion framework that operates on protein language model representations. Through systematic exploration of architectural choices and diffusion components, we develop a robust methodology that generalizes across multiple protein encoders ranging from 8M to 3B parameters. We demonstrate that our framework achieves consistently high performance across sequence-only (ESM2, ESMc), dual-decodable (CHEAP), and multimodal (SaProt) representations using the same architecture and training approach. We extensively evaluate existing methods alongside DiMA using multiple metrics across two protein modalities, covering quality, diversity, novelty, and distribution matching of generated proteins. DiMA consistently produces novel, high-quality and diverse protein sequences and achieves strong results compared to baselines such as autoregressive, discrete diffusion and flow matching language models. The model demonstrates versatile functionality, supporting conditional generation tasks including protein family-generation, motif scaffolding and infilling, and fold-specific sequence design. This work provides a universal continuous diffusion framework for protein sequence generation, offering both architectural insights and practical applicability across various protein design scenarios. Code is released at GitHub .

* Equal contribution 1 Constructor University, Bremen, Germany 2 AIRI, Moscow, Russia. Correspondence to: Viacheslav Meshchaninov &lt;meshchaninov.viacheslav@gmail.com&gt; , Pavel Strashnov &lt;strashnov@airi.net&gt;, Andrey Shevtsov &lt;shevtsov@airi.net&gt; .

Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).

## 1. Introduction

Protein generation is emerging as a key area in academic research, with applications spanning bioinformatics, synthetic biology, and protein-based therapeutics (Wu et al. , 2021; Ovchinnikov &amp; Huang , 2021). Recent progress in this field has been driven by three main approaches: autoregressive models (Madani et al. , 2023; Ferruz et al. , 2022; Shin et al. , 2021; Lv et al. , 2025), which have demonstrated effectiveness in capturing sequential dependencies, diffusion models (Alamdari et al. , 2023; Wang et al. , 2024; 2025), which have shown promise in both sequence and structure generation tasks and flow- based models (Huguet et al. , 2024; Campbell et al. , 2024; Yim et al. , 2024; Lin et al. , 2024), which have achieved impressive results on protein conditional structure generation.

While discrete diffusion models have been successfully adapted for amino acid sequence generation (Alamdari et al. , 2023; Wang et al. , 2024), and significant advances have been made in three-dimensional protein diffusion (Watson et al. , 2023; Wu et al. , 2024; Lin &amp; AlQuraishi , 2023; Fu et al. , 2024), continuous diffusion on protein sequence representations remains underexplored. Previous attempts (Lee et al. , 2023; Zhang et al. , 2023a) have been limited to specific protein representations or focused primarily on conditional tasks, leaving the potential of continuous diffusion for general protein generation largely untapped.

Recent advances in protein language models (pLMs) have produced increasingly sophisticated continuous representations of proteins (Lin et al. , 2023a; Lu et al. , 2024; Su et al. , 2023). These representations capture both sequence and structural information, providing a natural foundation for latent diffusion approaches. Notably, studies have demonstrated that reconstructing proteins from continuous representations yields greater accuracy than from discrete ones (Lu et al. , 2024; Gaujac et al. , 2024), suggesting potential advantages for continuous diffusion models in capturing protein properties.

In this study, we develop DiMA, a new latent diffusion model that operates on pLM representations. We demonstrate that continuous diffusion on protein embeddings enables effective sequence and structure generation across multiple tasks and encoder architectures. DiMA addresses the

Figure 1. DiMA. The framework consists of three main components: (1) a pre-trained protein language model encoder that maps amino acid sequences to continuous latent representations, (2) a diffusion denoiser that generates latent vectors from Gaussian noise, and (3) sequence and structure decoders that reconstruct amino acid sequences and protein structures from the generated latent representations. During training, the model learns to denoise corrupted protein representations. During inference, the framework supports both unconditional generation and conditional generation tasks including motif scaffolding, fold conditioning, and family-specific generation. The approach enables joint sequence-structure generation while operating entirely in continuous latent space.

<!-- image -->

limitations of previous continuous diffusion approaches that have been limited to specific representations by establishing a unified framework that generalizes across diverse protein encoders. By operating in the continuous latent space of these pre-trained encoders, our approach circumvents the challenges associated with discrete sequence modeling while maintaining the expressiveness needed for complex protein design tasks. The framework is designed to be encoder-agnostic, allowing it to benefit from advances in protein representation learning without requiring architectural modifications. Our key contributions are as follows:

- We develop DiMA, a continuous latent diffusion model that operates effectively across different pLM representations, ranging from sequence-only ESM-2 to dual-decodable CHEAP and structure-aware SaProt encoders.
- Through systematic exploration of architectural choices and diffusion components, we establish a robust methodology that generalizes across encoders ranging from 8M to 3B parameters.
- Despite having only 35M parameters, DiMA demonstrates strong performance on multiple benchmarks, matching or exceeding specialized models in unconditional generation, family-specific design, motif scaffolding and sequence infilling, and fold-conditioned generation.
- Our empirical evaluation across protein sequence and structure metrics demonstrates that DiMA maintains high structural quality while generating diverse and
- novel proteins, as validated on both SwissProt and AFDBv4-90 datasets.
- We show that a single architecture optimized for unconditional generation can effectively adapt to conditional tasks and backbone generation through lightweight modifications, suggesting a path toward unified protein sequence generation.

## 2. Continuous diffusion on LM representations of protein sequences

DiMA is a latent diffusion model that operates on continuous protein representations. DiMA consists of three components: a pre-trained encoder (E) that provides a meaningful latent space representation, a diffusion model (F) that generates latent vectors from Gaussian noise, and a decoder (D) that maps generated latents back to amino acid sequences.

Continuous vs. Discrete Diffusion for Proteins. While discrete diffusion may appear more intuitive for sequence generation, continuous representations offer compelling advantages for protein modeling. Continuous encodings from protein language models capture rich semantic and structural information that has proven effective across diverse protein tasks, from representation learning to structure prediction. Recent studies have shown that continuous representations yield superior reconstruction accuracy compared to discrete alternatives (Lu et al. , 2024), suggesting their potential for generative modeling.

Continuous diffusion offers several theoretical and practical

advantages over discrete approaches such as direct application of established score-based techniques like classifier and classifier-free guidance without requiring discrete approximations. It also provides seamless integration with multimodal representations that jointly encode sequence and structure (e.g. CHEAP, SaProt). Also, continuous diffusion is more stable and efficient in training compared to discrete ones. However, the sequential and discrete nature of protein sequences presents unique challenges for continuous diffusion that necessitate careful adaptation of diffusion components to effectively capture the complexity of protein space.

Latent Space Adaptation. We utilize a pre-trained transformer-based pLM as an encoder with ESM-2 (Lin et al. , 2023a) being the default choice unless otherwise specified. The encoder maps the sequence of discrete amino acids y = [y1, ..., y s ] of length s to the latent vectors x = [x1, ..., x s ] ∈ R s×d , x = E(y). We apply normalization z 0 = Normalize(x) over the hidden dimension d: for each component di, we precompute the mean and variance over the training data and then apply normalization using these statistics to achieve zero mean and unit variance. This transformation allows us to adapt the discrete protein input to a standard Gaussian diffusion framework.

Noise Schedule Optimization. We have found that the linear and cosine noise schedulers widely employed in the image domain (Song et al. , 2021b; Ho et al. , 2020; Nichol &amp; Dhariwal , 2021) are sub-optimal for the protein domain. We conjecture that this happens due to the sequential and discrete nature of the protein representations.

The reconstruction loss of diffusion models trained with such schedulers remains minimal at small noise scales. Consequently, the reconstruction of z0 from zt = √
αtz0 + √
1 − αtε becomes quite trivial for the model for a long period of time, leading to inefficient training. We adopted the noise schedule from (Hoogeboom et al. , 2023):

<!-- formula-not-decoded -->

where d is a hyperparameter reflecting the schedule's rate. The larger the value of d, the greater the data corruption rate. In this work we use d = 10, hence the schedule is named tan-10. We utilize a heuristic approach based on the observation that the reconstruction loss should exhibit an approximately linear increase over diffusion time (Figure 2). The rationale behind the tan-10 schedule is discussed in Apeendix C.5 .

Self-Conditioning. Following recent advances in sequence generation, we apply the self-conditioning technique (Chen et al. , 2022). The denoising network predicts zˆ ˆ 0 using the latent variable zt and timestep t as input. Selfconditioning additionally utilizes predicted zˆ ˆ 0,s from the

Figure 2. Left: the diffusion reconstruction loss of z0 from zt with different noise schedules: ||z0 − z ˆ θ (zt, t)|| 2 . Right: √αt(1).

<!-- image -->

previous timestep s for estimation zˆ ˆ 0,t = ˆzθ(zt, t, zˆ ˆ 0,s ) , t &lt; s .

During training, we sample timestep t ∼ U[0; 1]. In half of the cases, we provide no additional input to the model, setting zˆ ˆ 0,t = ∅, where ∅ is a zero vector. In the remaining cases, we estimate zˆ ˆ 0,t = ˆzθ(zt, t, ∅). The loss is computed as:

<!-- formula-not-decoded -->

where SG[·] denotes the stop-gradient operation. Unlike (Chen et al. , 2022), we apply a linear transformation to z ˆ 0,t and incorporate it into each transformer block's input. The effect of self-conditioning rate on the quality-diversity trade-off is depicted on Figure 12

Token Reconstruction. Our architecture utilizes the ESM2 decoder, which was pre-trained alongside the encoder on masked language modeling objectives. We found that additional fine-tuning of the decoder, specifically for amino acid reconstruction, improves sequence generation accuracy from latent representations x during inference. The decoder maintains a simple architecture with a single linear layer. For CHEAP and SaProt representations, we fine-tune their corresponding pretrained decoders alongside the diffusion model, similar to our approach with ESM-2. We found that low-dimensional embeddings (e.g., ESM-2 8M, d = 320) are less robust to small perturbations than higherdimensional ones (ESM2/SaProt 650M, d = 1280, CHEAP, d = 1024). Fine-tuning the decoders helps minimize these effects during diffusion generation.

Length Determination. Determining sequence length is a key challenge in the inference phase. While many discrete diffusion models (Gong et al. , 2023; Yuan et al. , 2022; Li et al. , 2022; Mahabadi et al. , 2023) generate padding tokens alongside semantic tokens, we found this approach suboptimal for protein generation. Instead, we employ two distinct strategies for training and inference.

During training, we use an attention mask to focus the model exclusively on semantic tokens. This masking strategy is

crucial as encodings of special tokens (like padding) often contain irrelevant or potentially detrimental information to the diffusion model. By excluding these tokens from reconstruction loss computation, we improve both training stability and generation quality.

During inference, we first sample the target sequence length from the training data distribution to ensure realistic protein lengths. We then sample a random Gaussian vector of the appropriate dimension and apply T steps of iterative refinement to generate zˆ ˆ 0 . Finally, we denormalize the latent representation and decode it into an amino acid sequence. Additional details about length distribution modeling are provided in Appendix C.3 .

Model Architecture. Our diffusion model employs a transformer architecture with 12 layers, 16 attention heads, and a hidden size of 320. To adapt this architecture for protein sequence generation, we implement two key modifications (detailed architecture specifications are provided in Appendix C.1).

To incorporate time-dependent information, we integrate time embeddings into the transformer blocks via a linear projection, followed by summation at the input of each block. We extend this mechanism to enable conditional generation tasks, including protein family-specific generation, motif scaffolding, and fold-specific sequence design. For scaffolding and fold-specific sequence design, we further enhance conditioning by introducing cross-attention blocks that process condition encodings, allowing the model to leverage structural constraints effectively.

Inspired by (Bao et al. , 2023), we incorporate long skip connections throughout the transformer architecture. Our experiments demonstrate that these connections significantly accelerate model convergence and enhance training efficiency, facilitating the generation of high-quality protein sequences with improved stability.

## 3. Experiments

In this section, we first identify which architectural components significantly affect generation quality, diversity, and distributional properties (§3.2). We then show that our architecture maintains strong performance across protein encoders of different sizes and dataset scales, without requiring modifications (§3.4). We conclude by demonstrating practical applications in family-specific design (§3.6.2), motif scaffolding (§3.6.1), and fold-conditioned generation (3.6.3), using quantitative metrics to assess success in each task.

## 3.1. Evaluation Metrics

We evaluate protein generation using four key metrics that capture essential aspects of the task: sequence Frechet dis- ´ ´

tance (FD-seq) measures how well generated sequences match the distribution of natural proteins on the held-out test set, predicted Local Distance Difference Test (pLDDT) assesses the structural plausibility of generated proteins, clustering density at 50% sequence identity (CD0 . 5 ) quantifies sequence diversity by measuring the fraction of distinct protein clusters in generated samples, and novelty scores evaluate similarity to training data to detect potential memorization. These metrics complement each other, allowing us to evaluate both the quality and diversity of generated proteins across sequence and structure modalities.

Our comprehensive evaluation framework also includes structural metrics (TM-score, scPerplexity), additional distributional similarity measures (maximum mean discrepancy, 1-Wasserstein optimal transport) computed on both sequence and structure representations, perplexity scores from state-of-the-art pLMs, and diversity assessment through multiple sequence identity thresholds. To ensure robustness, we evaluate all metrics on large sample sizes of 2,048 sequences and validate against independent test sets. While we present key findings in the main text for clarity, each experimental result is accompanied by a complete evaluation using the full metric suite in the corresponding appendix subsection. Detailed descriptions and implementation details of the evaluation metrics are provided in Appendix B .

## 3.2. Denoiser Component Analysis

Existing protein generation methods based on Gaussian diffusion (Lee et al. , 2023; Zhang et al. , 2023a) insufficiently address the selection of optimal methodologies, largely relying on techniques adapted from image diffusion models. In this study, we recognize the need to carefully select the diffusion components, in order to develop a Gaussian diffusion model that can effectively capture the complex patterns of protein space.

In this part of our study, we utilize the ESM-8M encoder and DiMA-35M model for our experiments. To assess the contribution of the proposed design choices to the performance of DiMA, we train several models from scratch with the following modifications: removing the long skip-connections between the shallow and the deep transformer blocks; using time conditioning through admixing the time embeddings to the corrupted latent vectors of amino acids instead of employing a dedicated time layer before each transformer block; omitting the transformer encoder (ESM-2), retaining only its embedding matrix; training the model without selfconditioning; training models with linear and cosine noise schedule; training models with padding reconstruction and without prior length sampling; omitting finetuning the decoder; and using flow matching paradigm in our latent generative model.

Table 1 demonstrates that each proposed feature contributes

Table 1. Ablation study of key components of DiMA trained on SwissProt dataset using ESM-8M encoder. The complete results are presented in Table 5 .

| Model FD-seq (↓) pLDDT (↑) CD0             | . 5  (↑) Novelty (↑)   |
|--------------------------------------------|------------------------|
| Dataset 0.13 80.7 1.000 25.3               |                        |
| Random sequences 3.97 24.8 1.000 85.1      |                        |
| DiMA  0.34  83.3                           | 0.611  35.7            |
| w/o skip connections 0.45 77.3 0.619 43.1  |                        |
| w/o time layers 0.41 79.4 0.550 38.4       |                        |
| w/o ESM encoder 1.07 62.7 0.619 50.0       |                        |
| w/o self-conditioning 0.55 68.2 0.929 56.6 |                        |
| w/o finetuned decoder 0.54 80.1 0.589 38.4 |                        |
| w/o length sampling 0.67 65.0 0.880 58.4   |                        |
| w linear schedule 0.47 77.0 0.611 45.8     |                        |
| w cosine schedule 0.94 54.1 0.878 67.0     |                        |
| w flow-matching 0.71 63.4                  | 0.960 72.2             |

significantly to the model's performance individually. The most substantial decrease in both the quality and distribution similarity of the generated sequences occurs in the ablated models without the ESM-2 encoder, without length sampling, and when trained without self-conditioning. Removing skip-connections and time layers results in a less pronounced impact, but still significant decrease in repetitions of generated sequences and a slight improvement in overall quality.

To ablate the impact of the tan-10 noise schedule, we train our diffusion model with standard linear and cosine schedules, leaving other parameters intact. We find that tan-10 significantly outperforms the cosine schedule in both quality and distribution similarity. It also achieves less expressed but better results than the linear schedule. The detailed results are provided in the Appendix E.1 .

## 3.3. Comparison Across Generative Paradigms

We evaluate DiMA, a latent Gaussian diffusion model trained on ESM-8M encodings, against various generative models for protein sequence generation in a smallscale setup. To ensure a fair comparison, all models are trained from scratch with 35M parameters on SwissProt. For methods requiring predefined sequence lengths, we sample lengths from the training set distribution. This comparison focuses on sequence-based models with publicly available code.

We consider five groups of baselines: autoregressive models (RITA (Hesslow et al. , 2022), SeqDesign (Shin et al. , 2021), nanoGPT (Karpathy , 2023)), score-based models (WalkJump (Frey et al. , 2024)), generative adversarial networks (ProteinGAN (Repecka et al. , 2021)), discrete diffusion models (EvoDiff-OADM (Alamdari et al. , 2023), DPLM (Wang et al. , 2024), D3PM (Austin et al. , 2021)), and flowbased models (DFM (Campbell et al. , 2024)).

Table 2 presents the result of the comparison of existing methods and DiMA. The evaluation demonstrates that the proposed latent Gaussian diffusion approach performs exceptionally well in unconditional protein generation com-

Table 2. Performance comparison between DiMA and alternative generative models for the protein generation of the same parameter count trained on SwissProt dataset. The complete results are presented in Table 6 .

| Model FD-seq (↓) pLDDT (↑) CD0        | . 5  (↑) Novelty (↑)      |
|---------------------------------------|---------------------------|
| Dataset 0.13 80.7 1.000 25.3          |                           |
| Random sequences 3.97 24.8 1.000 85.1 |                           |
| Walk-Jump 2.63 32.4                   | 1.000  82.2               |
|                                       | RITA 1.19 43.9 0.988 60.4 |
| proteinGAN 2.94 30.4 0.955            | 83.5                      |
| SeqDesign 3.53 43.1 0.929 81.2        |                           |
| EvoDiff-OADM 1.49 37.1 0.986 77.6     |                           |
| D3PM 1.50 36.7 0.994 78.4             |                           |
|                                       | DFM 1.46 37.8 0.996 77.2  |
| DPLM 0.50                             | 84.0  0.494 11.5          |
| nanoGPT 1.24 61.0 0.900 53.7          |                           |
| DiMA  0.34                            | 83.3  0.611  35.7         |

pared to alternative generative paradigms. DiMA achieves the best proximity to the distribution of real proteins on the held-out test set, while also showcasing high quality in the generated proteins. NanoGPT, an autoregressive model, shows promising results but fails to match dataset-level metrics, struggling with quality and distribution alignment in protein space. DPLM, a discrete diffusion model, generates structurally plausible proteins but suffers from excessive amino acid repetition and whole-sequence duplication, leading to low diversity and novelty. Compared to DiMA, DPLM exhibits threefold lower novelty and twice the repetition rate, despite close pLDDT scores (see Table 2 and Figure 16). This suggests that while DPLM captures structural features, it lacks the diversity essential for realistic protein generation.

In comparison, other baselines exhibit notably poorer performance. SeqDesign and ProteinGAN, initially designed for narrow classes of proteins, may not be suitable for training on diverse datasets. While EvoDiff outperforms SeqDesign and ProteinGAN, it still demonstrates metric values closer to a random sample than to the dataset, consistent with observations in the original EvoDiff paper (Table S3 of (Alamdari et al. , 2023)).

Detailed results with additional metrics are presented in the Appendix E.2 .

## 3.3.1. BIOLOGICAL RELEVANCE

To explore the biological relevance of the generated sequences, we employ the established protein annotation tool InterProScan (Paysan-Lafosse et al. , 2023; Jones et al. , 2014). We use three different Swissprot-trained models: DPLM, DiMA, and nanoGPT. Our analysis shows that DiMA and DPLM, models exhibiting high-quality metrics, consistently generate sequences with a high degree of annotation compared to the lower-performing nanoGPT (Figure 23A). This pattern is further reflected through the annotation intersections, where DiMA and DPLM demonstrate more significant overlap in their annotations (Figure 23B).

Table 3. Performance of protein sequence generation using DiMA and different encoders on AFDBv4-90 dataset. The complete results are presented in Table 7

| Encoder FD-seq (↓) pLDDT (↑) CD0       | . 5  (↑) Novelty (↑)   |
|----------------------------------------|------------------------|
| Dataset 0.11 83.9 0.994 57.6           |                        |
| Random 2.55 22.16 1.000 84.7           |                        |
| ESM-2 8M 0.560 74.25 0.981 68.0        |                        |
| ESM-2 35M 0.340 75.71 0.986            | 69.1                   |
| ESM-2 150M 0.323 80.07                 | 0.988  65.6            |
| ESM-2 650M 0.318 82.48 0.986 64.1      |                        |
| ESM-2 3B  0.314 83.40                  | 0.969 63.0             |
| ESMc 300M 0.326 82.70 0.963 64.2       |                        |
| CHEAP-shorten-1 0.346 81.92 0.951 64.6 |                        |
| CHEAP-shorten-2 0.340 78.81 0.946 66.2 |                        |
| SaProt 35M 0.366 82.23 0.976 65.5      |                        |
| SaProt 650M 0.411 83.01 0.980 65.7     |                        |

While both approaches achieve similar levels of annotated proteins, their domain length characteristics differ. DiMA accurately reproduces dataset domain lengths and tends to generate small domains (50-75 amino acids). In contrast, DPLM frequently produces more extended domains (approaching 254 amino acids in length) (Figure 23C). We hypothesize that the prevalence of long domains in DPLM correlates with its lower generation diversity, as evidenced by our diversity and distribution similarity metrics (Table 6).

## 3.4. Representation Space Scaling

In the previous sections, we established a robust diffusion model through comprehensive ablation studies and demonstrated its strong performance against baseline generative approaches. Building on this foundation, we now explore the model's adaptability across diverse protein representation spaces using the large-scale AFDBv4-90 dataset.

The AFDBv4-90 dataset represents a carefully curated subset of UniRef50. Comprising 2.2 million protein sequences with &gt; 90% AlphaFold2-predicted structural confidence, this dataset by design excludes intrinsically disordered proteins and low-entropy sequences, ensuring a high-quality protein corpus. As a consequence, this curation allows us to use pLDDT as a reliable, consistent measure of protein structural quality.

We maintain the core denoising model architecture developed in our ablation studies (§3.2), using only simple linear projection techniques to adapt to different encoder dimensions. Here, we examine three representative encoder architectures:

- The ESM-2 sequence-only encoder family, spanning models from 8M to 3B parameters, allows us to investigate performance scaling with model complexity (§3.4.1).
- CHEAP representations, which uniquely enable decoding into both sequence and 3D-structure from its latent

spaces (§3.4.2).

- The SaProt encoder, which integrates structural tokens from FoldSeek's 3Di vocabulary, introducing a hybrid approach to sequence representation with structural awareness (§3.4.3).

By preserving the core diffusion methodology, we aim to understand how different representation spaces influence protein sequence generation, providing insights into the generative capabilities across diverse protein embedding approaches.

## 3.4.1. PERFORMANCE SCALING WITH ESM-2 ENCODERS

We analyze the latent spaces of the ESM encoder family (ESM-2 8M through ESM-2 3B), investigating how performance scales with model capacity (Table 3). DiMA successfully generates designable proteins across the entire encoder spectrum, with generation quality (pLDDT) consistently improving from ESM-2 8M (74.3) to ESM-2 3B (83.4).

The scaling analysis reveals a key trade-off between quality and diversity. While larger encoders demonstrate enhanced precision in capturing protein patterns and improved distributional matching (FD-seq decreases from 0.560 to 0.314), they show reduced sequence diversity (CD0 . 5 declines from 0.981 to 0.969). This pattern manifests in the repetition metric – smaller encoders generate more repetitive patterns, while larger ones produce more refined but potentially less novel sequences (Table 7).

The recently introduced ESM-C 300M (ESM Team , 2024) matches ESM-2 650M in generation quality (82.7 vs 82.5 pLDDT) but shows slightly reduced coverage of protein space (FD-seq 0.326 vs 0.318, CD0 . 5 0.963 vs 0.986). The marginal quality improvement from ESM-2 650M to ESM2 3B suggests that mid-range architectures may offer an optimal balance between performance and computational efficiency.

This systematic evaluation demonstrates that DiMA can effectively leverage encoder representations of varying complexity while maintaining a lightweight deployment footprint, as the encoder becomes dispensable during inference.

## 3.4.2. ADAPTATION TO CHEAP REPRESENTATIONS

CHEAP is an encoder that enables efficient dual-modal representations of proteins, providing access to both sequence and structure information from sequence input alone. Based on ESMFold, it aggregates information from all layers of ESM-2 3B encoder and compresses the continuous space, achieving significant dimensionality reduction while preserving high-fidelity structural and sequence reconstruction. We explore two compression variants: CHEAP shorten 1 dim 1024 and

CHEAP shorten 2 dim 1024, both reducing the channel dimension while the latter additionally compresses the sequence length dimension.

Sequence Generation. DiMA trained with CHEAP encoder demonstrates strong performance in sequence space (Table 3). Both compression variants maintain high generation quality: CHEAP shorten 1 dim 1024 achieves pLDDT of 81.9 while CHEAP shorten 2 dim 1024 reaches 78.8, approaching the dataset quality benchmark of 83.9. The variant without sequence length reduction shows superior performance, suggesting that preserving the full sequence dimension benefits generation quality.

Structure Generation. We assess structural consistency through two evaluation protocols: co-design, comparing generated backbones against structures reconstructed from predicted sequences, and structure-only evaluation using standart protocol desribed in Yim et al. (2023a) (Table 11). The structure-only approach achieves 92.3% success rate with mean scRMSD of 1.091, while co-design demonstrates strong performance with 88.8% success rate and mean scRMSD of 1.043A. These results indicate that DiMA effec- ˚ A
˚ tively leverages CHEAP's compressed yet information-rich latent space for both sequence and structure generation. Notably, the sequence length reduction in CHEAP shorten 2 enables using half-length transformer context for the denoising model, offering substantial computational advantages while maintaining generation quality.

Additional results for the DiMA model using CHEAP encoders for structure generation are available in Appendix E.6 .

## 3.4.3. ADAPTATION TO SAPROT ENCODER

Multimodal SaProt encoder (Su et al. , 2023) presents an alternative approach to protein representation, integrating both sequence and local structural information. Unlike sequenceonly models, SaProt enriches its representations using structural tokens from FoldSeek's (Van Kempen et al. , 2023) 3Di vocabulary, providing compact yet informative structural descriptors.

We evaluate SaProt encoders of two sizes (35M and 650M parameters) using the same hyperparameters for the denoiser model that we established through our ablation studies. The quality metrics demonstrate that structural awareness in SaProt's representations translates to improved generation capabilities - models achieve higher pLDDT scores compared to same-sized ESM-2 variants (82.23 vs 75.71 for 35M and 83.01 vs 82.48 for 650M architectures, Table 3). This improvement is particularly notable with the smaller 35M parameter encoder, suggesting that structural tokens provide an efficient way to encode protein properties.

The integration with SaProt demonstrates DiMA's ability to leverage different types of protein embeddings without

Figure 3. Comparison of DiMA with large pretrained protein generative models in quality and diversity. Circle size represents model scale. The complete results are presented in Table 8 .

<!-- image -->

architectural modifications. This adaptability, combined with strong performance in structure-aware tasks like motifscaffolding (detailed in §3.6.1), indicates that our diffusion framework can effectively capitalize on both sequence and structural information encoded in the latent space.

## 3.5. Comparison with Large Pretrained Models

In this section we compare DiMA-35M trained over ESM3B encodings with existing large protein models, we demonstrate that DiMA achieves performance comparable to existing pre-trained large protein models.

We evaluate DiMA-35M, trained on ESM-3B encodings, against state-of-the-art pre-trained protein models, namely, RITA (Hesslow et al. , 2022), ProtGPT2 (Ferruz et al. , 2022), ProGen2 (Madani et al. , 2023), EvoDiff (Alamdari et al. , 2023), ProLLAMA (Lv et al. , 2025), DPLM (Wang et al. , 2024), Chroma (Ingraham et al. , 2023), Multiflow (Campbell et al. , 2024), RFDiffusion (Watson et al. , 2023), and PLAID-100M (Lu et al. , 2025). For all models, we use the authors' recommended sampling parameters to ensure a fair comparison. However, for autoregressive models ProGen2 and ProLLAMA, which show suboptimal quality and collapse to highly repetitive sequences on default settings, we performed grid searches to identify optimal temperature and top-p values. We consider only models with publicly available pre-trained weights, ensuring transparency and reproducibility. Figure 3 illustrates the relationship between quality, diversity, and model size.

DiMA demonstrates high-quality and diverse protein generation, achieving performance on par with much larger models despite using two orders of magnitude fewer parameters. While DPLM and other diffusion-based models excel in structural plausibility, they lag in diversity, where DiMA outperforms them significantly. Notably, Multiflow, trained

with structural data, achieves similar performance to DiMA despite operating at a comparable parameter scale. An important advantage of DiMA is that it achieves performance comparable to models trained using structural information, while being trained exclusively on amino acid sequences. This demonstrates the model's efficiency and ability to extract meaningful representations from sequence data alone, making it a highly versatile and resource-efficient solution for protein generation tasks.

These findings underscore DiMA's efficiency and scalability, establishing it as a compelling approach for protein sequence generation, even in comparison to large-scale pretrained models. Full results across model sizes and evaluation metrics are provided in Appendix E.4 and Table 8 .

## 3.6. Advanced generation tasks

## 3.6.1. FUNCTIONAL -MOTIF SCAFFOLDING

We evaluate the conditional generation capabilities of DiMA on a challenging task of functional-motif scaffolding using the established RFDiffusion benchmark (Watson et al. , 2023). This task requires designing entirely new protein structures that incorporate and preserve specific functional motifs.

We evaluate DiMA on 24 benchmark problems, where each problem requires generating protein sequences that maintain precise spatial positioning of functionally important residues. For conditioning, we augment DiMA with an encoder that provides motif information to each transformer layer. Following established protocols (Alamdari et al. , 2023), we sample 100 designs per problem and consider a problem solved if at least one design achieves both structural quality (pLDDT ≥ 70) and motif preservation (RMSD ≤ 1A for motif residues). ˚ A 
˚

Using the SaProt-650M encoder, DiMA successfully solves 19 out of 24 problems, outperforming other sequence-based methods, including EvoDiff, DPLM, and DPLM2 (Figure 4). While structure-based methods like RFDiffusion achieve higher overall success rates, DiMA generates more diverse successful scaffolds as measured by unique success rate (0.1 vs 0.06 for RFDiffusion) matching the diversity of ESM-3 while using two orders of magnitude fewer parameters. Interestingly, DiMA and ESM-3 show complementary strengths across different scaffold types, with DiMA achieving the highest unique success rates on 6E6R-type problems. Complete results and experimental details are provided in Appendix E.8 .

## 3.6.2. FAMILY -SPECIFIC GENERATION

Generating proteins that belong to specific functional families is a key task in protein engineering, enabling targeted exploration of sequence spaces with desired characteristics.

Figure 4. Motif-scaffolding: performance comparison across different model sizes. Methods are colored by input modality: sequence-based (purple), sequence-structure co-generation (green), and structure-based (orange). DiMA solves 19/24 benchmark problems while being significantly more compact than other highperforming models. The complete results are presented in Table 13

<!-- image -->

We investigate two approaches for training DiMA (35M parameters) to perform family-specific generation: classifier guidance and conditional fine-tuning.

Using ESM2-650M encodings and the AFDBv4 90 dataset, we train and evaluate our model on eight diverse protein families, including CRISPR-associated proteins, calmodulins, and glycosyl hydrolases. For classifier guidance, we train a lightweight classifier (3 transformer blocks) on noisy protein encodings to predict family membership. For conditional fine-tuning, we augment DiMA with family class label embeddings and fine-tune on all families simultaneously. We compare against significantly larger baselines, including ProLLAMA (7B parameters), ProGen2 (151M parameters), and EvoDiff (640M parameters).

We evaluate generations using multiple complementary metrics: InterProScan for family membership verification (Fidelity), pLDDT for structural quality assessment, and cluster diversity at 50% sequence identity (CD0.5) to measure generated sequence diversity. Both DiMA variants achieve high fidelity to target families while maintaining structural quality. While autoregressive models demonstrate high sequence diversity but struggle with fidelity and quality (pLDDT ≈ 60), discrete diffusion achieves comparable fidelity and quality but generates fewer novel proteins. Notably, the classifierguided variant achieves competitive performance without requiring model fine-tuning, offering a practical advantage for targeted protein design. Complete results and experimental details are provided in Appendix E.5 .

## 3.6.3. FOLD -CONDITIONED GENERATION

Generation of proteins with specific structural properties represents a distinct challenge from sequence-based

Figure 5. Examples of successful proteins generated via foldconditioning, aligned on the corresponding target proteins. Sequence identity percentage between the target protein and the generated one is reported for each design.

<!-- image -->

tasks, requiring the model to capture complex threedimensional relationships. We explore DiMA's capabilities in fold-conditioned generation using the CHEAP encoder (shorten=1, dim=1024), which provides access to ESMFold's latent space representation of protein structure.

We finetune DiMA on the CATH S40 non-redundant dataset ( 27k proteins) and evaluate performance on a hold-out set of 100 structures. For each structure, we generate 10 proteins and assess their similarity to the target fold using the TM-score. Following the protocol of (Watson et al. , 2023), we consider generation successful if at least one design achieves TM-score &gt; 0 . 5, indicating significant structural similarity. For comparison, we evaluate against RFDiffusion, a specialized structure generation model capable of fold-conditioning.

DiMA achieves a mean TM-score of 0.93 with a 100% success rate across the benchmark set, compared to RFDiffusion's mean TM-score of 0.48 and 41% success rate. The average RMSD between the best attempts and their target folds is 2.6A, indicating that DiMA generates structurally simi- ˚ A
˚

lar but non-identical proteins. This performance difference likely stems from DiMA's use of richer structural encodings compared to RFDiffusion's secondary structure and blockadjacency representations. Figure 5 illustrates examples of generated proteins alongside their target folds. Complete experimental details are provided in Appendix E.9 .

## 4. Conclusion

In this paper, we introduce DiMA, a continuous latent diffusion framework for protein sequence generation that operates effectively across diverse protein language model representations. Through systematic exploration of architectural choices and diffusion components, we establish a robust methodology that generalizes from sequence-only encoders to multimodal representations spanning 8M to 3B parameters.

DiMA demonstrates versatile functionality across multiple protein design scenarios, including unconditional generation, family-specific design, motif scaffolding, sequence infilling, and fold-conditioned generation. The framework achieves competitive performance with significantly larger models while maintaining computational efficiency, as the encoder becomes dispensable during inference.

This work provides both architectural insights and practical applicability for protein sequence generation, offering a unified approach that bridges sequence and structure generation through continuous diffusion. The systematic methodology we establish creates a foundation for future developments in computational protein design, demonstrating how domainspecific adaptations can unlock the potential of diffusion models for biological sequence generation.

## Impact statement

This work advances machine learning approaches for protein sequence generation, introducing a framework that bridges discrete and continuous methodologies. Beyond the core technical contributions to diffusion models and protein representation learning, our work has potential applications in therapeutic protein design and synthetic biology. While these applications could benefit society through new drug development and biotechnology advances, we acknowledge they require careful consideration of biosafety and ethical implications in deployment. We believe the benefits of advancing protein design capabilities outweigh potential risks when proper safeguards and responsible development practices are followed. Our framework emphasizes transparency through extensive evaluation metrics and validation protocols, promoting reproducible and responsible progress in computational biology.

