# APIE
[📄arXiv]([[2508.10036\] Reflect then Learn: Active Prompting for Information Extraction Guided by Introspective Confusion](https://arxiv.org/abs/2508.10036)) 

> This repository provides the official implementation of the following paper:
>
> [Reflect then Learn: Active Prompting for Information Extraction Guided by Introspective Confusion](https://arxiv.org/abs/2508.10036)
>
> Dong Zhao<sup>1*</sup>, Yadong Wang<sup>1*</sup>, Xiang Chen<sup>1†</sup>, Chenxi Wang<sup>2</sup>, Hongliang Dai<sup>1</sup>,
> Chuanxing Geng<sup>1</sup>, Shengzhong Zhang<sup>1</sup>, Shao-Yuan Li<sup>1</sup>, Sheng-Jun Huang<sup>1†</sup>
>
> <sup>1</sup>MIIT Key Laboratory of Pattern Analysis and Machine Intelligence, College of Computer Science and Technology, Nanjing University of Aeronautics and Astronautics
>
> <sup>2</sup>Zhejiang University

## Overview

![](document\apie-2main-v6.drawio.drawio.png)

Large Language Models (LLMs) show remarkable potential for few-shot information extraction (IE), yet their performance is highly sensitive to the choice of in-context examples. Conventional selection strategies often fail to provide informative guidance, as they overlook a key source of model fallibility: confusion stemming not just from semantic content, but also from the generation of well-structured formats required by IE tasks. To address this, we introduce Active Prompting for Information Extraction (APIE), a novel active prompting framework guided by a principle we term introspective confusion. Our method empowers an LLM to assess its own confusion through a dual-component uncertainty metric that uniquely quantifies both Format Uncertainty (difficulty in generating correct syntax) and Content Uncertainty (inconsistency in extracted semantics). By ranking unlabeled data with this comprehensive score, our framework actively selects the most challenging and informative samples to serve as few-shot exemplars. Extensive experiments on four benchmarks show that our approach consistently outperforms strong baselines, yielding significant improvements in both extraction accuracy and robustness. Our work highlights the critical importance of a fine-grained, dual-level view of model uncertainty when it comes to building effective and reliable structured generation systems.
