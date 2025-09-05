# Data Science Project: Clinical Phenotype Extraction and HPO Mapping with Expert Annotations and LLM Generated Synthetic Data

## Overview

This repository contains a scalable framework for HPO extraction and standardization from unstructured clinical text. The framework addresses the issue of data scarcity in clinical research by combining expert-annotated data with synthetically generated data to improve diagnostic accuracy, particularly in the diagnosis of hereditary diseases.

The framework integrates several components including:
- Entity recognition
- Relation extraction
- HPO standardization

It employs a hybrid data approach and ensures data privacy protection, which is crucial for handling sensitive medical information.

## Features

- **Scalable Framework**: Fully automated pipeline for phenotype extraction and standardization.
- **Data Privacy**: Ensures the protection of sensitive data.
- **Hybrid Data Approach**: Combines high-quality annotated data and 20,056 synthetically generated samples.
- **High Accuracy**: Outperforms existing models with an F1 score of 0.927 for single-label HPO term recognition.
- **Custom Standardization**: Utilizes dictionary matching, cross-encoder refinement, and API-based retrieval for phenotype standardization.

## Dataset

This project uses a **dual-source dataset**, which consists of:
- **Expert-annotated data**: A high-quality annotated dataset by medical professionals.
- **Synthetic data**: 20,056 samples generated using a large language model (LLM) to overcome the scarcity of annotated data.

## Installation

To get started with the framework, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ManshuWu/Data_Science_Project.git
   cd Data_Science_Project

