# MITHEM

A Python implementation of **MITHEM (Multi-label Imbalance-aware Text Classification using Hybrid Ensemble Model)** for **biomedical text classification**. This project applies **TF-IDF, SMOTE-based rebalancing, and ensemble learning** to handle multi-label imbalanced datasets like **MIMIC-III, BioASQ, PubMed RCT, and Medline**. **In this Repo the code is not in it's entirety** due to the sensitive nature of the work & publication of the academic paper in due. The authors of the work decided to make MITHEM source code partially available to public.

## 🚀 Key Features
- **Multi-label classification** for biomedical text datasets  
- **Handles imbalanced datasets** with label binning and **SMOTE**  
- **Uses multiple classifiers (SVM, DT, RF) and meta-learning**  
- **Supports real-world datasets** (MIMIC-III, BioASQ, Medline, etc.)  

---

## 📥 Installation
### Clone the Repository
```bash
git clone https://github.com/mastersubhajit/MITHEM.git
cd MITHEM
```
## Dataset Information & Copyright Notice
For copyright reasons, the MIMIC-III, BioASQ, PubMed RCT, and Medline datasets are NOT included in this repository. These datasets contain sensitive biomedical data and require special access permissions.

### However, this repository includes:
- ✅ Preprocessed outputs and evaluation results in .yaml format
- ✅ Code to train models on any biomedical dataset

### How to Access Biomedical Datasets?
- MIMIC-III: Request access from PhysioNet
- BioASQ: Download from BioASQ Challenge
- PubMed RCT: Available at PubMed RCT dataset
- Medline: Access via NCBI
