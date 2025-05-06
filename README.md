# Vessel Traffic Analysis for Maritime Fairway Designation

This repository contains code, notebooks, and documentation developed for the capstone project titled **"Vessel Traffic Analysis for Maritime Fairway Designation"**, conducted at the U.S. Coast Guard Academy in partnership with the United States Coast Guard Navigation Center (NAVCEN).

The project provides a repeatable, data-driven framework for identifying and evaluating vessel traffic within proposed maritime fairways using AIS data, with both rule-based filtering and unsupervised machine learning approaches.

---

## Repository Contents

| File | Description |
|------|-------------|
| `Random_Sampling.py` | Randomly samples full AIS tracks from a large dataset to support manageable analysis on low-resource machines. |
| `Business_Cases.py` | Applies a rules-based approach to identify fairway users using average speed, sinuosity, and percent alignment. |
| `K_Means_Clustering.ipynb` | Clusters vessel tracks using k-means clustering on six-dimensional feature vectors (start/end location, average speed/course, and sinuosity). |
| `Gaussian_Mixture.ipynb` | Applies Gaussian Mixture Model (GMM) clustering to the same track features for improved separation of overlapping patterns. |
| `DBSCAN_DTW.ipynb` | Implements DBSCAN clustering with Dynamic Time Warping (DTW) distance for identifying dense trajectory patterns and noise. |

---

## Project Overview

Fairways are designated corridors free of fixed structures to support safe vessel navigation. The goal of this project is to provide NAVCEN analysts with robust, repeatable tools for:

- Identifying vessels likely to use a proposed fairway.
- Revising fairway boundaries based on actual traffic patterns.
- Comparing traditional rule-based filtering with data-driven clustering.

---

## Methodologies

### Clustering Approaches:
- **KMeans**: Hard-assignment clustering on standardized feature vectors.
- **Gaussian Mixture**: Soft-assignment model allowing overlapping clusters.
- **DBSCAN with DTW**: Density-based clustering using time-series shape similarity.


### Business Rules-Based Filtering:
Applies cutoff thresholds to metrics such as:
- Average Speed Over Ground (SOG)
- Sinuosity
- Percent Alignment with the fairway heading

---

## 🛠️ Requirements

This project uses the following Python libraries:
- `polars`
- `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `ipywidgets`
- `fastdtw` (for DBSCAN with DTW)

---

## Getting Started

1. Download AIS data from [Marine Cadastre](https://marinecadastre.gov/ais).
2. Use `Random_Sampling.py` to generate a manageable dataset.
3. Clip to a proposed fairway polygon in ArcGIS Pro.
4. Choose either:
   - `Business_Cases.py` to apply filtering metrics, or
   - One of the clustering notebooks for unsupervised analysis.
5. Visualize results using built-in plots or export to ArcGIS Pro.
6. Modify fairway and re-run step 4 with revised fairway.
7. Compare results and iterate until satisfied.
