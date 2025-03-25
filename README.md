# Support Vector Machine (SVM) Classification: A Deep Dive into Margin Optimization

## Overview
This project presents a comprehensive exploration of Support Vector Machines (SVMs) for binary classification. I implemented an SVM classifier from scratch and investigated the principles of margin maximization and the kernel trick. The goal was to understand and visualize how SVMs delineate decision boundaries and identify support vectors, all through an independent, self-driven exploration.

## Mathematical Framework
Given training data \(\{(\mathbf{x}_i, y_i)\}_{i=1}^{N}\) with labels \(y_i \in \{-1, +1\}\), the SVM optimization problem is:
\[
\min_{\mathbf{w}, b} \quad \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{subject to} \quad y_i(\mathbf{w}^T\mathbf{x}_i + b) \ge 1 \quad \forall i.
\]
The optimal hyperplane is defined by:
\[
\mathbf{w}^T\mathbf{x} + b = 0,
\]
and the support vectors are those data points that satisfy \(y_i(\mathbf{w}^T\mathbf{x}_i + b) = 1\). I also delved into the dual formulation and explored how kernel functions can transform the input space to enable nonlinear classification.

## Implementation & Experimentation
- **Algorithm Development:** I built the SVM classifier independently and also compared it with scikit-learn’s SVC for validation.
- **Visualization:** Decision boundaries, margins, and support vectors were visualized for both linear and radial basis function (RBF) kernels.
- **Parameter Analysis:** I extensively tuned the penalty parameter \(C\) and kernel parameters, studying their effect on model generalization and robustness.
- **Insights:** The project revealed the powerful interplay between geometry and optimization in SVMs, emphasizing the significance of support vectors in determining the classification boundary.

## Usage
- **Prerequisites:** Python, scikit-learn, NumPy, Matplotlib.
- **Run the Notebook:** Open `Classification_Support_Vector_Machine.ipynb` and execute the cells.
- **Customization:** Experiment with various kernel functions and adjust parameters to study different classification scenarios.

---
