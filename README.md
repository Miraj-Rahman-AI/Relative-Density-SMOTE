# **Relative Density SMOTE (R-SmoteKClasses)**

## **Overview**  
**Relative Density SMOTE (R-SmoteKClasses)** is an advanced oversampling technique designed for handling **multi-class imbalanced datasets**. Traditional SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic samples by interpolating between existing minority class samples. However, it does not consider the **density distribution** of different classes, which can lead to poor generalization.  

This implementation of **R-SmoteKClasses** aims to improve SMOTE by incorporating **relative density-based sampling** to generate synthetic data while preserving the original class distribution. The algorithm ensures better representation of minority classes while preventing **overfitting**.  

---

## **Features**  
✅ Handles **multi-class imbalanced datasets**  
✅ Uses **relative density-based** oversampling  
✅ Prevents **overfitting** and **mode collapse**  
✅ Compatible with **Scikit-learn datasets**  
✅ Provides **data visualization tools**  

---

## **How It Works**  
1. **Identifies class distribution**: The algorithm calculates the distribution of all class labels in the dataset.  
2. **Determines the majority class**: The class with the highest sample count is treated as the **reference class**.  
3. **Generates synthetic samples**: Using **nearest neighbor interpolation**, synthetic samples are generated based on class densities.  
4. **Balances the dataset**: Synthetic samples are added to the minority class until the dataset is balanced.  
5. **Visualizes the data**: The tool includes a visualization function to compare **original vs. resampled datasets**.  

---

## **Installation**  
Clone the repository and install the dependencies:
```bash
# Clone the repository
git clone https://github.com/yourusername/relative-density-smote.git

# Change into the directory
cd relative-density-smote

# Install dependencies
pip install -r requirements.txt
```

---

## **Usage**  
```python
from rsmote import RSmoteKClasses  # Import the module

# Initialize the RSmoteKClasses instance
rsmotek = RSmoteKClasses(ir=1, k=5, random_state=42)

# Apply the resampling
X_resampled, y_resampled = rsmotek.fit_resample(X, y)

# Visualize the dataset
rsmotek.visualize_data(X_resampled[:, :2], y_resampled, title="Resampled Dataset")
```

---

## **Dependencies**  
- **Python 3.7+**  
- **NumPy**  
- **Matplotlib**  
- **Scikit-learn**  

Install dependencies manually using:
```bash
pip install numpy matplotlib scikit-learn
```

---

## **Applications**  
📌 **Fraud detection** (handling rare fraudulent transactions)  
📌 **Medical diagnosis** (addressing class imbalance in disease prediction)  
📌 **Customer churn modeling** (balancing datasets in classification tasks)  

---

## **Contributing**  
Contributions and improvements are welcome! Please **fork**, **star**, and submit a **pull request**.  




