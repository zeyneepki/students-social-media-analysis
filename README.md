# Students' Daily Social Media Usage Analysis

## 📌 Project Objective

This project analyzes students' average daily social media usage time using statistical methods.

The main objectives are to:
- Examine the distribution of daily usage hours,
- Calculate descriptive statistical measures,
- Perform statistical inference on the population mean,
- Support findings with visualizations.

All analyses were conducted using Python.

---

## 🛠 Technologies Used

- Python
- Pandas (data handling and processing)
- SciPy (statistical computations)
- Matplotlib & Seaborn (data visualization)

---

## 📊 Statistical Analyses Performed

The following analyses were applied:

### 1️⃣ Descriptive Statistics
- Mean
- Median
- Variance
- Standard deviation
- Standard error

These measures were calculated to understand the central tendency and variability of daily social media usage.

---

### 2️⃣ Confidence Interval Estimation

A 95% confidence interval for the mean daily usage time was computed.

This estimation provides a range in which the true population mean is likely to fall.

---

### 3️⃣ Hypothesis Testing (Z-Test)

The following hypothesis was tested:

H₀: The average daily social media usage is 5 hours.  
H₁: The average daily social media usage is different from 5 hours.

This test evaluates whether the sample data is statistically consistent with the assumption of a 5-hour mean usage.

---

### 4️⃣ Sample Size Estimation

The required sample size was estimated for a given margin of error and confidence level.

This step demonstrates statistical planning and inference capability.

---

## 📈 Visual Analysis

### Histogram

The histogram illustrates the distribution of daily social media usage hours.

Observations:
- The data appears approximately normally distributed.
- Most values cluster between 4 and 6 hours.
- Extremely low (below 2 hours) and high (above 8 hours) values are rare.

This suggests that the majority of students spend a similar amount of time on social media daily.

---

### Boxplot

The boxplot provides a summary of the distribution.

Observations:
- The median is approximately 5 hours.
- 50% of the data lies roughly between 4 and 6 hours.
- A small number of outliers are present.
- The distribution is relatively balanced without extreme skewness.

This indicates a stable and moderately concentrated usage pattern.

---

## 🎯 Overall Conclusion

In this project:

- Students' daily social media usage was statistically analyzed.
- The distribution of usage hours was examined.
- Statistical inference was performed on the population mean.
- A hypothesis test was conducted to evaluate a 5-hour usage assumption.
- Results were supported with visualizations.

This project demonstrates the application of fundamental statistical analysis and data visualization techniques in Python.

---

## ▶️ How to Run the Project

Install required libraries:

pip install pandas scipy matplotlib seaborn

Then run:

python stats_analysis.py
