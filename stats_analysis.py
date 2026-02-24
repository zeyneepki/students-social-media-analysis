import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

# CSV dosyası okunur ve analizde kullanılacak sütun seçilir
df = pd.read_csv("Students Social Media Addiction.csv")
usage = df["Avg_Daily_Usage_Hours"]

# Örneklem büyüklüğü belirlenir
n = len(usage)

# Ortalama, medyan, varyans, standart sapma ve standart hata hesaplanır
mean = usage.mean()
median = usage.median()
var = usage.var(ddof=1)
std = usage.std(ddof=1)
se = stats.sem(usage)

# Temel istatistiksel değerler ekrana yazdırılır
print(f"Sample size: {n}")
print(f"Mean: {mean:.2f} hours")
print(f"Median: {median:.2f} hours")
print(f"Variance: {var:.2f}")
print(f"Standard deviation: {std:.2f}")
print(f"Standard error: {se:.4f}")

# Ortalama için %95 güven aralığı hesaplanır
z = 1.96
lower_mean = mean - z * se
upper_mean = mean + z * se

# Varyans için %95 güven aralığı, ki-kare dağılımı kullanılarak hesaplanır
chi2_low = stats.chi2.ppf(0.025, df=n-1)
chi2_high = stats.chi2.ppf(0.975, df=n-1)
lower_var = (n-1) * var / chi2_high
upper_var = (n-1) * var / chi2_low

# Güven aralıkları ekrana yazdırılır
print(f"95% confidence interval for mean: [{lower_mean:.2f}, {upper_mean:.2f}]")
print(f"95% confidence interval for variance: [{lower_var:.2f}, {upper_var:.2f}]")

# Hipotez testi yapılır, H0: ortalama 5 saat (z-testi)
mu0 = 5
z_stat = (mean - mu0) / se
p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))

print("\nHypothesis Test (H0: µ = 5) using Z-test")
print(f"z = {z_stat:.2f}, p = {p_val:.4f}")
print("Result:", "H0 rejected (mean is different from 5)" if p_val < 0.05 else "H0 not rejected (mean is not different from 5)")

# Örneklem büyüklüğü tahmini (±0.1 hata payı, %90 güven)
z90 = stats.norm.ppf(0.95)
E = 0.1
required_n = ((z90 * std) / E) ** 2

print(f"\nRequired sample size (±0.1 margin, 90% confidence): {int(required_n) + 1}")

# Grafikler oluşturulup kaydedilir
plt.figure(figsize=(8, 5))
sns.histplot(usage, bins=10)
plt.title("Histogram of Avg Daily Usage Hours")
plt.xlabel("Hours per Day")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("histogram.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.boxplot(y=usage)
plt.title("Boxplot of Avg Daily Usage Hours")
plt.ylabel("Hours per Day")
plt.tight_layout()
plt.savefig("boxplot.png")
plt.close()

print("Plots saved: histogram.png and boxplot.png")
