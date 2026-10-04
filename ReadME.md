##  Project Summary

###  What did I do ?
I built and compared **six unsupervised anomaly detection methods** on time series data, to find out whether a computer can learn what "normal" looks like and spot unusual periods **without ever seeing a label**. The labeled events were used only afterwards, to check the results.

- 📈 **Prediction-based:** AutoARIMA and NHITS forecast the next value and flag big misses.
- 🧠 **Reconstruction-based:** an LSTM autoencoder, a dense autoencoder, a variational autoencoder (VAE) and a GAN rebuild their input and flag what they rebuild badly.

### 📊 Data
- 🚕 **NYC taxi demand:** 10,320 readings, one every 30 minutes (June 2014 to January 2015), with 5 labeled unusual periods (10% of the data): the marathon, Thanksgiving, Christmas, New Year and a snowstorm.
- 📉 **M3 Quarterly series (Q1):** a small quarterly series of about 44 values.

### 🏆 Findings

| Method | Result |
|---|---|
| AutoARIMA | ✅ Flagged the sudden jump from about 3,380 to 4,950 |
| NHITS | ⚠️ Caught the marathon (error spike ~20,400) and New Year (~9,500), but missed 3 events |
| LSTM autoencoder | ❌ No usable signal (error flat at 0.989 to 1.000) |
| **Dense autoencoder** | 🥇 **Best:** rose at the marathon and Christmas and reached ~1.0 at the snowstorm |
| Variational autoencoder | ✅ Good: rose at the marathon and the snowstorm, with training loss down 36.6% |
| GAN | ❌ No usable signal (score only followed the daily cycle) |

**Key insights**
- 💡 A **simple model beat the complex ones.**
- 🌨️ **Big events** (the snowstorm, where trips fell to 8) are easy to detect. **Subtle ones** (Thanksgiving, about 12% below normal) are mostly missed.
- 🤝 When two different models agree, the detection is more trustworthy.
- 🔎 A busy-looking score is not proof. Always check against known events.

### 🛠️ Technologies
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Lightning](https://img.shields.io/badge/PyTorch_Lightning-792EE5?logo=lightning&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?logo=googlecolab&logoColor=white)

`statsforecast` · `neuralforecast` · `PyTorch Forecasting` · `PyOD` · `datasetsforecast` · `Matplotlib`

### 🔮 Next steps
Add precision, recall and ROC-AUC, compare against a simple rolling-average baseline, and test pre-trained time series models (such as TSPulse, MOMENT and Chronos) on real IoT sensor data.
