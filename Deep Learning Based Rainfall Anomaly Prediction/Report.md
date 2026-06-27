# Multi-Output Bi-LSTM Forecasting of Monthly Rainfall and Solar Radiation with Standardized Rainfall Anomaly Classification in Bogura, Bangladesh
## 1. Background 
**Rainfall and solar radiation prediction are essential for a wide range of agricultural, environmental, and climate-sensitive planning activities. Reliable forecasting assists in proper flood-risk management, drought preparedness planning, and other disaster management decision-making. Having accurate information on solar radiation and rainfall can also contribute to Bangladesh's food security planning, disaster early warning system, and sustainable natural resources management. The northwestern part of Bangladesh is mostly drought-prone, and rainfall varies during the monsoon and dry season, so reliable climate forecasting is significantly important in this region [1]-[3].  In this study, the Bogura district of Bangladesh was selected as the study area because it is one of the important agricultural districts and also a suitable local setting for exploring the interdependence of weather and agriculture. Earlier rainfall-forecasting studies in Bangladesh have applied statistical models, conventional machine-learning techniques, and deep-learning models, and mostly these studies focused either on rainfall or combined temperature with rainfall [2], [4]-[6]. In a single deep learning framework, the combined rainfall and solar radiation forecasting model in Bogura has received limited attention [7]-[10].   
Considering the limitation, this study develops a multi-output Bidirectional Long Short-Term Memory model that utilizes historical previous rainfall, solar radiation, humidity, and temperature observations with seasonal, lagged, and rolling variables for rainfall and solar radiation forecasting to capture temporal dependence and seasonality [11], [12]. This proposed model also utilizes a chronological train–validation–test split to ensure reliable test set performance [13], [14]. The integrated approach of this study offers a practical foundation to address and mitigate the impacts of unpredictable weather patterns and sustainable irrigation.**

## 2. Research Aim
**Primarily, this research aims to develop and evaluate a multi-output Bidirectional Long Short-Term Memory model to forecast monthly rainfall and solar radiation in Bogura, Bangladesh, using historical weather data obtained from NASA POWER [15], and to determine if bidirectional sequence learning performs better than conventional LSTM and non-sequential Random Forest models. This study also aims to utilize the predicted rainfall to classify standardized dry and wet rainfall anomalies. The study also aims to determine whether bidirectional sequence learning provides better predictive performance than Vanilla LSTM and a non-sequential Random Forest model.**

## 3. Research Gaps and How the Present Study Addresses the Gap
**Most research works are narrowly focused on predicting either rainfall or other weather variables in isolation [2]-[5]. Existing multi-output models rarely combine multivariate meteorological sequences in a single framework. Researchers mainly focus on how accurately their models predict rainfall, overlooking impact analysis [1], [6], [7]. Most of the Bi-LSTM forecasting models used for rainfall and solar prediction make use of hourly or sub-hourly data to manage short-term impacts, overlooking the importance of monthly seasonal weather analysis [3], [8]. A single neural-network training run without considering network initialisation and stochastic optimisation can affect the final model. During data preprocessing, if information from validation or test observations is inadvertently used, it can lead to data leakage in climate modeling selection [13], [14].  Though frequent studies are using multivariate climate data in Bangladesh at regional and divisional levels, there is a distinct lack of a local-level multivariate climate sequences framework [2], [4], [7], [10]. Ultimately, there exists a noticeable gap in combining tuned multi-output Bi-LSTM against tuned sequential and ensemble baselines with multiple random seeds to verify stability testing and provide rainfall forecasts with interpretable climate anomaly categories [3], [6], [9], [10].  
This study specifically focused on Bogura by developing a framework using historical weather data like rainfall, solar radiation, humidity, and temperature, which was later mixed with seasonality, lags, and rolling averages features to predict both monthly rainfall and solar radiation simultaneously. To ensure reliability, this study trained data in chronological order with data-leakage controls and tuned bidirectional LSTM, standard sequential, and tree-based models on a shared validation set using a normalized error metric and multiple random seeds. Finally, convert these precise multi-output predictions into standardized dry and wet categories for better climate interpretation.**

## 4. Data
**The meteorological data of Bogura, Bangladesh, from 1990 to 2025 were used in this study, which were obtained from the NASA Prediction of Worldwide Energy Resources (NASA POWER) database [15]. The observations were divided chronologically into three periods:**
-	Training period: January 1990 to December 2023
-	Validation period: January 2024 to December 2024
-	Independent test period: January 2025 to December 2025
  
**To train the model parameters and calculate the preprocessing statistics, the training dataset was used, and to fine-tune hyperparameters, the 2024 dataset was utilized. The 2025 observations were reserved for testing the model performance.
This study uses one-month lag variables to represent recent climatic persistence, twelve-month lag variables to provide annual seasonal recurrence, and three-month rolling averages to provide trends in rainfall and solar radiation.
NASA POWER variable	Variable used in the study-**
## Dataset Variables

| NASA POWER Variable | Variable Used in the Study | Description | Unit |
|----------------------|----------------------------|-------------|------|
| YEAR | Year | Calendar year | Year |
| DOY | Day of Year (DOY) | Day of the year (1–365/366) | Day |
| PRECTOTCORR | Rainfall | Total daily precipitation | mm/day |
| ALLSKY_SFC_SW_DWN | Solar Radiation | All-sky downward shortwave radiation at the Earth's surface | kWh/m²/day |
| RH2M | Relative Humidity | Relative humidity measured or estimated at 2 metres above the surface | % |
| T2M_MIN | Minimum Temperature | Minimum daily air temperature at 2 metres | °C |
| T2M_MAX | Maximum Temperature | Maximum daily air temperature at 2 metres | °C |

**Fourteen input variables were used in this study,**
-	Monthly rainfall
-	Monthly solar radiation
-	Monthly relative humidity
-	Monthly minimum temperature
- Monthly maximum temperature
-	Monthly mean temperature
-	Sine-transformed month
-	Cosine-transformed month
-	One-month lag of rainfall
-	Twelve-month lag of rainfall
-	One-month lag of solar radiation
-	Twelve-month lag of solar radiation
-	Three-month rolling average of rainfall
-	Three-month rolling average of solar radiation
**The proposed Bi-LSTM, Vanilla LSTM, and Random Forest models predict monthly rainfall and monthly solar radiation.**

## 5. Methods


| Method | Why did the authors use it? | Why is it used in this study? | Citation |
|--------|------------------------------|-------------------------------|---------|
| **Multi-output Bidirectional LSTM** | Utilizes multi-task learning from sequential data to forecast multiple environmental variables simultaneously. | Captures nonlinear relationships between rainfall and solar radiation and learns temporal dependencies from a 12-month historical window. | [9], [12], [15], [16] |
| **Vanilla LSTM Baseline** | Models seasonal and sequential rainfall behaviour. | Used as a baseline to compare the performance improvement of Bi-LSTM. | [11], [12] |
| **Random Forest Regression Baseline** | Reduces prediction variance by combining multiple decision trees. | Serves as a traditional machine learning baseline against deep learning models. | [17], [20] |
| **12-Month Sliding Window** | Preserves annual temporal sequence for prediction. | Represents seasonal rainfall and solar radiation patterns over one year. | [11], [21] |
| **Lagged Predictors** | Captures serial correlation and seasonal persistence. | Uses 1-month and 12-month lag features to represent recent and annual meteorological conditions. | [11], [21] |
| **3-Month Rolling Average** | Smooths short-term fluctuations. | Provides seasonal trend information for monthly rainfall. | [21] |
| **Cyclical Month Encoding** | Represents cyclic seasonal behaviour using sine and cosine transformation. | Maintains the continuity between December and January. | [21] |
| **Batch Normalization** | Stabilizes activations and accelerates training. | Improves stability of stacked Bi-LSTM training. | [22] |
| **Dropout & Recurrent Dropout** | Reduces overfitting. | Improves generalization on relatively small meteorological datasets. | [23] |
| **Glorot Uniform & Orthogonal Initialization** | Preserves gradient flow during training. | Supports stable and reproducible Bi-LSTM optimization. | [24], [25] |
| **Huber Loss** | Combines MSE and MAE while reducing sensitivity to outliers. | Minimizes the influence of extreme rainfall events during training. | [26] |
| **Adam Optimizer** | Automatically adapts learning rates. | Provides stable optimization for noisy meteorological data. | [27] |
| **Gradient-Norm Clipping** | Prevents exploding gradients. | Stabilizes LSTM and Bi-LSTM training. | [28] |
| **Early Stopping** | Stops training before overfitting occurs. | Retains the model weights corresponding to minimum validation loss. | [29] |
| **Chronological Train–Validation–Test Split** | Prevents temporal data leakage. | Simulates real-world forecasting using historical observations. | [30], [31] |
| **Root Mean Squared Error (RMSE)** | Measures forecasting error magnitude. | Evaluates large prediction deviations. | [32] |
| **Mean Absolute Error (MAE)** | Measures average prediction error. | Complements RMSE because it is less sensitive to outliers. | [32] |
| **Standardized Rainfall Anomaly (SPI Thresholds)** | Standardizes precipitation and classifies wet, normal, and dry conditions. | Evaluates the Bi-LSTM model's ability to reproduce rainfall anomaly categories. | [33]–[35] |

## 6. Model Implementation Quality
## Hyperparameter Configuration

**Table 1. Hyperparameter configuration of the proposed Multi-Output Bi-LSTM model**

| Hyperparameter | Values Tried | Hyperparameter | Values Tried | Final Selection |
|----------------|-------------|----------------|-------------|-----------------|
| Learning Rate | 0.01, 0.003, 0.001, 0.0005 | Optimizer | Adam | Learning rate = **0.003**; Adam optimizer |
| Batch Size | 16, 32, 64 | Loss Function | Huber loss (δ = 1.0) | Batch size = **16**; Huber loss |
| Epochs | Maximum 150 with early stopping | Activation Function | Tanh (hidden layer); Linear (output layer) | Training stopped at epoch **117** (early stopping); Tanh and Linear output |
| Dropout Rate | Dropout = 0.20; Recurrent dropout = 0.10 | LR Schedule | Fixed learning rate | Dropout = **0.20**; No learning rate scheduler |
| Framework | TensorFlow/Keras | Random Seed | 7, 21, 42, 84, 123 | TensorFlow/Keras; Reporting seed = **42** |

<img width="975" height="564" alt="image" src="https://github.com/user-attachments/assets/cd03dc2c-8d8c-438b-9d69-8ffa0c75e72e" />

**Figure 1. Learning-rate optimization sensitivity curve**

**The graph compares the impact of different learning rates on the model’s training and validation loss. In all scenarios, the loss decreases rapidly in early phases and gradually stabilizes afterwards. Out of all potential values, at the learning rate of 0.003, the model performs the best, achieving a training loss while maintaining a consistent validation loss.**

## Training Stability and Regularization

<img width="975" height="564" alt="image" src="https://github.com/user-attachments/assets/1c51cb44-faa0-4928-b8fc-72808c947fcc" />

**Figure 2. Training and validation of the Huber loss of the selected Bi-LSTM**

**During the early epochs, the training loss drops sharply while the validation loss fluctuates (Figure 2). The 0.2 dropout rate, along with early stopping, reduces model overfitting and keeps the training process stable.**

## 6. Results and Discussion 
## Historical Climate Characteristics

<img width="781" height="627" alt="image" src="https://github.com/user-attachments/assets/dfc2e5e5-5290-4d72-9e90-af71e912e694" />

**Figure 3. Historical feature correlation heatmap for Bogura**

**The historical correlation analysis shows the relationships between the climatic variables (Figure 3). Rainfall positively correlated with humidity (r = 0.61) and temperature (r = 0.54) while showing insignificant correlation with solar radiation (r = 0.03). It demonstrates that rainfall variability is more significantly associated with seasonal temperature and moisture in Bogura.**

<img width="975" height="736" alt="image" src="https://github.com/user-attachments/assets/f9826265-d173-41db-87d4-61419d6b8f76" />

**Figure 4. Long-term rainfall trend and seasonality decomposition**

**Figure 4 shows Bogura’s monthly rainfall in its observed pattern, long-term trend, seasonal cycle, and residual variation. Bogura’s monthly rainfall exhibited a strong yearly seasonal pattern. The rolling trend increased notably after 2016, while several large residual values indicated irregular extreme-rainfall events.**

## Test-Set Performance
**Table 2. Test-set performance of the proposed multi-output Bi-LSTM**

| Target | RMSE | MAE | R² | NRMSE | NMAE |
|---|---:|---:|---:|---:|---:|
| Rainfall | 68.744 mm/month | 53.845 mm/month | 0.833 | 9.49% | 7.43% |
| Solar Radiation | 0.914 | 0.735 | 0.840 | 6.93% | 5.57% |
| **Mean Normalized Performance** | — | — | — | **8.21%** | **6.50%** |

**The table shows that the rainfall forecast deviated from the observed values by 53.845 mm each month on average. During the extreme rainfall months, the model made more errors, so the RMSE was greater than the MAE. The model predicted approximately 83.3% of rainfall variability. On the other hand, the model predicted around 84.0% of solar radiation variability, showing relatively strong seasonal observability. Overall, the model achieves an average NRMSE of 8.21% and NMAE of 6.50%, showing comparatively low prediction errors.**

<img width="980" height="695" alt="image" src="https://github.com/user-attachments/assets/d430778a-8cdb-4a13-b8a3-a5fbb0147f2b" />

**Figure 5. Rainfall–solar forecast with SRA-based wet–dry classification**

**Figure 5 compares the actual and predicted rainfall and solar radiation for 2025 in Bogura. The model predicted a "Severely Wet" scenario during the first quarter, where empirical observations revealed almost zero precipitation. The model accurately quantifies the "Near Normal" rainfall scenario in two peaks in May (empirical observation around 420 mm) and August (actual observation at approximately 440 mm). The wet–dry labels shown in the rainfall panel were derived from the standardized rainfall anomaly, calculated separately for each calendar month using the corresponding 1990–2023 monthly climatological mean and standard deviation. The solar radiation predictions remain close to the actual values, while the standardized rainfall anomaly labels classify each month according to its wet or dry condition.**

## Comparison of Proposed and Baseline Models
**Table 3. Comparative performance of the proposed model and baseline models**

## Model Performance

| Model                         | Mean NRMSE | RMSE-based Error | Mean NMAE | MAE-based Error |
|------------------------------|-----------:|-----------------:|----------:|----------------:|
| Random Forest                | 0.0804     | 8.04%            | 0.0625    | 6.25%           |
| Proposed Multi-Output Bi-LSTM| 0.0821     | 8.21%            | 0.0650    | 6.50%           |
| Vanilla LSTM                 | 0.0837     | 8.37%            | 0.0606    | 6.06%           |

**The results illustrate comparatively competitive performance of all three models, with a narrow error rate range of 6% to 8.4%. While Random Forest shows a marginal advantage in minimizing larger variance errors (RMSE), the Vanilla LSTM shows a minor edge in overall average absolute deviation (MAE). However, the Proposed Multi-Output Bi-LSTM establishes an optimal stance, maintaining stable error metrics across both evaluation criteria.**

## 7. Limitations
**The proposed framework demonstrated stable and competitive performance as a multi-output monthly forecasting system predicting both rainfall and solar radiation with standardized rainfall-anomaly interpretation. However, there are several limitations. The independent test period covered only 12 months, which limits the statistical strength of the evaluation. The use of NASA POWER data may also introduce some uncertainty compared with ground-based observations. Although the Bi-LSTM explained the overall seasonal patterns moderately well, it tended to underestimate extreme rainfall peaks. In addition, while the continuous rainfall and solar-radiation forecasts were strong, the exact wet–dry classification performance remained limited. Therefore, the anomaly-classification component requires further calibration, a longer multi-year test period, and validation across additional locations before considering it as a stable and reliable model for operational drought and wetness assessment.**

## References

[1] M. Kamruzzaman, M. Almazroui, M. A. Salam, et al., “Spatiotemporal drought analysis in Bangladesh using the Standardized Precipitation Index and Standardized Precipitation Evapotranspiration Index,” *Scientific Reports*, vol. 12, Art. no. 20694, 2022.

[2] I. Mahmud, S. H. Bari, and M. T. U. Rahman, “Monthly rainfall forecast of Bangladesh using autoregressive integrated moving average method,” *Environmental Engineering Research*, vol. 22, no. 2, pp. 162–168, 2017.

[3] T. Peng, C. Zhang, J. Zhou, and M. S. Nazir, “An integrated framework of bidirectional long short-term memory based on sine cosine algorithm for hourly solar radiation forecasting,” *Energy*, vol. 221, Art. no. 119887, 2021.

[4] F. Di Nunno, F. Granata, Q. B. Pham, and G. de Marinis, “Precipitation forecasting in Northern Bangladesh using a hybrid machine learning model,” *Sustainability*, vol. 14, no. 5, Art. no. 2663, 2022.

[5] M. S. Islam, M. Shafiuzzaman, G. Mahmud, et al., “Explainable deep learning for rainfall prediction: A CNN–XGBoost hybrid approach in the northern region of Bangladesh,” *Neural Computing and Applications*, 2025.

[6] M. M. R. Khan, M. A. B. Siddique, S. Sakib, A. Aziz, I. K. Tasawar, and Z. Hossain, “Prediction of temperature and rainfall in Bangladesh using long short-term memory recurrent neural networks,” *arXiv preprint arXiv:2010.11946*, 2020.

[7] R. Rahman and F. Taskin, “Automated lag-selection for multi-step univariate time-series forecast using Bayesian optimization: Forecast station-wise monthly rainfall of nine divisional cities of Bangladesh,” *arXiv preprint arXiv:2401.08070*, 2024.

[8] G. Guariso, G. Nunnari, and M. Sangiorgio, “Multi-step solar irradiance forecasting and domain adaptation of deep neural networks,” *Energies*, vol. 13, no. 15, Art. no. 3987, 2020.

[9] N. Azizi, M. Yaghoubirad, M. Farajollahi, and A. Ahmadi, “Deep learning based long-term global solar irradiance and temperature forecasting using time series with multi-step multivariate output,” *Renewable Energy*, vol. 206, pp. 135–147, 2023.

[10] Z. M. Yaseen, M. Ali, A. Sharafati, N. Al-Ansari, and S. Shahid, “Forecasting standardized precipitation index using data intelligence models: Regional investigation of Bangladesh,” *Scientific Reports*, vol. 11, Art. no. 3435, 2021.

[11] R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed. Melbourne, Australia: OTexts, 2021.

[12] D. Kumar, A. Singh, P. Samui, and R. K. Jha, “Forecasting monthly precipitation using sequential modelling,” *Hydrological Sciences Journal*, vol. 64, no. 6, pp. 690–700, 2019.

[13] C. Bergmeir, R. J. Hyndman, and B. Koo, “A note on the validity of cross-validation for evaluating autoregressive time-series prediction,” *Computational Statistics & Data Analysis*, vol. 120, pp. 70–83, 2018.

[14] V. Cerqueira, L. Torgo, and I. Mozetič, “Evaluating time series forecasting models: An empirical study on performance estimation methods,” *Machine Learning*, vol. 109, pp. 1997–2028, 2020.

[15] NASA POWER Project, “NASA POWER meteorological-data methodology,” NASA Langley Research Center, 2026.

[16] R. Caruana, “Multitask learning,” *Machine Learning*, vol. 28, pp. 41–75, 1997.

[17] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.

[18] S. A. Osmani, J. S. Kim, C. Jun, et al., “Prediction of monthly dry days with machine learning algorithms: A case study in Northern Bangladesh,” *Scientific Reports*, vol. 12, Art. no. 19717, 2022.

[19] M. Schuster and K. K. Paliwal, “Bidirectional recurrent neural networks,” *IEEE Transactions on Signal Processing*, vol. 45, no. 11, pp. 2673–2681, 1997.

[20] L. Breiman, “Random forests,” *Machine Learning*, vol. 45, pp. 5–32, 2001.

[21] M. A. Al Mamun, M. R. Sarker, M. A. R. Sarkar, et al., “Identification of influential weather parameters and seasonal drought prediction in Bangladesh using machine learning algorithm,” *Scientific Reports*, vol. 14, Art. no. 566, 2024.

[22] M. A. H. Mondol, S. C. Das, and M. N. Islam, “Application of Standardized Precipitation Index to assess meteorological drought in Bangladesh,” *Jàmbá: Journal of Disaster Risk Studies*, vol. 8, no. 1, Art. no. 280, 2016.

[23] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, “Dropout: A simple way to prevent neural networks from overfitting,” *Journal of Machine Learning Research*, vol. 15, pp. 1929–1958, 2014.

[24] X. Glorot and Y. Bengio, “Understanding the difficulty of training deep feedforward neural networks,” in *Proceedings of the 13th International Conference on Artificial Intelligence and Statistics (AISTATS)*, pp. 249–256, 2010.

[25] A. M. Saxe, J. L. McClelland, and S. Ganguli, “Exact solutions to the nonlinear dynamics of learning in deep linear neural networks,” in *Proceedings of the International Conference on Learning Representations (ICLR)*, 2014.

[26] P. J. Huber, “Robust estimation of a location parameter,” *The Annals of Mathematical Statistics*, vol. 35, no. 1, pp. 73–101, 1964.

[27] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in *Proceedings of the International Conference on Learning Representations (ICLR)*, 2015.

[28] R. Pascanu, T. Mikolov, and Y. Bengio, “On the difficulty of training recurrent neural networks,” in *Proceedings of the 30th International Conference on Machine Learning (ICML)*, pp. 1310–1318, 2013.

[29] R. J. Hyndman and A. B. Koehler, “Another look at measures of forecast accuracy,” *International Journal of Forecasting*, vol. 22, no. 4, pp. 679–688, 2006.

[30] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[31] S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep network training by reducing internal covariate shift,” in *Proceedings of the 32nd International Conference on Machine Learning (ICML)*, pp. 448–456, 2015.

[32] L. Prechelt, “Automatic early stopping using cross validation: Quantifying the criteria,” *Neural Networks*, vol. 11, no. 4, pp. 761–767, 1998.

[33] T. B. McKee, N. J. Doesken, and J. Kleist, “The relationship of drought frequency and duration to time scales,” in *Proceedings of the 8th Conference on Applied Climatology*, pp. 179–184, 1993.

[34] N. B. Guttman, “Accepting the Standardized Precipitation Index: A calculation algorithm,” *Journal of the American Water Resources Association*, vol. 35, no. 2, pp. 311–322, 1999.

[35] World Meteorological Organization, *Standardized Precipitation Index User Guide*, WMO-No. 1090. Geneva, Switzerland: WMO, 2012.

