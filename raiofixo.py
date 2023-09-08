import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pykalman import KalmanFilter
from sklearn.metrics import r2_score
from scipy import signal
import copy
from dtw import dtw,accelerated_dtw
from scipy.signal import hilbert, butter, filtfilt
from scipy.fftpack import fft,fftfreq,rfft,irfft,ifft
import math
import os
os.environ["R_HOME"] = r"C:\Program Files\R\R-4.3.0" # change as needed
import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.vectors import StrVector
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation



def compute_rqa_metrics(series_de_entrada,raios_x):
    data_points = series_de_entrada
    time_series = TimeSeries(data_points,
                                 embedding_dimension=2,
                                 time_delay=2)
    settings = Settings(time_series,
                            analysis_type=Classic,
                            neighbourhood=FixedRadius(raios_x),#0.65 default
                            similarity_measure=EuclideanMetric,
                            theiler_corrector=1)
    computation = RQAComputation.create(settings,
                                            verbose=True)
    result = computation.run()
    result.min_diagonal_line_length = 2
    result.min_vertical_line_length = 2
    result.min_white_vertical_line_length = 2
    
    # Construir um dicionário com as métricas e seus valores
    metrics_dict = {
        "L_min": result.min_diagonal_line_length,
        "V_min": result.min_vertical_line_length,
        "W_min": result.min_white_vertical_line_length,
        "RR": result.recurrence_rate,
        "DET": result.determinism,
        "L": result.average_diagonal_line,
        "L_max": result.longest_diagonal_line,
        "DIV": result.divergence,
        "L_entr": result.entropy_diagonal_lines,
        "LAM": result.laminarity,
        "TT": result.trapping_time,
        "V_max": result.longest_diagonal_line,
        "V_entr": result.longest_vertical_line,
        "W": result.average_white_vertical_line,
        "W_max": result.longest_white_vertical_line,
        "W_div": result.longest_white_vertical_line_inverse,
        "W_entr": result.entropy_white_vertical_lines,
        "DET_RR": result.ratio_determinism_recurrence_rate,
        "LAM_DET": result.ratio_laminarity_determinism,
        "NRLINE": result.number_of_recurrence_points,
        "LAM": result.laminarity,
        "matrix":result.recurrence_matrix_reverse
    }
    
    # Retornar o dicionário com as métricas calculadas
    return metrics_dict

robjects.r('entrada <- read.csv("Kalman.csv",sep=",",header = TRUE)')
robjects.r('''
syncTmaxA <-entrada$syncTmaxA
syncTair <-entrada$syncTair
syncHumidity <-entrada$syncHumidity
syncWindspeed <-entrada$syncWindspeed
syncIncSolar <-entrada$syncIncSolar
syncIa <-entrada$syncIa

TmaxA_kalman<-syncTmaxA
Tair_kalman<-syncTair
Humidity_kalman<-syncHumidity
Windspeed_kalman<-syncWindspeed
IncSolar_kalman<-syncIncSolar
Ia_kalman<-syncIa
''')




robjects.r('entrada <- read.csv("Kalman.csv",sep=",",header = TRUE)')
robjects.r('''
syncTmaxA <-entrada$syncTmaxA
syncTair <-entrada$syncTair
syncHumidity <-entrada$syncHumidity
syncWindspeed <-entrada$syncWindspeed
syncIncSolar <-entrada$syncIncSolar
syncIa <-entrada$syncIa

TmaxA_kalman<-syncTmaxA
Tair_kalman<-syncTair
Humidity_kalman<-syncHumidity
Windspeed_kalman<-syncWindspeed
IncSolar_kalman<-syncIncSolar
Ia_kalman<-syncIa
''')

robjects.r('''
time_series_decomposition <- function(x) {
  # Calculate the spectrum
  espectro <- spec.pgram(x, plot = FALSE)
  frequencia <- 1 / espectro$freq[which.max(espectro$spec)]
  
  # Convert to time series with appropriate frequency
  x <- ts(x, frequency = frequencia)
  
  # Perform time series decomposition
  if (frequencia > 1) {
    decomposicao <- decompose(x)
    S1 <- na.omit(decomposicao$seasonal)
    T1 <- na.omit(decomposicao$trend)
    R1 <- na.omit(decomposicao$random)
  } else {
    S1 <- T1 <- R1 <- 0
  }
  
  R1X <- R1
 
  return(R1X)
}

# Example usage
''')
robjects.r('''

TmaxArandom_component <- time_series_decomposition(syncTmaxA)
Tairrandom_component <- time_series_decomposition(syncTair)
Humidityrandom_component <- time_series_decomposition(syncHumidity)          
#Windspeedrandom_component <- time_series_decomposition(syncWindspeed)
IncSolarrandom_component <- time_series_decomposition(syncIncSolar)
Iarandom_component <- time_series_decomposition(syncIa)           
''')
  
minEpslonTmaxA = robjects.r('''minimo_epslon_ruido1 <- 5*(sqrt(sum(TmaxArandom_component^2) /  length(TmaxArandom_component)))''')
minEpslonTair = robjects.r('''minimo_epslon_ruido1 <- 5*(sqrt(sum(Tairrandom_component^2) / length(Tairrandom_component)))''')
minEpslonHumidity = robjects.r('''minimo_epslon_ruido1 <- 5*(sqrt(sum(Humidityrandom_component^2) / length(Humidityrandom_component)))''')
#minEpslonWindspeed = robjects.r('''minimo_epslon_ruido1 <- 5*(sqrt(sum(Windspeedrandom_component^2) / length(Windspeedrandom_component)))''')
minEpslonIncSolar = robjects.r('''minimo_epslon_ruido1 <- 5*(sqrt(sum(IncSolarrandom_component^2) / length(IncSolarrandom_component)))''')
minEpslonIa = robjects.r('''minimo_epslon_ruido1 <- 5*(sqrt(sum(Iarandom_component^2) / length(Iarandom_component)))''')

robjects.r('''
# Função para calcular o valor limite
calculate_limit_value <- function(component) {
  mean_component <- mean(component)
  max_component <- max(component)
  
  limit_value <- min(0.1 * mean_component, max_component)
  
  return(limit_value)
}
''')

# Exemplo de uso para cada componente
limit_TmaxA_sup = robjects.r('''limit_TmaxA <- calculate_limit_value(TmaxArandom_component)''')
limit_Tair_sup = robjects.r('''limit_Tair <- calculate_limit_value(Tairrandom_component)''')
limit_Humidity_sup = robjects.r('''limit_Humidity <- calculate_limit_value(Humidityrandom_component)''')
#limit_Windspeed_sup = robjects.r('''limit_Windspeed <- calculate_limit_value(Windspeedrandom_component)''')
limit_IncSolar_sup = robjects.r('''limit_IncSolar <- calculate_limit_value(IncSolarrandom_component)''')
limit_Ia_sup = robjects.r('''limit_Ia <- calculate_limit_value(Iarandom_component)''')

import pandas as pd
print(limit_Ia_sup)
# Valores de raios a serem testados
#raios_x = np.linspace(minEpslonTmaxA, limit_TmaxA_sup)  # Você pode ajustar o número de pontos aqui
'''
# Criar uma lista para armazenar os resultados dos raios
all_results = []

# Loop while para calcular as métricas com diferentes raios
r = minEpslonTmaxA
while r < limit_TmaxA_sup:
    
    
    result_rqa_TmaxA = compute_rqa_metrics(TmaxA, r)
    result_rqa_Tair = compute_rqa_metrics(Tair, r)
    result_rqa_Humidity = compute_rqa_metrics(Humidity, r)
    result_rqa_Windspeed = compute_rqa_metrics(Windspeed, r)
    result_rqa_IncSolar = compute_rqa_metrics(IncSolar, r)
    result_rqa_Ia = compute_rqa_metrics(Ia, r)
    
    # Armazenar os resultados em um dicionário
    results = {
        'Raio': r,
        'Tair': result_rqa_Tair,
        'Humidity': result_rqa_Humidity,
        'Windspeed': result_rqa_Windspeed,
        'IncSolar': result_rqa_IncSolar,
        'Ia': result_rqa_Ia,
        'TmaxA': result_rqa_TmaxA
    }
    
    all_results.append(results)
    
    r += 0.1

# Criar um DataFrame a partir dos resultados
result_df = pd.DataFrame(all_results)

# Salvar o DataFrame em um arquivo CSV
result_df.to_csv("rqa_results.csv", index=False)
'''