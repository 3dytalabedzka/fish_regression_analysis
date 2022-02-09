import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.stats import kstest, jarque_bera, norm, t, anderson, shapiro
from statsmodels.distributions.empirical_distribution import ECDF
import statsmodels.api as sm
#import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("Fish.csv")

#sn.heatmap(data.corr(), annot=True)
#plt.show()

X = np.array(data.loc[:, "Width"]) #data["Width"]
Y = np.array(data.loc[:, "Weight"]) #data["Weight"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
#print('X_train: ', np.shape(X_train))
#print('Y_train: ', np.shape(Y_train))
#print('X_test: ', np.shape(X_test))
#print('Y_test: ', np.shape(Y_test))

def scatter_plot(x, y):
    plt.scatter(x, y, s=5, color=(0.13, 0.7, 0.67))
    plt.grid(color='lightgray',linestyle='--')
    plt.xlabel("Szerokość [cm]")
    plt.ylabel("{}".format(r'$\sqrt{\mathrm{Waga [g]}}$'))
    plt.title("Wykres pierwiastka z wagi ryby w zależności od jej szerokości")
    plt.show()

def regression_coef(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    b1 = np.sum([ (y[i] - mean_y) * x[i] for i in range(len(x)) ]) / np.sum([ (x[i] - mean_x)**2 for i in range(len(x)) ])
    b0 = mean_y - b1*mean_x

    return b1, b0

def regression_plot(x, y, b0, b1, title, ylabel):
    X = np.linspace(min(x), max(x), 1000)
    plt.plot(X, b0 + b1*X, color=(0.0, 0.25, 0.42))
    plt.scatter(x, y, s=5, color=(0.13, 0.7, 0.67))
    plt.grid(color='lightgray',linestyle='--')
    plt.title(title)
    plt.xlabel("Szerokość [cm]")
    plt.ylabel(ylabel)
    plt.show()

def anal_errors(e):
    mu = np.mean(e)
    sigma = np.std(e)
    print('Średnia e_i: {}, Sigma e_i: {}'.format(mu, sigma))

    print(kstest( (e-mu)/sigma , 'norm'))
    print('JarqueBeraResult',jarque_bera(e))
    print(anderson(e, dist='norm'))
    print("Shapiro", shapiro(e))
    
    n = len(e)
    x = np.linspace(1, n, n)
    plt.plot(x, e, color=(0.0, 0.25, 0.42))
    plt.title('Wykres residuów {}'.format(r'$e_i$'))
    plt.grid(color='lightgray',linestyle='--')
    plt.xlabel("i")
    plt.ylabel('{}'.format(r'$e_i$'))
    plt.hlines(y=0, xmin=1, xmax=n, linestyles='--', color=(0.13, 0.7, 0.67))
    plt.show()

    plt.boxplot(e)
    plt.title('Wykres pudełkowy residuów {}'.format(r'$e_i$'))
    plt.axhline(0, linestyle='--', color=(0.13, 0.7, 0.67))
    plt.show()

    #plt.acorr(e, color=(0.0, 0.25, 0.42))
    sm.graphics.tsa.plot_acf(np.array(e), lags=40, color=(0.0, 0.25, 0.42), title="Wykres autokorelacji residuów z przedziałami ufności")
    plt.grid(color='lightgray',linestyle='--')
    plt.show()

    x2 = np.linspace(min(e), max(e), 1000)
    ecdf_1 = ECDF(e)
    plt.title('Porównanie dystrybuanty rozkładu residuów z rozkładem normalnym')
    plt.plot(x2, ecdf_1(x2), color=(0.0, 0.25, 0.42), label='Dystrybuanta empiryczna residuów')
    plt.plot(x2, norm.cdf(x2, mu, sigma), color=(0.13, 0.7, 0.67), label='Dystrybuanta rozkładu normalnego')
    plt.grid(color='lightgray',linestyle='--')
    plt.legend()
    plt.show()

    plt.title('Porównanie histogramu prawdopodobieństwa residuów z gęstością rozkładu normalnego')
    plt.hist(e, bins=30, density=True, color=(0.0, 0.25, 0.42), label='Histogram prawdopodobieństwa residuów' )
    plt.plot(x2, norm.pdf(x2, mu ,sigma), color=(0.13, 0.7, 0.67), label='Gęstość rozkładu normalnego')
    plt.grid(color='lightgray',linestyle='--')
    plt.legend()
    plt.show()

def clean_data(X, Y):
    b1, b0 = regression_coef(X, Y)
    print("Współczynniki b0: {}, b1: {}".format(b0, b1))
    regression_plot(X, Y, b0, b1, "Regresja liniowa dla danych", "Waga [g]")

    est_Y = b0 + b1*X
    R_2 = np.sum( (est_Y - np.mean(Y))**2 ) / np.sum( (Y - np.mean(Y))**2 )
    print('R^2 wynosi {}'.format(R_2))
    
    #e = Y - est_Y
    #anal_errors(e)

    clean_X = X
    clean_Y = np.sqrt(Y)
    clean_b1, clean_b0 = regression_coef(clean_X, clean_Y)
    print("Współczynniki po transformacji b0: {}, b1: {}".format(clean_b0, clean_b1))
    regression_plot(clean_X, clean_Y, clean_b0, clean_b1, "Regresja liniowa dla danych po transformacji", "{}".format(r'$\sqrt{\mathrm{Waga [g]}}$'))
    
    c_est_Y = [clean_b0 + clean_b1*x for x in clean_X]
    R_2 = np.sum( (c_est_Y - np.mean(clean_Y))**2 ) / np.sum( (clean_Y - np.mean(clean_Y))**2 )
    print('R^2 po transformacie wynosi {}'.format(R_2))

    clean_e = [clean_Y[i] - c_est_Y[i] for i in range(len(clean_Y))]
    anal_errors(clean_e)

    #df_describe = pd.DataFrame(clean_e)
    #print('e', df_describe.describe())

    Q1 = np.percentile(clean_e, 25)
    Q3 = np.percentile(clean_e, 75)
    IQR = Q3-Q1
    count = 0
    for i in range(len(clean_e)):
        if not((Q1 - 1.5*IQR <= clean_e[i]) & (Q3 + 1.5*IQR >= clean_e[i])):
            count += 1
    print('{} obserwacji odstających'.format(count))


def confidence_intervals(X, Y, alpha = 0.05):
    est_b1, est_b0 = regression_coef(X, Y)
    n = len(X)

    est_Y = [est_b0 + est_b1*x for x in X]
    s = np.sqrt( np.sum( [(Y[i] - est_Y[i])**2 for i in range(len(Y))]) / (n-2) )
    T = np.sum( [ (X[i] - np.mean(X))**2 for i in range(n)] )

    min_b0 = est_b0 - t.ppf(1 - alpha/2, n-2)*s*np.sqrt( (1/n) + np.mean(X)**2/T )
    max_b0 = est_b0 + t.ppf(1 - alpha/2, n-2)*s*np.sqrt( (1/n) + np.mean(X)**2/T )
    print('Przedział ufności b0: ({}, {})'.format(min_b0, max_b0))

    min_b1 = est_b1 - t.ppf(1 - alpha/2, n-2)*s*1/np.sqrt(T)
    max_b1 = est_b1 + t.ppf(1 - alpha/2, n-2)*s*1/np.sqrt(T)
    print('Przedział ufności b1: ({}, {})'.format(min_b1, max_b1))


def predict(X_train, Y_train, X_test, Y_test, alpha=0.05):
    b1, b0 = regression_coef(X_train, Y_train)
    Y_pred = b0 + b1*X_test

    R_2 = np.sum( (Y_pred - np.mean(Y_test))**2 ) / np.sum( (Y_test - np.mean(Y_test))**2 )
    print('R^2 wynosi {}'.format(R_2))

    plt.scatter(X_test, Y_test, color=(0.13, 0.7, 0.67), s=5, label="Dane do testu")
    plt.scatter(X_test, Y_pred, color=(0.0, 0.25, 0.42), s=5, label="Prognozowane wartości")
    plt.grid(color='lightgray',linestyle='--')
    plt.legend()
    plt.title("Porównanie wartości testowych z prognozowanymi")
    plt.xlabel("Szerokość [cm]")
    plt.ylabel("{}".format(r'$\sqrt{\mathrm{Waga [g]}}$'))
    plt.show()

    n = len(X_test)
    s = np.sqrt( np.sum( [(Y_test[i] - Y_pred[i])**2 for i in range(len(Y_test))]) / (n-2) )
    T = np.sum( [ (X_test[i] - np.mean(X_test))**2 for i in range(n)] )
    
    plt.scatter(X_test, Y_test, s=5, color=(0.13, 0.7, 0.67), label="Dane do testu")
    plt.scatter(X_test, Y_pred, s=5, color=(0.0, 0.25, 0.42), label="Wartości wyestymowane")

    pred_max = []
    pred_min = [] 
    X_test = np.sort(X_test)  
    for i in range(len(X_test)):
        est_Y_2 = b0 + b1*X_test[i]
        pred_max.append(est_Y_2 + t.ppf(1 - alpha/2, n-2)*s*np.sqrt(1+(1/n)+ (X_test[i]-np.mean(X_test))**2/(T) ) )
        pred_min.append(est_Y_2 - t.ppf(1 - alpha/2, n-2)*s*np.sqrt(1+(1/n)+ (X_test[i]-np.mean(X_test))**2/(T) ) )

    plt.plot(X_test, pred_max, color=(0.0, 0.25, 0.42))
    plt.plot(X_test, pred_min, color=(0.0, 0.25, 0.42))
    #plt.vlines(X_test, pred_min, pred_max, color=(0.0, 0.25, 0.42))
    plt.legend()
    plt.grid(color='lightgray',linestyle='--')
    plt.xlabel("Szerokość [cm]")
    plt.ylabel("{}".format(r'$\sqrt{\mathrm{Waga [g]}}$'))
    plt.title("Przedziały ufności dla estymowanych wartości Y")
    plt.show()

#scatter_plot(X_train, Y_train)
#scatter_plot(X_train, np.sqrt(Y_train))
#clean_data(X_train, Y_train)
#confidence_intervals(X_train, np.sqrt(Y_train))
predict(X_train, np.sqrt(Y_train), X_test, np.sqrt(Y_test))

#df_describe_x = pd.DataFrame(X_train)
#print('X', df_describe_x.describe())
#df_describe_y = pd.DataFrame(Y_train)
#print('Y', df_describe_y.describe())

