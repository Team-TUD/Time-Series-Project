import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression

# x = [
#     [72500, 72500, 5000],
#     [70000, 70000, 10000],
#     [67500, 67500, 15000],
#     [65000, 65000, 20000],
#     [62500, 62500, 25000],
#     [60000, 60000, 30000],
#     [57500, 57500, 35000],
#     [55000, 55000, 40000],
#     [52500, 52500, 45000],
#     [50000, 50000, 50000],
# ]

x = [
    [72.5, 72.5, 5],
    [70, 70, 10],
    [67.5, 67.5, 15],
    [65, 65, 20],
    [62.5, 62.5, 25],
    [60, 60, 30],
    [57.5, 57.5, 35],
    [55, 55, 40],
    [52.5, 52.5, 45],
    [50, 50, 50],
]

x_s = [
    [72.5],
    [70],
    [67.5],
    [65],
    [62.5],
    [60],
    [57.5],
    [55],
    [52.5],
    [50],
]

y = [
    [0.174, 0.040, 48],
    [0.171, 0.039, 55],
    [0.160, 0.039, 62],
    [0.103, 0.038, 68],
    [0.141, 0.039, 75],
    [0.107, 0.039, 89],
    [0.101, 0.038, 90],
    [0.088, 0.038, 104],
    [0.096, 0.037, 114],
    [0.141, 0.039, 125],
]

y_d = [[0.174], [0.171], [0.160], [0.103], [0.141], [0.107], [0.101], [0.088], [0.096], [0.141]]
y_p = [[0.040], [0.039], [0.039], [0.038], [0.039], [0.039], [0.038], [0.038], [0.037], [0.039]]
y_t = [[48], [55], [62], [68], [75], [89], [90], [104], [114], [125]]

y_t_p = [[61.6], [56], [50.7], [45.6], [40.3], [29.1], [27.7], [17.1], [8.8]]
y_d_p = [[-23.9], [-21.7], [-13.9], [26.6], [-0.2], [23.7], [28.2], [37.5], [32]]
y_p_p = [[-4.3], [-0.9], [-0.9], [2.6], [-1.7], [0], [2.6], [2.6], [3.4]]

x, y, y_d, y_p, y_t, y_t_p, y_d_p, y_p_p = np.array(x), np.array(y), np.array(y_d), np.array(y_p), np.array(y_t), np.array(y_t_p), np.array(y_d_p), np.array(y_p_p)

model = LinearRegression().fit(x, y)
model_d = LinearRegression().fit(x, y_d)
model_p = LinearRegression().fit(x, y_p)
model_t = LinearRegression().fit(x, y_t)

r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

plt.scatter(x_s, y_d,  color='black')
plt.plot(x_s, model_d.predict(x), color='blue', linewidth=3)
plt.xlabel('iterations for first two phases')
plt.ylabel('discriminative score')
plt.show()

plt.scatter(x_s, y_p,  color='black')
plt.plot(x_s, model_p.predict(x), color='blue', linewidth=3)
plt.xlabel('iterations for first two phases')
plt.ylabel('predictive score')
plt.show()

plt.scatter(x_s, y_t,  color='black')
plt.plot(x_s, model_t.predict(x), color='blue', linewidth=3)
plt.xlabel('iterations for first two phases')
plt.ylabel('time (in minutes)')
plt.show()

df_stock = pd.DataFrame(
    {'configuration': ["C1", "C1", "C1", "C2", "C2", "C2", "C3", "C3", "C3", "C4", "C4", "C4", "C5", "C5", "C5", "C6", "C6", "C6", "C7", "C7", "C7", "C8", "C8", "C8", "C9", "C9", "C9", "C1", "C1", "C1", "C2", "C2", "C2", "C3", "C3", "C3", "C4", "C4", "C4", "C5", "C5", "C5", "C6", "C6", "C6", "C7", "C7", "C7", "C8", "C8", "C8", "C9", "C9", "C9", "C1", "C1", "C1", "C2", "C2", "C2", "C3", "C3", "C3", "C4", "C4", "C4", "C5", "C5", "C5", "C6", "C6", "C6", "C7", "C7", "C7", "C8", "C8", "C8", "C9", "C9", "C9", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B"],
     'metric': ['discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'time', 'time', 'time', 'time', 'time', 'time', 'time','time', 'time', 'time', 'time', 'time', 'time', 'time', 'time', 'time','time', 'time', 'time', 'time', 'time', 'time', 'time', 'time', 'time','time', 'time', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'predictive', 'predictive', 'predictive', 'time', 'time', 'time'],
     'performance': [18.1, -63.9, -27.8, 21.2, -38.5, -54.6, 44.2, -18, -95.4, 49.1, 32.8, -4.6, 63.7, -108.2, 19.4, 44.7, 10.7, 18.5, 51.8, 13.1, 18.5, 52.2, 32, 32.4, 61.1, 30.3, -5.6, -5.3, 7.3, -16.2, 2.6, -2.4, -2.7, 0, 2.4, -5.4, 0, 9.8, -2.7, 2.6, 4.9, -13.5, -5.3, 9.8, -5.4, 2.6, 7.3, -2.7, 0, 7.3, 0, 2.6, 9.8, -2.7, 61, 57.1, 65.7, 56.9, 47.3, 62.1, 47.2, 46.4, 57.1, 44.7, 38.4, 52.1, 39.8, 33, 46.4, 17.9, 26.8, 40.7, 26.0, 19.6, 35.7, 10.6, 8, 30, -7.3, 6.3, 25, 23.3, 24.0, 13.3, -60.6, 4.3, -6, 1.7, -12, 10.4, 1.6]
     })

df_sine = pd.DataFrame(
    {'configuration': ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B"],
     'metric': ['time', 'time', 'time', 'time', 'time', 'time', 'time','time', 'time', 'time', 'time', 'time', 'time', 'time', 'time', 'time','time', 'time', 'time', 'time', 'time', 'time', 'time', 'time', 'time','time', 'time', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'predictive', 'discriminative', 'discriminative', 'discriminative', 'discriminative', 'predictive', 'predictive', 'predictive', 'time', 'time', 'time'],
     'performance': [60.3, 56.2, 49.6, 33.9, 25.6, 18.2, 23.1, 16.5, 8.3, 57.1, 51.8, 43.8, 39.3, 33.0, 22.3, 0.9, 4.5, 0.0, 66.9, 59.2, 51.4, 44.4, 36.6, 29.6, 22.5, 23.9, 7.7, -304.3, -154.3, 10.9, -28.3, 4.3, 2.2, 47.8, 19.6, 67.4, -554.3, -137.1, -14.3, -57.1, -48.6, -14.3, -37.1, -54.3, 62.9, -453.3, -243.3, -30.0, -113.3, -36.7, 3.3, -10.0, 46.7, -13.3, 1.0, -11.5, -5.2, -9.4, -10.4, 0.0, 1.0, -2.1, 1.0, -38.9, -11.6, -3.2, -17.9, 1.1, 0.0, 1.1, 1.1, -5.3, -24.2, -15.2, -1.0, 2.0, -4.0, -1.0, 3.0, 4.0, 4.0, 24.1, -2.2, 12.4, -34.3, 0.7, 1.7, -2.4, 3.2, 10.4, -13.6]
    })

sns.set_theme(style="darkgrid")
dinges = sns.relplot(x="configuration", y="performance", kind="line", hue="metric", markers=True, data=df_stock, palette=["C0", "C1", "C2"])
dinges.set(xlabel='Configurations', ylabel='Performance increase/decrease (in %)')
dinges.savefig("my_plot-stock.png", dpi=400)

sns.set_theme(style="darkgrid")
dinges2 = sns.relplot(x="configuration", y="performance", kind="line", hue="metric", markers=True, data=df_sine, palette=["C2", "C0", "C1"])
dinges2.set(xlabel='Configurations', ylabel='Performance increase/decrease (in %)')
dinges2.savefig("my_plot-sine.png", dpi=400)
