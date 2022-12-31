import numpy as np
# y_preds는 예측값들이 담긴 데이터, y_test는 실제값 데이터들입니다.

y_preds = []
f = open("C:/Users/user/rmse_1.7.txt", 'r')
lines = f.readlines()
for line in lines:
    line = line.strip()
    line_int = float(line)
    y_preds.append(line_int)
f.close()

y_real = 1.7
for i in y_preds:
    rmse = np.sqrt(np.mean(((y_real - i)**2)))

print(rmse)
