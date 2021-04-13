# 매개변수 학습
def linear_regression(X, Y, num_of_data = 300, epochs = 1000, learning_rate = 0.0001):
    a_current = 0       # a of y = ax^3 + bx^2 + cx + d
    b_current = 0       # b of y = ax^3 + bx^2 + cx + d
    c_current = 0       # c of y = ax^3 + bx^2 + cx + d
    d_current = 0       # d of y = ax^3 + bx^2 + cx + d

    # 데이터 개수 300개
    N = num_of_data
    for cnt in range(epochs):
        cost = 0            # 책의 J 값(평균제곱오차), 최소화 하는 것이 목적
        a_gradient = 0      # a 편미분 값
        b_gradient = 0      # b 편미분 값
        c_gradient = 0      # c 편미분 값
        d_gradient = 0      # d 편미분 값
        y_current = []      # i번째 x일 때 예측값 y_current = ax^3 + bx^2 + cx + d
        for i in range(N):
            # i번째 x일 때 예측값 y_current = ax^3 + bx^2 + cx + d
            y_current.append((a_current * X[i] ** 3) + (b_current * X[i] ** 2) + (c_current * X[i]) + (d_current))

            a_gradient += (X[i] ** 3) * (Y[i] - y_current[i])   # a 편미분 1단계 시그마
            b_gradient += (X[i] ** 2) * (Y[i] - y_current[i])   # b 편미분 1단계 시그마
            c_gradient += X[i] * (Y[i] - y_current[i])          # c 편미분 1단계 시그마
            d_gradient += Y[i] - y_current[i]                   # d 편미분 1단계 시그마
            cost += (Y[i] - y_current[i]) ** 2  # 책의 J 값(평균제곱오차), 최소화 하는 것이 목적

        cost /= (2 * float(N))                         # 책의 J 값(평균제곱오차), 최소화 하는 것이 목적
        a_gradient = -(1 / float(N)) * a_gradient                      # a 편미분 2단계 N 나누기
        b_gradient = -(1 / float(N)) * b_gradient                      # b 편미분 2단계 N 나누기
        c_gradient = -(1 / float(N)) * c_gradient                      # c 편미분 2단계 N 나누기
        d_gradient = -(1 / float(N)) * d_gradient                      # d 편미분 2단계 N 나누기

        a_current = a_current - (learning_rate * a_gradient)    # a 값, 학습률에 따라 경사하강법
        b_current = b_current - (learning_rate * b_gradient)    # b 값, 학습률에 따라 경사하강법
        c_current = c_current - (learning_rate * c_gradient)    # c 값, 학습률에 따라 경사하강법
        d_current = d_current - (learning_rate * d_gradient)    # d 값, 학습률에 따라 경사하강법

        if (cnt+1) % 50 == 0:
            print(f'============================== Epoch {cnt+1} =============================================')
            print(f'y = {a_current:.4f}x^3 + {b_current:.4f}x^2 + {c_current:.4f}x + {d_current:.4f}')
            print(f'cost = {cost}')

            x_range = np.linspace(0, 3, 300)
            y_range = [a_current * n * n * n + b_current * n * n + c_current * n + d_current for n in x_range]
            plt.plot(x_range, y_range)

    return a_current, b_current, c_current, d_current, cost


# 랜덤한 데이터 생성
import random


def initializing(num_of_data = 300):
    x = []
    y = []

    # [0, 3] 사이의 랜덤한 실수 x, x 에 따른 y
    for i in range(num_of_data):
        temp_x = random.uniform(0, 3)
        temp_y = (temp_x * temp_x * temp_x) + (4.5 * temp_x * temp_x) \
                 + (6 * temp_x) + 2 \
                 + random.uniform(-0.5, 0.5)
        x.append(temp_x)
        y.append(temp_y)

    # x값 기준으로 오름차순 정렬
    temp = [[a, b] for a, b in zip(x, y)]
    temp.sort(key=lambda x:x[0])
    x = [a[0] for a in temp]
    y = [b[1] for b in temp]

    return x, y


######################################################### Main #########################################################
import matplotlib.pyplot as plt
import numpy as np

num_of_data = 300
x, y = initializing(num_of_data)

# 생성한 데이터 출력
for i in range(len(x)):
    print(f'{i+1}: {x[i]}, {y[i]}')

# 경사하강법 시행
a, b, c, d, cost = linear_regression(x, y, num_of_data=num_of_data, epochs=10000, learning_rate=0.0001)

# 생성한 데이터 그래프 출력
plt.plot(x, y, 'g.')
plt.plot(x, y)

# 근사함수 그래프 출력
x_range = np.linspace(0, 3, 300)
y_range = [a*n*n*n + b*n*n + c*n + d for n in x_range]
plt.plot(x_range, y_range)
plt.show()