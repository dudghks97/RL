import random
import matplotlib.pyplot as plt
import numpy as np


# 매개변수 학습
# y = ax^3 + bx^2 + cx + d
# 데이터 개수 300개, 에포크 1000번, learning rate 는 학습률(스텝사이즈)
def linear_regression(X, Y, num_of_data = 300, epochs = 1000, learning_rate = 0.0001):
    a_current = 0       # 매개변수 a 의 현재값
    b_current = 0       # 매개변수 b 의 현재값
    c_current = 0       # 매개변수 c 의 현재값
    d_current = 0       # 매개변수 d 의 현재값

    # 데이터 개수 300개
    N = num_of_data
    for cnt in range(epochs):
        cost = 0            # 비용함수(cost function) J 값(평균제곱오차), 최소화 하는 것이 목적
        a_gradient = 0      # 매개변수 a 편미분 값
        b_gradient = 0      # 매개변수 b 편미분 값
        c_gradient = 0      # 매개변수 c 편미분 값
        d_gradient = 0      # 매개변수 d 편미분 값
        y_current = []      # i번째 x일 때 예측값 y_current = ax^3 + bx^2 + cx + d
        for i in range(N):
            # i번째 x일 때 예측값 y_current = ax^3 + bx^2 + cx + d
            y_current.append((a_current * (X[i] ** 3)) + (b_current * (X[i] ** 2)) + (c_current * X[i]) + d_current)

            a_gradient += (X[i] ** 3) * (Y[i] - y_current[i])   # a 편미분 1단계
            b_gradient += (X[i] ** 2) * (Y[i] - y_current[i])   # b 편미분 1단계
            c_gradient += X[i] * (Y[i] - y_current[i])          # c 편미분 1단계
            d_gradient += Y[i] - y_current[i]                   # d 편미분 1단계
            cost += (Y[i] - y_current[i]) ** 2  # 비용함수(cost function) J 값(평균제곱오차), 최소화 하는 것이 목적

        cost /= (2 * float(N))                         # 비용함수(cost function) J 값(평균제곱오차), 최소화 하는 것이 목적
        a_gradient = -(1 / float(N)) * a_gradient                      # a 편미분 2단계
        b_gradient = -(1 / float(N)) * b_gradient                      # b 편미분 2단계
        c_gradient = -(1 / float(N)) * c_gradient                      # c 편미분 2단계
        d_gradient = -(1 / float(N)) * d_gradient                      # d 편미분 2단계

        a_current = a_current - (learning_rate * a_gradient)    # a 값, 학습률에 따라 경사하강법
        b_current = b_current - (learning_rate * b_gradient)    # b 값, 학습률에 따라 경사하강법
        c_current = c_current - (learning_rate * c_gradient)    # c 값, 학습률에 따라 경사하강법
        d_current = d_current - (learning_rate * d_gradient)    # d 값, 학습률에 따라 경사하강법


        # 에포크 10 마다 근사함수 값 출력
        if (cnt+1) % 10 == 0:
            print(f'================= Epoch : {cnt+1} =================')
            print(f'y = {a_current:.4f}x^3 + {b_current:.4f}x^2 + {c_current:.4f}x + {d_current:.4f}')
            print(f'cost = {cost}')

    return a_current, b_current, c_current, d_current, cost


# 문제의 조건에 맞는 랜덤한 데이터 생성
def initializing(num_of_data = 300):
    X = []
    Y = []

    for i in range(num_of_data):
        # x 는 [0, 3] 구간의 랜덤한 실수
        x = random.uniform(0, 3)
        # y = x^3 - 4.5x^2 + 6x + 2 의 함수에 랜덤한 실수 x 값 대입 후
        # [-0.5, 0.5] 사이의 값을 랜덤하게 생성하여 더함
        y = x**3 - (4.5 * x**2) + (6 * x) + 2 + random.uniform(-0.5, 0.5)
        X.append(x)
        Y.append(y)

    # x 값 기준으로 오름차순 정렬
    temp = [[a, b] for a, b in zip(X, Y)]
    temp.sort(key=lambda X:X[0])

    X = [a[0] for a in temp]
    Y = [b[0] for b in temp]

    return X, Y


# main 부분

num_of_data = 300
X, Y = initializing(num_of_data)

# 생성한 데이터 출력 (실제값)
for i in range(len(X)):
    print(f'{i+1}: {X[i]}, {Y[i]}')
print("============================== End of Initializing ==============================")
print()
print()

# 경사하강법 시행
a, b, c, d, cost = linear_regression(X, Y, num_of_data=num_of_data, epochs=10000, learning_rate=0.0001)

# 생성한 데이터 그래프 출력
plt.subplot(1, 3, 1)
plt.title("Data")
plt.plot(X, Y, 'r.')

# 근사함수 그래프 출력
plt.subplot(1, 3, 2)
plt.title("Graph of Function Approximation")
# x_range = np.linspace(0, 3, 300)
x_range = np.array(X)
y_range = np.array([a*(x**3) + b*(x**2) + c*x + d for x in x_range])
plt.plot(x_range, y_range, 'b')

# 생성한 데이터 그래프와 근사함수 그래프 동시 출력
plt.subplot(1, 3, 3)
plt.title("Graph of Data & Function Approximation")
plt.plot(X, Y, 'r.', label='Data')
plt.plot(x_range, y_range, 'b', label='Function Approximation')
plt.legend()

plt.show()
