"""
Nhóm 08
Nguyễn Đăng Khoa, msv: 22001603, Lớp: K67A4 Khoa học máy tính và thông tin
Hoàng Thiết Lâm, msv: 22001605, Lớp: K67A4 Khoa học máy tính và thông tin
Vũ Duy Linh, msv: 22001607, Lớp: K67A4 Khoa học máy tính và thông tin
"""
import numpy as np
import matplotlib.pyplot as plt


#Three-hump camel function. f(x_1,x_2) = 2 x_1^2 - 1.05 x_1^4 + (1/6) x_1^6 + x_1 x_2 + x_2^2
#Points A(1,1) and B(-2,0)

#Defining the three hump camel function
A = [1, 1]
B = [-2,0]
def func(x1, x2):
    return 2 * (x1**2) - 1.05 * (x1**4) + (1/6) * (x1**6) + x1 * x2 + x2**2
#Defining its gradient as a row vector
def gradient(x1, x2):
    return 4 * x1 - 4.2 * (x1 ** 3) + x1 ** 5 + x2, x1 + 2 * x2
#Defining its hessian matrix as a 2x2 matrix
def hessian(x1, x2):
    first_row = [4 - 12.6 * (x1 ** 2) + 5 * (x1**4), 1]
    second_row = [1, 2]
    return [first_row, second_row]
#Defining norm of vector
def norm(x1, x2):
    return x1**2 + x2**2
#Function to plots point traversed throughout the algorithm
def plot_points(list_point, algo):
    x1 = [point[0] for point in list_point]
    x2 = [point[1] for point in list_point]
    x = np.arange(-2, 2, 0.05)
    y = np.arange(-2, 2, 0.05)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)
    plt.contour(X, Y, Z, 100)
    plt.plot(x1, x2, 'o-')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('$f\,(x_1, x_2) \,= 2(x_1)^2 - 1.05(x_1)^2 + (1/6)(x_1)^6 + (x_1 x_2) + (x_2)^2$' + algo)
    plt.show()


def backtracking_gradient_descent(step_length, shrink_size, x, grad, t):
    # Backtracking line search algorithm to find tk for GRADIENT DESCENT algorithm
    while (func((x[0] - t * grad[0]), (x[1] - t * grad[1])) >
           func(x[0], x[1]) - step_length * t * norm(grad[0], grad[1])):
        t = t * shrink_size
    return t
    #Each iteration: t = t_init and while
        #f(x - t * ∇f(x)) > f(x) - α * t * ||∇f(x)||2-2
        #t = β * t
        #else, perform GD update
        #x+ = x - t * ∇f(x)
def gradient_descent(x0):
    shrink_size = 0.5 #β
    step_length = 0.5 #α
    t = 1 #start with t = 1
    count = 1
    xk = x0
    list_point = [xk]
    grad = gradient(xk[0], xk[1])
    while count <= 1000 and (norm(grad[0], grad[1])**(1/2)) > 1e-4:
        t = backtracking_gradient_descent(step_length,shrink_size, xk, grad, t)
        new_point = [0, 0]
        new_point[0] = xk[0] - t * grad[0]
        new_point[1] = xk[1] - t * grad[1]
        list_point.append(new_point)
        xk = new_point
        count = count + 1
        grad = gradient(xk[0], xk[1])
    return xk,count, list_point

grad_descent = gradient_descent(A)
result_point = grad_descent[0]
print(result_point)
iteration = grad_descent[1]
print(iteration)
print(func(result_point[0], result_point[1]))
plot_points(grad_descent[2], "GD")


def backtracking_newton(step_length, shrink_size, x, grad, hess, t):
    # Backtracking line search algorithm to find tk for NEWTON algorithm
    hess_matrix = np.array([[hess[0][0], hess[0][1]],[hess[1][0], hess[1][1]]])
    grad_array = np.array([grad[0], grad[1]])
    v = -np.dot(np.linalg.inv(hess_matrix), grad_array)
    while func(x[0] + t * v[0], x[1] + t * v[1]) > func(x[0], x[1]) + step_length * t * np.dot(grad_array, v):
        t = t * shrink_size
    return t
#Each iteration: t = t_init and while
    #f(x + t * v) > f(x) + α * t * ∇f(x)T * v
    #t = β * t
    #else, perform GD update
    #x+ = x - t * (∇^2f(x))^-1 * ∇f(x)
def newton_method(x0):
    shrink_size = 0.5  # β
    step_length = 0.5  # α
    t = 1 #step size t
    xk = x0
    grad = gradient(xk[0], xk[1])
    hess = hessian(xk[0], xk[1])
    count = 1
    result_list = [xk]
    while count <= 1000 and (norm(grad[0], grad[1])**(1/2)) > 1e-4:
        t = backtracking_newton(step_length, shrink_size, xk, grad, hess, t)
        #Solving equation: (∇^2f(x)) * (xk+1 - xk) = -tk * ∇f(xk)
        a = np.array([[hess[0][0], hess[0][1]],[hess[1][0], hess[1][1]]])
        b = np.array([grad[0] * (-t), grad[1] * (-t)])
        y = np.linalg.solve(a,b)
        new_point = [0,0]
        new_point[0] = xk[0] + y[0]
        new_point[1] = xk[1] + y[1]
        result_list.append(new_point)
        xk = new_point
        grad = gradient(xk[0], xk[1])
        hess = hessian(xk[0], xk[1])
        count = count + 1
    return xk, count, result_list

newton = newton_method(A)
result_point = newton[0]
iterations = newton[1]
result_point_list = newton[2]
print(result_point)
print(iterations)
print(func(result_point[0], result_point[1]))
plot_points(result_point_list, " NEWTON")




