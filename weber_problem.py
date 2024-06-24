import numpy as np
import matplotlib.pyplot as plt

# Hàm tính khoảng cách Euclidean giữa hai điểm
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Hàm thực hiện thuật toán Weiszfeld
def weiszfeld(points, weights, tol=1e-5, max_iter=1000):
    # Chuyển danh sách các điểm và trọng số thành mảng numpy
    points = np.array(points)
    weights = np.array(weights)

    # Khởi tạo điểm trung vị ban đầu bằng trung bình có trọng số của các điểm
    median = np.average(points, axis=0, weights=weights)

    # Bắt đầu vòng lặp
    for _ in range(max_iter):
        # Khởi tạo tử số và mẫu số
        numerator = np.zeros(2)  # Tử số khởi tạo là mảng [0, 0]
        denominator = 0  # Mẫu số khởi tạo là 0

        # Tính tử số và mẫu số cho mỗi điểm
        for i in range(len(points)):
            dist = euclidean_distance(median, points[i])  # Tính khoảng cách từ điểm trung vị đến điểm thứ i
            if dist != 0:
                weight = weights[i] / dist  # Tính trọng số điều chỉnh
                numerator += weight * points[i]  # Cập nhật tử số
                denominator += weight  # Cập nhật mẫu số

        # Tính vị trí mới dựa trên tử số và mẫu số
        new_median = numerator / denominator

        # Kiểm tra điều kiện dừng: nếu khoảng cách giữa trung vị mới và trung vị hiện tại nhỏ hơn ngưỡng tol, dừng lại
        if euclidean_distance(median, new_median) < tol:
            break

        # Cập nhật vị trí trung vị
        median = new_median

    return median

# Dữ liệu đầu vào
# Danh sách các điểm (tọa độ) và trọng số tương ứng
# points = [(2, 3), (5, 7), (1, 9), (3, 2), (6, 4), (7, 1), (8, 5), (4, 8), (9, 6), (5, 3)]
# weights = [10, 15, 8, 20, 12, 25, 10, 18, 14, 22]
points = [(0, 3), (5, 9), (4, 7), (6, 6), (6, 9), (8, 3), (3, 3), (4, 8)]
weights = [9, 15, 3, 23, 11, 9, 10, 7]

# Tìm điểm trung vị sử dụng thuật toán Weiszfeld
optimal_location = weiszfeld(points, weights)
print("Optimal Location:", optimal_location)

# Vẽ các điểm và vị trí tối ưu
points = np.array(points)  # Chuyển danh sách các điểm thành mảng numpy
plt.scatter(points[:, 0], points[:, 1], c='blue', label='Cửa hàng')  # Vẽ các điểm (cửa hàng)

# Thêm trọng số lên mỗi điểm
for i, txt in enumerate(weights):
    plt.annotate(txt, (points[i, 0], points[i, 1]), textcoords="offset points", xytext=(0, 5), ha='center')

# Vẽ vị trí tối ưu (nhà máy) bằng màu đỏ, với kích thước và kiểu marker đặc biệt
plt.scatter(optimal_location[0], optimal_location[1], c='red', marker='*', s=300, label='Nhà máy')
plt.xlabel('X')  # Nhãn trục X
plt.ylabel('Y')  # Nhãn trục Y
plt.title('Weber Problem - Single Facility Location')  # Tiêu đề biểu đồ
plt.legend()  # Hiển thị chú giải
plt.grid(True)  # Hiển thị lưới
plt.show()  # Hiển thị biểu đồ



#minh họa một cách tiếp cận cho bài toán Weber Problem. Thuật toán được sử dụng trong đoạn mã
# là phương pháp Weiszfeld, một thuật toán phổ biến được sử dụng để giải quyết bài toán Weber Problem - Single Facility FLPs
# Trong mã này, hàm weiszfeld được triển khai để tính toán vị trí tối ưu cho cơ sở dựa trên dữ liệu đầu vào (các điểm và trọng số tương ứng).
# Thuật toán sử dụng một phương pháp lặp để cập nhật vị trí ước lượng cho cơ sở dựa trên các trọng số và khoảng cách Euclidean.
# Sau khi chạy mã, optimal_location sẽ cho biết vị trí tối ưu của cơ sở, được tính dựa trên thuật toán Weiszfeld.
# Điểm này là giá trị gần đúng cho vị trí tối ưu của cơ sở dựa trên dữ liệu đầu vào được cung cấp.