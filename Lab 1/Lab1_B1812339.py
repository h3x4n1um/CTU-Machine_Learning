import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    # 1. Đọc file dữ liệu” “baitap1.csv”
    dt = pd.read_csv("baitap1.csv", delimiter=',')

    # 2.Hiển thị dữ liệu vừa đọc
    print("CSV data:")
    print(dt)

    # 3.Hiển thị tất cả dữ liệu cột số 2
    print("\nCol 2 data:")
    print(dt.iloc[:, 1].to_frame())

    # 4.Hiển thị dữ liệu từ dòng 7 đến dòng 13
    print("\nRow 7 -> 13:")
    print(dt.iloc[7:14])

    # 5.Hiển thị dữ liệu cột 1, 2 của dòng 5
    print("\nCol 1, 2 of row 5:")
    print(dt.iloc[5, 0:2].to_frame())

    # 6.Tạo biến “x” là dữ liệu của cột 2, biến “y” là dữ liệu của cột 3, biễu diễn dữ liệu này lên mặt phẳng toạ độ với biến x = trục hoành, biến y = trục tung
    print("\nPlot x = Col 2, y = Col 3")
    x = dt.iloc[:, 1]
    y = dt.iloc[:, 2]
    plt.scatter(x, y)
    plt.title("Sự tương quan giữa tuổi và cân nặng")
    plt.xlabel("Tuổi")
    plt.ylabel("Cân nặng")
    plt.grid()
    plt.show()

    # 7.Sử dụng vòng lăp for để in ra các số chẳn từ 1 đến 100
    print("\nEven number from 1 to 100")
    for i in range(50):
        print((i+1)*2)