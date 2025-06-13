import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import os
import warnings
import random
import subprocess

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)

# 设置字体
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

lines = []
colors = []
labels = []
mmsis = []

def read_csv_files(folder_path):
    global lines, colors, labels, mmsis
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder_path, filename))
            latitudes = df['lat'].tolist()
            longitudes = df['lon'].tolist()
            lines.append((longitudes, latitudes))
            colors.append((random.random(), random.random(), random.random()))

            if 'label' in df.columns and not df['label'].empty:
                labels.append(df['label'].iloc[0])
            else:
                labels.append([filename[:-4]])

            if 'mmsi' in df.columns and not df['mmsi'].empty:
                mmsis.append(df['mmsi'].iloc[0])
            else:
                mmsis.append([filename[:-4]])

def update(frame):
    ax.clear()
    ax.set_title('航迹动态绘制')
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    # ax.set_extent([110, 130, 18, 30], crs=ccrs.PlateCarree())
    # ax.set_extent([0, 180, 0, 90], crs=ccrs.PlateCarree())
    ax.set_extent([108, 124, 18, 28], crs=ccrs.PlateCarree())

    for (lon, lat), color, label, mmsi in zip(lines, colors, labels, mmsis):
        ax.plot(lon[:frame], lat[:frame], color=color, marker='.', alpha=0.7, linewidth=0.0000001, transform=ccrs.PlateCarree())
        # 把原本20进行修改
        a = random.randint(20, 30)
        if frame >= a:
            num_labels = min(frame - a, len(lon))
            if num_labels > 0:
                ax.annotate(label, xy=(lon[num_labels-1], lat[num_labels-1]),
                            xytext=(lon[num_labels-1] + 0.5, lat[num_labels-1] + 0.5),
                            arrowprops=dict(arrowstyle='->', color=color),
                            fontsize=8, color=color)


def load_data():
    global folder_path
    folder_path = filedialog.askdirectory()
    if folder_path:
        messagebox.showinfo("选择测试数据", "文件夹已选择，请进行模型测试")

def model_test():
    if not folder_path:
        messagebox.showwarning("警告", "请先选择文件夹！")
        return

    # 执行外部 Python 文件
    try:
        result = subprocess.run(['python', 'test_model.py'], capture_output=True, text=True)
        # output_text = result.stdout.strip()
        output_text = "模型测试完毕，请进行航迹绘制"
        messagebox.showinfo("模型测试进度", output_text)  # 显示结果
        read_csv_files(folder_path)  # 读取 CSV 数据
        max_length = max(len(lat) for _, lat in lines)
        ani = FuncAnimation(fig, update, frames=max_length, repeat=False, interval=0.01)
        canvas.draw()
    except Exception as e:
        messagebox.showerror("错误", f"执行模型测试时出错: {e}")


def resize_fig(event):
    fig.set_size_inches(event.width / 100, event.height / 100)  # 100是一个比例因子




# 创建Tkinter窗口
root = tk.Tk()
root.title("航迹分类演示")
button_load = tk.Button(root, text="选择测试数据文件夹", command=load_data)
button_load.pack()


button_test = tk.Button(root, text="模型测试", command=model_test)
button_test.pack()

# 创建Matplotlib图形与Cartopy轴
fig = plt.figure(figsize=(30, 14))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# # 在Tkinter窗口中添加事件绑定
# root.bind("<Configure>", resize_fig)


# 启动Tkinter主循环
root.mainloop()