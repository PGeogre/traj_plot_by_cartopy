import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from mpl_toolkits.basemap import Basemap
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
m = None  # Basemap实例

def read_csv_files(folder_path):
    global lines, colors, labels, mmsis
    lines.clear()
    colors.clear()
    labels.clear()
    mmsis.clear()
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder_path, filename))
            latitudes = df['lat'].tolist()
            longitudes = df['lon'].tolist()
            lines.append((longitudes, latitudes))
            colors.append((random.random(), random.random(), random.random()))

            label = filename[:-4]  # 默认使用文件名
            if 'label' in df.columns and not df['label'].empty:
                label = df['label'].iloc[0]
            labels.append(label)

            mmsi = filename[:-4]  # 默认使用文件名
            if 'mmsi' in df.columns and not df['mmsi'].empty:
                mmsi = df['mmsi'].iloc[0]
            mmsis.append(mmsi)

def update(frame):
    global m
    ax.clear()
    
    # 重新创建Basemap实例
    m = Basemap(projection='merc',
                llcrnrlon=108,
                llcrnrlat=18,
                urcrnrlon=124,
                urcrnrlat=28,
                resolution='i',

                ax=ax)
    
    m.drawcoastlines()
    m.drawcountries()
    # m.fillcontinents(color='#E0E0E0', lake_color='#F5F5F5')  # 更柔和的配色
    m.fillcontinents(color='lightgray', lake_color='lightblue')
    # m.drawmapboundary(fill_color='#F5F5F5')
    m.drawmapboundary(fill_color='lightblue')
    
    ax.set_title('航迹动态绘制', fontsize=14, pad=20)

    for (lon, lat), color, label, mmsi in zip(lines, colors, labels, mmsis):
        if frame >= len(lon):  # 防止索引越界
            continue
            
        x, y = m(lon[:frame], lat[:frame])
        line = ax.plot(x, y, color=color, marker='.', 
                      markersize=4, alpha=0.8, linewidth=0.5)
        
        # 动态标注优化
        # if frame >= 20 and frame % 5 == 0:  # 每10帧更新一次标注
        # a = random.randint(10, 20) # 设置随机数，决定何时显示标注
        a = random.randint(0, 10) # 设置随机数，决定何时显示标注
        if frame >= a:  
            last_idx = min(frame, len(lon)-1)
            x_annot, y_annot = m(lon[last_idx], lat[last_idx])
            ax.annotate(f"{label}\nMMSI: {mmsi}",
                        xy=(x_annot, y_annot),
                        xytext=(10, 10),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color=color, lw=0.8),
                        fontsize=8,
                        color=color,
                        bbox=dict(boxstyle='round,pad=0.3', 
                                 fc='white', 
                                 ec=color,
                                 lw=0.5,
                                 alpha=0.8))

def load_data():
    global folder_path
    folder_path = filedialog.askdirectory()
    if folder_path:
        messagebox.showinfo("选择测试数据", f"已选择文件夹：\n{folder_path}")

def model_test():
    if not folder_path:
        messagebox.showwarning("警告", "请先选择文件夹！")
        return

    try:
        # 显示进度提示
        progress_label.config(text="正在进行模型测试...")
        root.update()
        
        result = subprocess.run(['python', 'test_model.py'], capture_output=True, text=True)
        output_text = "模型测试完毕，请进行航迹绘制"
        messagebox.showinfo("模型测试进度", output_text)
        
        # 读取数据并初始化
        progress_label.config(text="正在加载数据...")
        root.update()
        read_csv_files(folder_path)
        
        # 初始化动画
        progress_label.config(text="正在准备可视化...")
        root.update()
        max_length = max(len(lat) for _, lat in lines)
        ani = FuncAnimation(fig, update, frames=max_length, repeat=False, interval=0.01)
        
        canvas.draw()
        progress_label.config(text="就绪")
    except Exception as e:
        messagebox.showerror("错误", f"执行出错: {str(e)}")
        progress_label.config(text="发生错误")

# 创建主窗口
root = tk.Tk()
root.title("航迹分类演示系统")
root.geometry("1200x800")

# 创建顶部控制栏
control_frame = tk.Frame(root, bg='#F0F0F0', padx=10, pady=5)
control_frame.pack(side='top', fill='x')

# 按钮组
button_load = tk.Button(control_frame, 
                       text="选择测试数据文件夹", 
                       command=load_data,
                       width=18,
                       bg='#4CAF50',
                       fg='white')
button_load.pack(side='left', padx=5)

button_test = tk.Button(control_frame,
                       text="开始模型测试", 
                       command=model_test,
                       width=15,
                       bg='#2196F3',
                       fg='white')
button_test.pack(side='left', padx=5)

# 进度标签
progress_label = tk.Label(control_frame,
                         text="就绪",
                         fg='#666666',
                         bg='#F0F0F0',
                         anchor='w')
progress_label.pack(side='left', padx=20)

# 创建Matplotlib图形
fig = plt.figure(figsize=(12, 8), dpi=100)
ax = fig.add_subplot(1, 1, 1)

# 初始化Basemap
m = Basemap(projection='merc',
            llcrnrlon=108,
            llcrnrlat=18,
            urcrnrlon=124,
            urcrnrlat=28,
            resolution='i',
            ax=ax)

# 创建画布
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

# 底部状态栏
status_bar = tk.Label(root, 
                     text="航迹可视化系统 v1.0 | 作者: XXX", 
                     bd=1, 
                     relief='sunken',
                     anchor='w',
                     bg='#E0E0E0')
status_bar.pack(side='bottom', fill='x')

# 启动主循环
root.mainloop()