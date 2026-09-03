import tkinter as tk
from tkinter import filedialog

from PIL import Image, ImageTk

from predict import recognize

img = None
file_path = None


# 处理图片
def choose_image(need_file=True):
    global img, file_path

    if need_file:
        file_path = filedialog.askopenfilename()
        if file_path:
            img = Image.open(file_path)
            img = img.convert("RGB")
            img = img.resize((180, 100))
    else:
        img = Image.new("RGB", (180, 100), (255, 255, 255))

    img.thumbnail((200, 200))

    _img = ImageTk.PhotoImage(img)
    image_label.config(image=_img)
    image_label.image = _img
    image_label.file_path = file_path


num = 0
true_num = 0


# 处理验证
def verify_code():
    global num, true_num
    content = ""
    accuracy = 0

    if img and file_path:
        content, true_content = recognize(img, file_path)
        num += 1
        if content == true_content:
            true_num += 1

        accuracy = int(round(true_num / num, 2) * 100) if num else 0

    accuracy_label.config(text=f"准确率: {accuracy}%", fg="black")

    content_label.config(text=f"验证码: {content}", fg="black")


# 创建主窗口
root = tk.Tk()
root.title("验证码识别系统")

title_label = tk.Label(
    root,
    text="验证码识别系统",
    bg="white",
    font=("Arial", 15),
    width=25,
    height=2,
    fg="black",
)
title_label.grid(row=0, column=0, padx=10, pady=10, columnspan=2)


# 选择图片按钮
choose_image_btn = tk.Button(root, text="选择图片", height=1, command=choose_image)
choose_image_btn.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)

# 所选图片的标签
image_label = tk.Label(root)
image_label.grid(row=1, column=1, padx=10, pady=10, rowspan=4)

# 默认展示白色图片
choose_image(False)

# 预测内容的标签
content_label = tk.Label(root, text="验证码: ", width=15, anchor=tk.NW, fg="black")
content_label.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)

# 预测准确率的标签
accuracy_label = tk.Label(root, text="准确率: ", width=15, anchor=tk.NW, fg="black")
accuracy_label.grid(row=3, column=0, padx=10, pady=10, sticky=tk.W)

# 验证识别按钮
verify_code_btn = tk.Button(root, text="识别", command=verify_code)
verify_code_btn.grid(row=4, column=0, padx=10, pady=10, sticky=tk.W)

# 运行主循环
root.mainloop()
