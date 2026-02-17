import tkinter as tk

class TopCenterDisplay:
    def __init__(self):
        self.root = tk.Tk()
        self.w, self.h = 360, 30
        self.root.title("Excel_tool")

        # 核心配置：置顶、无边框、背景透明
        self.root.attributes("-topmost", True)
        self.root.overrideredirect(True)
        self.root.config(bg="#000000")
        self.root.attributes("-transparentcolor", "#000000")  # 透明关键

        # 顶端居中定位
        screen_w = self.root.winfo_screenwidth()
        x = (screen_w - self.w) // 2
        self.root.geometry(f"{self.w}x{self.h}+{x}+0")

        # 4个文本标签（红色、透明背景、初始none）
        self.labels = []
        for i in range(5):
            lbl = tk.Label(
                self.root,
                text="none",
                font=("微软雅黑", 12, "bold"),
                fg="red",
                bg="#000000",  # 透明背景
                width=6, height=1
            )
            lbl.grid(row=0, column=i, padx=4, pady=2)
            self.labels.append(lbl)

    def withdraw(self):
        self.root.withdraw()

    def deiconify(self):
        self.root.deiconify()

    def destroy(self):
        # 延迟销毁窗口，确保线程完全退出
        self.root.after(100, self.root.destroy())

    # 单个修改核心方法
    def edit_single(self, idx, text):  # 把原texts改成text，命名更规范
        if not 0 <= idx < len(self.labels):
            raise IndexError(f"索引只能是0-3，你传了{idx}")
        self.labels[idx].config(text=str(text))


    def run(self):
        self.root.mainloop()

