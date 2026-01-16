import tkinter as tk
from tkinter import messagebox, Toplevel, ttk
from typing import Dict, Callable

from utils.asset_manager import AssetManager
import utils.constants as const

class _CostEditorWindow(Toplevel):
    """一个用于查看和编辑方块代价的弹窗类"""

    def __init__(
        self,
        parent,
        asset_manager: AssetManager,
        current_costs: Dict[int, int],
        num_types: int,
        save_callback: Callable,
    ):
        super().__init__(parent)
        self.title("查看/编辑代价")
        self.geometry("350x450")
        self.transient(parent)  # 保持在父窗口之上
        self.grab_set()  # 模态窗口，阻止与其他窗口交互

        self.asset_manager = asset_manager
        self.save_callback = save_callback
        self.cost_entries: Dict[int, tk.StringVar] = {}

        # --- 可滚动的框架 ---
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(
            main_frame, orient="vertical", command=self.canvas.yview
        )
        scrollable_frame = ttk.Frame(self.canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=scrollable_frame, anchor="nw"
        )
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # --- 填充内容 ---
        i = 0
        for type_id, cost in sorted(current_costs.items()):
            i += 1
            if i > num_types:
                break

            item_frame = tk.Frame(scrollable_frame, pady=5)
            item_frame.pack(fill=tk.X)

            visual = self.asset_manager.get_visual_for_type(type_id)
            if visual["type"] == "image" and visual.get("data"):
                tk.Label(item_frame, image=visual["data"]).pack(side=tk.LEFT, padx=5)
            else:
                tk.Label(
                    item_frame,
                    text=visual.get("number", str(type_id)),
                    bg=visual.get("color", "white"),
                    width=4,
                    height=2,
                ).pack(side=tk.LEFT, padx=5)

            tk.Label(item_frame, text=f"种类 {type_id} 代价:").pack(
                side=tk.LEFT, padx=5
            )
            var = tk.StringVar(value=str(cost))
            ttk.Entry(item_frame, textvariable=var, width=5).pack(side=tk.RIGHT, padx=5)
            self.cost_entries[type_id] = var

        # --- 按钮 ---
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        tk.Button(
            btn_frame, text="保存", font=const.FONT_BUTTON, command=self._save_costs
        ).pack(side=tk.LEFT, expand=True, padx=5)
        tk.Button(
            btn_frame, text="取消", font=const.FONT_BUTTON, command=self.destroy
        ).pack(side=tk.RIGHT, expand=True, padx=5)

        self._bind_scroll_events(scrollable_frame)
        self._center_window(parent)

    def _save_costs(self):
        new_costs = {}
        try:
            for type_id, var in self.cost_entries.items():
                cost = int(var.get())
                if cost <= 0:
                    raise IndexError("代价不能为零或负数")
                new_costs[type_id] = cost
            self.save_callback(new_costs)
            self.destroy()
        except ValueError as e:
            messagebox.showerror("输入错误", f"{e}。请输入有效的正整数")
        except IndexError as e:
            messagebox.showerror("输入错误", str(e))

    def _center_window(self, parent):
        # 窗口居中
        self.update_idletasks()
        # 获取父窗口几何信息
        parent_x = parent.winfo_rootx()
        parent_y = parent.winfo_rooty()
        parent_width = parent.winfo_width()
        parent_height = parent.winfo_height()
        # 获取此弹窗自身的尺寸
        window_width = self.winfo_width()
        window_height = self.winfo_height()
        # 计算居中坐标
        pos_x = parent_x + (parent_width // 2) - (window_width // 2)
        pos_y = parent_y + (parent_height // 2) - (window_height // 2)
        # 设置新位置
        self.geometry(f"+{pos_x}+{pos_y}")

    def _bind_scroll_events(self, widget_to_bind):
        """为指定的控件及其子控件绑定鼠标滚轮事件"""
        # --- 绑定鼠标滚轮 ---
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)
        # 当Canvas大小改变时，确保内部Frame宽度跟随改变
        self.canvas.bind("<Configure>", self._on_canvas_configure)

    def _on_mousewheel(self, event):
        """处理鼠标滚轮事件"""
        if event.num == 4:
            delta = -1
        elif event.num == 5:
            delta = 1
        else:
            delta = -1 * int(event.delta / 120)

        self.canvas.yview_scroll(delta, "units")

    def _on_canvas_configure(self, event):
        """当Canvas大小改变时，调整内部Frame的宽度"""
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def destroy(self):
        """重写destroy方法以解绑全局事件"""
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")
        super().destroy()
