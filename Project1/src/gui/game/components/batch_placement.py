import tkinter as tk
from tkinter import messagebox, Toplevel, ttk
from typing import Callable, List, Tuple

from core.board import Board


class _BatchPlacementWindow(Toplevel):
    """一个用于批量添加/删除特定种类方块位置的弹窗"""

    def __init__(
        self,
        parent,
        current_board: Board,
        current_coords: List[Tuple[int, int]],
        save_callback: Callable,
    ):
        super().__init__(parent)
        self.title("批量操作坐标")
        self.geometry("300x400")
        self.transient(parent)
        self.grab_set()

        self.current_board: Board = current_board
        self.current_coords: List[Tuple[int, int]] = current_coords
        self.save_callback = save_callback
        self.board_size = current_board.size
        self.num_of_empty = current_board.get_num_of_empty() - len(current_coords)
        self.rows = []
        self.row_widgets = []

        # --- 可滚动的框架 ---
        canvas_frame = tk.Frame(self)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor="nw"
        )
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # Control buttons
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Button(btn_frame, text="[+] 添加行", command=self._add_row).pack(
            side=tk.LEFT
        )
        tk.Button(btn_frame, text="取消", command=self.destroy).pack(
            side=tk.RIGHT, padx=5
        )
        tk.Button(btn_frame, text="保存", command=self._save).pack(side=tk.RIGHT)

        self._populate_initial_rows(current_coords)
        self._bind_scroll_events()
        self._center_window(parent)

    def _populate_initial_rows(self, coords):
        for r, c in coords:
            self.rows.append({"x": str(r), "y": str(c)})
        if len(self.rows) < 2:
            self.rows.extend([{"x": "", "y": ""} for _ in range(2 - len(self.rows))])
        self._redraw_rows()

    def _redraw_rows(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.row_widgets = []

        for i, row_data in enumerate(self.rows):
            row_frame = tk.Frame(self.scrollable_frame)
            row_frame.pack(fill=tk.X, pady=2)

            tk.Label(row_frame, text=f"方块 {i+1}:  X:").pack(side=tk.LEFT)
            x_var = tk.StringVar(value=row_data["x"])
            ttk.Entry(row_frame, textvariable=x_var, width=4).pack(side=tk.LEFT)

            tk.Label(row_frame, text=" Y:").pack(side=tk.LEFT)
            y_var = tk.StringVar(value=row_data["y"])
            ttk.Entry(row_frame, textvariable=y_var, width=4).pack(side=tk.LEFT)

            tk.Button(
                row_frame, text="[-]", command=lambda idx=i: self._remove_row(idx)
            ).pack(side=tk.RIGHT, padx=5)
            row_data["x_var"], row_data["y_var"] = (
                x_var,
                y_var,
            )

            self.row_widgets.append({"x_var": x_var, "y_var": y_var})

    def _sync_data_from_ui(self):
        """将UI输入框中的当前值同步到 self.rows 中"""
        for i, widget_vars in enumerate(self.row_widgets):
            self.rows[i]["x"] = widget_vars["x_var"].get()
            self.rows[i]["y"] = widget_vars["y_var"].get()

    def _add_row(self):
        num_of_existing = len(self.rows)
        if num_of_existing >= self.num_of_empty:
            messagebox.showerror(
                "操作无效", f"棋盘剩余空位最多放置 {self.num_of_empty} 个方块"
            )
            return

        self._sync_data_from_ui()
        num_to_add = min(
            2 if num_of_existing % 2 == 0 else 1, self.num_of_empty - num_of_existing
        )
        for _ in range(num_to_add):
            self.rows.append({"x": "", "y": ""})
        self._redraw_rows()

    def _remove_row(self, index):
        self._sync_data_from_ui()
        self.rows.pop(index)
        self._redraw_rows()

    def _save(self):
        validated_coords = []
        self._sync_data_from_ui()
        try:
            for i, row_data in enumerate(self.rows):
                x_str, y_str = row_data["x"], row_data["y"]
                if not x_str and not y_str:
                    continue
                if not x_str or not y_str:
                    raise IndexError(f"第 {i+1} 个方块坐标不完整")

                r, c = int(x_str), int(y_str)
                pos = (r, c)
                if not (0 <= r < self.board_size and 0 <= c < self.board_size):
                    raise ValueError(
                        f"第 {i+1} 行坐标 {pos} 超出棋盘边界 0-{self.board_size - 1}"
                    )

                if self.current_board.get_tile_at(pos):
                    raise IndexError(f"第 {i+1} 行坐标 {pos} 已被其他种类方块占据")
                if pos in validated_coords:
                    raise IndexError(f"第 {i+1} 行坐标 {pos} 在列表中重复")

                validated_coords.append(pos)
        except ValueError as e:
            messagebox.showerror("输入错误", f"{e}。请输入有效的正整数")
            return False
        except IndexError as e:
            messagebox.showerror("输入错误", str(e))
            return False

        self.save_callback(validated_coords)
        self.destroy()

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

    def _bind_scroll_events(self):
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
