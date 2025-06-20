""" Редактор изображений
 Программа для загрузки, отображения и обработки изображений с помощью PyTorch, OpenCV и Tkinter.
 Возможности:
 - Загрузка изображения с диска или камеры
 - Отображение цветового канала (R, G, B)
 - Повышение яркости изображения
 - Построение красного круга с заданными координатами и радиусом
 - Инверсия цветов (негатив)
 - Сброс всех изменений
 - Сохранение изображения
"""
import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageTk
import numpy as np
import cv2
from typing import Optional

MAX_W, MAX_H = 900, 550  # Максимальные размеры превью

"""
Преобразует тензор PyTorch в изображение PIL и масштабирует его для отображения в окне Tkinter.
"""
def tensor_to_tk(tensor: torch.Tensor) -> ImageTk.PhotoImage:
    pil = TF.to_pil_image(tensor)
    w, h = pil.size
    scale = min(MAX_W / w, MAX_H / h, 1)
    new_size = (int(w * scale), int(h * scale))
    return ImageTk.PhotoImage(pil.resize(new_size, Image.Resampling.LANCZOS))

"""
Основной класс интерфейса приложения.
"""
class ImageApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Редактор изображений")
        self.geometry("1200x750")

        self.original: Optional[torch.Tensor] = None  # Исходное изображение
        self.current: Optional[torch.Tensor] = None  # Обработанное изображение

        self.channel = tk.StringVar(value="all")  # Активный цветовой канал

        self._build_ui()  # Создание интерфейса

    """
    Строит графический интерфейс: кнопки, поля, область просмотра изображения.
    """
    def _build_ui(self):
        top = tk.LabelFrame(self, text="Выбор изображения")
        top.pack(fill=tk.X, padx=10, pady=5)
        tk.Button(top, text="Загрузить из файла", command=self.load_image).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(top, text="Сделать снимок", command=self.capture_from_cam).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(top, text="Сохранить", command=self.save_image).pack(side=tk.RIGHT, padx=5, pady=5)

        view_frame = tk.LabelFrame(self, text="Изображение")
        view_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.canvas = tk.Label(view_frame)
        self.canvas.pack(expand=True)

        bottom = tk.LabelFrame(self, text="Опции обработки")
        bottom.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(bottom, text="Цветовой канал:").pack(side=tk.LEFT)
        for txt, val in [("Красный", "r"), ("Зелёный", "g"), ("Синий", "b")]:
            tk.Radiobutton(bottom, text=txt, variable=self.channel, value=val, command=self.update_view).pack(side=tk.LEFT)
        tk.Button(bottom, text="Сброс", command=self.reset_all).pack(side=tk.LEFT, padx=(3, 15))

        tk.Label(bottom, text="Круг X,Y,R (в пикселях):").pack(side=tk.LEFT)
        self.circ_x = tk.Entry(bottom, width=4)
        self.circ_y = tk.Entry(bottom, width=4)
        self.circ_r = tk.Entry(bottom, width=4)
        self.circ_x.pack(side=tk.LEFT)
        self.circ_y.pack(side=tk.LEFT)
        self.circ_r.pack(side=tk.LEFT)
        tk.Button(bottom, text="Нарисовать", command=self.draw_circle).pack(side=tk.LEFT, padx=5)

        tk.Label(bottom, text="Диапазон яркости (от -100 до 100):").pack(side=tk.LEFT)
        self.bright_e = tk.Entry(bottom, width=5)
        self.bright_e.pack(side=tk.LEFT)
        tk.Button(bottom, text="Повысить яркость", command=self.brighten).pack(side=tk.LEFT, padx=5)

        tk.Button(bottom, text="Негатив", command=self.negative).pack(side=tk.RIGHT, padx=5)

    """
    Загружает изображение из файла и преобразует его в тензор PyTorch.
    """
    def load_image(self):
        try:
            path = filedialog.askopenfilename(filetypes=[("Изображения", "*.png *.jpg")])
            if not path:
                return
            img = Image.open(path).convert("RGB")
            self.original = TF.to_tensor(img)
            self.current = self.original.clone()
            self.update_view()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка загрузки: {e}")

    """
    Делает снимок с веб-камеры и сохраняет его в виде тензора.
    """
    def capture_from_cam(self):
        try:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise RuntimeError("Камера недоступна")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            self.original = TF.to_tensor(pil)
            self.current = self.original.clone()
            self.update_view()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка камеры: {e}")

    """
    Обновляет отображение изображения с учётом выбранного цветового канала.
    """
    def update_view(self):
        if self.current is None:
            return
        tensor = self.current.clone()
        if self.channel.get() in ["r", "g", "b"]:
            mask = torch.zeros_like(tensor)
            idx = {"r": 0, "g": 1, "b": 2}[self.channel.get()]
            mask[idx] = tensor[idx]
            tensor = mask
        img = tensor_to_tk(tensor)
        self.canvas.configure(image=img)
        self.canvas.image = img

    """
    Сбрасывает все изменения: яркость, цветовой канал, круг, негатив.
    """
    def reset_all(self):
        if self.original is None:
            return
        self.current = self.original.clone()
        self.channel.set("all")
        self.circ_x.delete(0, tk.END)
        self.circ_y.delete(0, tk.END)
        self.circ_r.delete(0, tk.END)
        self.bright_e.delete(0, tk.END)
        self.update_view()

    """
    Рисует красный круг с заданными координатами X, Y и радиусом R.
    """

    def draw_circle(self):
        try:
            if self.current is None:
                messagebox.showerror("Ошибка", "Сначала загрузите изображение")
                return

            # Получаем размеры изображения
            img_height, img_width = self.current.shape[1], self.current.shape[2]

            # Получаем параметры круга с проверкой на пустые поля
            if not self.circ_x.get() or not self.circ_y.get() or not self.circ_r.get():
                raise ValueError("Все поля (X, Y, R) должны быть заполнены")

            x = int(self.circ_x.get())
            y = int(self.circ_y.get())
            r = int(self.circ_r.get())

            # Проверяем допустимость значений
            if r <= 0:
                raise ValueError(f"Радиус должен быть положительным числом (вы ввели: {r})")

            if x < 0 or x >= img_width:
                raise ValueError(f"Координата X должна быть от 0 до {img_width - 1} (вы ввели: {x})")

            if y < 0 or y >= img_height:
                raise ValueError(f"Координата Y должна быть от 0 до {img_height - 1} (вы ввели: {y})")

            if x - r < 0 or x + r >= img_width or y - r < 0 or y + r >= img_height:
                max_r = min(x, img_width - 1 - x, y, img_height - 1 - y)
                raise ValueError(f"Круг не помещается в изображение.\n"
                                 f"Максимальный радиус для X={x}, Y={y}: {max_r}\n"
                                 f"Вы ввели радиус: {r}")

            # Преобразуем тензор в изображение OpenCV
            img_np = self.current.mul(255).byte().cpu().numpy()
            img_np = np.transpose(img_np, (1, 2, 0))

            # Проверяем формат изображения
            if img_np.shape[2] != 3:
                raise ValueError("Изображение должно быть цветным (3 канала RGB)")

            # Рисуем круг
            img_with_circle = cv2.circle(img_np.copy(), (x, y), r, (255, 0, 0), 10)

            # Обратно в тензор
            self.current = TF.to_tensor(Image.fromarray(img_with_circle))
            self.update_view()

        except ValueError as e:
            # Формируем подробное сообщение об ошибке с подсказками
            if self.current is not None:
                img_height, img_width = self.current.shape[1], self.current.shape[2]
                error_msg = (
                    f"{str(e)}\n\n"
                    f"Допустимые значения:\n"
                    f"- Координата X: от 0 до {img_width - 1}\n"
                    f"- Координата Y: от 0 до {img_height - 1}\n"
                    f"- Радиус R: положительное число, чтобы круг помещался в изображение\n\n"
                    f"Текущий размер изображения: {img_width}x{img_height} пикселей"
                )
            else:
                error_msg = str(e)

            messagebox.showerror("Ошибка ввода параметров", error_msg)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось нарисовать круг: {str(e)}")

    """
    Повышает или понижает яркость изображения на указанное значение.
    """
    def brighten(self):
        if self.current is None:
            return
        try:
            delta = float(self.bright_e.get())
            if not -100 <= delta <= 100:
                raise ValueError("Диапазон: от -100 до 100")
            self.current = torch.clamp(self.current + delta / 255.0, 0.0, 1.0)
            self.update_view()
        except ValueError as e:
            messagebox.showerror("Ошибка", str(e))
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось изменить яркость: {e}")

    """
    Применяет эффект негатива к изображению.
    """
    def negative(self):
        if self.current is not None:
            self.current = 1.0 - self.current
            self.update_view()

    """
    Сохраняет текущее изображение в выбранный пользователем файл.
    """
    def save_image(self):
        if self.current is None:
            messagebox.showwarning("Нет изображения", "Сначала загрузите и обработайте изображение")
            return
        try:
            pil = TF.to_pil_image(self.current)
            path = filedialog.asksaveasfilename(defaultextension=".png",
                                               filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")])
            if path:
                pil.save(path)
                messagebox.showinfo("Успешно", f"Изображение сохранено в: {path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить изображение: {e}")

if __name__ == '__main__':
    app = ImageApp()
    app.mainloop()
