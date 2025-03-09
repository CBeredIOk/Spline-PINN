import tkinter as tk
import subprocess
from tkinter import ttk
from tkinter import messagebox


class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Spline-PINN GUI")
        self.geometry("300x150")

        label = ttk.Label(self, text="Выберите действие:")
        label.pack(pady=10)

        self.predict_button = ttk.Button(
            self, text="Предсказание (визуализация)", command=self.open_predict_window
        )
        self.predict_button.pack(pady=5)

        self.train_button = ttk.Button(
            self, text="Обучение модели", command=self.open_train_window
        )
        self.train_button.pack(pady=5)

    def open_predict_window(self):
        PredictWindow(self)

    def open_train_window(self):
        TrainWindow(self)


class PredictWindow(tk.Toplevel):
    """
    Окно со вводом параметров для режима предсказания (fluid_test.py)
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Параметры для предсказания")
        self.geometry("400x300")

        # Параметры для fluid_test.py
        # При желании можно добавить больше входных полей

        ttk.Label(self, text="Вязкость (mu):").grid(row=0, column=0, pady=5, padx=5, sticky="e")
        self.mu_entry = ttk.Entry(self)
        self.mu_entry.insert(0, "0.1")
        self.mu_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(self, text="Плотность (rho):").grid(row=1, column=0, pady=5, padx=5, sticky="e")
        self.rho_entry = ttk.Entry(self)
        self.rho_entry.insert(0, "10")
        self.rho_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(self, text="Hidden size:").grid(row=2, column=0, pady=5, padx=5, sticky="e")
        self.hidden_size_entry = ttk.Entry(self)
        self.hidden_size_entry.insert(0, "50")
        self.hidden_size_entry.grid(row=2, column=1, padx=5, pady=5)

        run_button = ttk.Button(self, text="Запустить визуализацию", command=self.run_prediction)
        run_button.grid(row=3, column=0, columnspan=2, pady=20)

    def run_prediction(self):
        """
        Собираем параметры и вызываем fluid_test.py в режиме предсказания
        """
        mu_value = self.mu_entry.get()
        rho_value = self.rho_entry.get()
        hidden_size_value = self.hidden_size_entry.get()

        try:
            # Пример вызова через командную строку
            subprocess.run([
                "python", "fluid_test.py",
                "--net=Fluid_model",
                f"--hidden_size={hidden_size_value}",
                f"--mu={mu_value}",
                f"--rho={rho_value}"
            ], check=True)

            messagebox.showinfo(
                "Предсказание",
                f"Скрипт fluid_test.py успешно завершен.\n\n"
                f"Параметры:\nmu={mu_value}, rho={rho_value}, hidden_size={hidden_size_value}"
            )
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Ошибка", f"Ошибка при запуске fluid_test.py:\n{e}")

        # Закрываем окно (по желанию можно оставить открытым)
        self.destroy()


class TrainWindow(tk.Toplevel):
    """
    Окно со вводом параметров для обучения (fluid_train.py)
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Параметры для обучения")
        self.geometry("400x400")

        ttk.Label(self, text="Вязкость (mu):").grid(row=0, column=0, pady=5, padx=5, sticky="e")
        self.mu_entry = ttk.Entry(self)
        self.mu_entry.insert(0, "0.5")
        self.mu_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(self, text="Плотность (rho):").grid(row=1, column=0, pady=5, padx=5, sticky="e")
        self.rho_entry = ttk.Entry(self)
        self.rho_entry.insert(0, "4")
        self.rho_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(self, text="Loss Bound (loss_bound):").grid(row=2, column=0, pady=5, padx=5, sticky="e")
        self.bound_entry = ttk.Entry(self)
        self.bound_entry.insert(0, "20")
        self.bound_entry.grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(self, text="Loss Domain Res (loss_domain_res):").grid(row=3, column=0, pady=5, padx=5, sticky="e")
        self.domainres_entry = ttk.Entry(self)
        self.domainres_entry.insert(0, "10")
        self.domainres_entry.grid(row=3, column=1, padx=5, pady=5)

        ttk.Label(self, text="Hidden size:").grid(row=4, column=0, pady=5, padx=5, sticky="e")
        self.hidden_size_entry = ttk.Entry(self)
        self.hidden_size_entry.insert(0, "50")
        self.hidden_size_entry.grid(row=4, column=1, padx=5, pady=5)

        ttk.Label(self, text="Learning rate (lr):").grid(row=5, column=0, pady=5, padx=5, sticky="e")
        self.lr_entry = ttk.Entry(self)
        self.lr_entry.insert(0, "0.0001")
        self.lr_entry.grid(row=5, column=1, padx=5, pady=5)

        run_button = ttk.Button(self, text="Запустить обучение", command=self.run_training)
        run_button.grid(row=6, column=0, columnspan=2, pady=20)

    def run_training(self):
        """
        Собираем параметры и вызываем fluid_train.py в режиме обучения
        """
        mu_value = self.mu_entry.get()
        rho_value = self.rho_entry.get()
        bound_value = self.bound_entry.get()
        domainres_value = self.domainres_entry.get()
        hidden_size_value = self.hidden_size_entry.get()
        lr_value = self.lr_entry.get()

        try:
            subprocess.run([
                "python", "fluid_train.py",
                "--net=Fluid_model",
                f"--hidden_size={hidden_size_value}",
                f"--mu={mu_value}",
                f"--rho={rho_value}",
                f"--loss_bound={bound_value}",
                f"--loss_domain_res={domainres_value}",
                f"--lr={lr_value}"
            ], check=True)

            messagebox.showinfo(
                "Обучение",
                f"Скрипт fluid_train.py успешно запущен.\n\n"
                f"Параметры:\nmu={mu_value}, rho={rho_value}, "
                f"loss_bound={bound_value}, loss_domain_res={domainres_value}, "
                f"hidden_size={hidden_size_value}, lr={lr_value}"
            )
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Ошибка", f"Ошибка при запуске fluid_train.py:\n{e}")

        # Закрываем окно
        self.destroy()


def main():
    app = MainApplication()
    app.mainloop()


if __name__ == "__main__":
    main()
