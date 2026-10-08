# Сравнение исходного и рабочего блокнотов

Исходный файл: `opt_hw _ start.ipynb`. Рабочий файл: `opt_hw.ipynb`.
Исходник не менялся; его SHA-256 до и после проверки: `3b7ccf64007ffea150691f4f957984ea5ff08c96bc8258384fd4bea90a55ffc1`.
Рабочий файл после выполнения: SHA-256 `d34032877d4eea581e39fbb22a9b20087f819cd3cc6df9737f16e97faa5919e6`.
Сравниваются все ячейки по их исходному тексту; индексы ячеек начинаются с 0.
Блоки diff включают изменения и контекст, Markdown и код не пропущены.

## Проверка выполнения и структуры

- Ячеек в исходнике / рабочем файле: 104 / 104.
- Ячеек добавлено или удалено: 0 / 0.
- Выполнено кодовых ячеек: 31. PNG-выводов сохранено: 14. Ошибок: 0.
- ID ячеек и их порядок сохранены.

## Сводка по измененным ячейкам

| Индекс | Тип | ID | PNG до -> после | Ошибки после |
| ---: | --- | --- | ---: | ---: |
| 2 | code | `35ab9136` | 0 -> 0 | 0 |
| 16 | code | `a685b55f` | 1 -> 1 | 0 |
| 43 | markdown | `3NXUSTt8wpGA` | 0 -> 0 | 0 |
| 53 | code | `pZps8ZhIvt5A` | 0 -> 0 | 0 |
| 59 | code | `nsgbU7_b0HNC` | 0 -> 0 | 0 |
| 68 | code | `1bc3c4ab` | 0 -> 0 | 0 |
| 73 | code | `8e407371` | 1 -> 1 | 0 |
| 75 | code | `vikRvh1v6KzG` | 1 -> 1 | 0 |
| 76 | markdown | `6a9570d1` | 0 -> 0 | 0 |
| 80 | code | `cb26b34f` | 1 -> 1 | 0 |
| 81 | code | `2uPalKeP_7yZ` | 1 -> 2 | 0 |
| 85 | code | `008d6e32` | 1 -> 1 | 0 |
| 86 | markdown | `6fe99e9e` | 0 -> 0 | 0 |
| 90 | code | `10ad95f9` | 1 -> 1 | 0 |
| 91 | markdown | `oEZUz1sksj9S` | 0 -> 0 | 0 |
| 95 | code | `0d0a97c9` | 1 -> 2 | 0 |
| 96 | markdown | `17f93635` | 0 -> 0 | 0 |
| 98 | code | `EI5H0v2TdVRq` | 1 -> 1 | 0 |
| 102 | code | `86132e4f` | 1 -> 1 | 0 |
| 103 | markdown | `5383b316` | 0 -> 0 | 0 |

## Смысловые изменения графиков

- Ячейки 80 и 81 используют исходную сетку `gamma=L`, `2L`; `delta=1`, `0.5`.
- Ячейка 98 строит пять кривых BFGS для масштабов начального обратного Гессиана `1/L`, `5/L`, `10/L`, `20/L`, `30/L`; для всех используется Wolfe line search с Armijo fallback.
- Ячейка 102 сравнивает семь лучших конфигураций и подписывает кривые именами конфигураций, как в исходном графике.
- Временные оси зависят от компьютера и версии библиотек, поэтому секунды не обязаны совпадать с сохраненными исходными запусками.
- В ячейке 16 train loader использует тренировочную часть CIFAR-10; в исходнике по ошибке использовалась тестовая часть, поэтому изображение-пример намеренно отличается.

## Полные текстовые diff всех измененных ячеек

### Ячейка 2 — code, ID `35ab9136`

```diff
--- исходник
+++ рабочий файл
@@ -5,9 +5,6 @@
 from itertools import product
 from typing import Callable, Iterator, Optional, List, Tuple
 
-import cvxpy as cp
-import jax
-import jax.numpy as jnp
 import matplotlib.pyplot as plt
 import numpy as np
 import scipy
```

### Ячейка 16 — code, ID `a685b55f`

```diff
--- исходник
+++ рабочий файл
@@ -1,5 +1,11 @@
 cifar_transform = transforms.Compose(
-    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
+    [
+        transforms.ToTensor(),
+        transforms.Normalize(
+            (0.5, 0.5, 0.5),
+            (0.5, 0.5, 0.5),
+        ),
+    ]
 )
 
 # загружаем датасет
@@ -32,7 +38,7 @@
 assert len(cifar_test_dataset) == 500
 
 cifar_train_loader = torch.utils.data.DataLoader(
-    cifar_test_dataset, batch_size=64, shuffle=True
+    cifar_train_dataset, batch_size=64, shuffle=True
 )
 
 cifar_test_loader = torch.utils.data.DataLoader(
@@ -52,13 +58,11 @@
     "truck",
 )
 
-
 def imshow(image):
     image = image / 2 + 0.5  # де-нормализуем
     np_image = image.numpy()
     plt.imshow(np.transpose(np_image, (1, 2, 0)))
     plt.show()
-
 
 # несколько случайных элементов
 images, labels = next(iter(cifar_train_loader))
```

### Ячейка 43 — markdown, ID `3NXUSTt8wpGA`

```diff
--- исходник
+++ рабочий файл
@@ -16,3 +16,15 @@
 Матрица 2x2 положительно полуопределена, если все ее главные миноры неотрицательны.
 - $2x_2^2 \geq 0$
 - $\det(\nabla^2 f(x)) = (2x_2^2)(2x_1^2) - (4x_1 x_2)^2 = 4x_1^2 x_2^2 - 16x_1^2 x_2^2 = -12x_1^2 x_2^2$
+
+При $x_1x_2 \ne 0$ определитель гессиана отрицателен, поэтому гессиан
+не является положительно полуопределенным. Значит, функция не выпукла
+на всей $\mathbb{R}^2$. Это также видно из контрпримера:
+
+$$
+A=(3/2,1/2), \quad B=(1/2,3/2), \quad (A+B)/2=(1,1).
+$$
+
+При этом $f(A)=f(B)=9/16$, а $f((A+B)/2)=1>9/16$. Неравенство
+выпуклости нарушено. В формулировке задания утверждение о выпуклости
+неверно.
```

### Ячейка 53 — code, ID `pZps8ZhIvt5A`

```diff
--- исходник
+++ рабочий файл
@@ -1 +1,10 @@
-!gdown --fuzzy https://drive.google.com/file/d/1lgwawQxGD_6XruWbquMH6W2yKPVQ5kdi/view -O mushrooms.txt
+import gdown
+mushroom_url = (
+    "https://drive.google.com/uc?id="
+    "1lgwawQxGD_6XruWbquMH6W2yKPVQ5kdi"
+)
+gdown.download(
+    mushroom_url,
+    "mushrooms.txt",
+    quiet=False,
+)
```

### Ячейка 59 — code, ID `nsgbU7_b0HNC`

```diff
--- исходник
+++ рабочий файл
@@ -1,22 +1,25 @@
 def sigmoid(z):
-    return 1 / (1 + np.exp(-z))
+    return scipy.special.expit(z)
 
 def mush_f(w, x=train_mush_x, y=train_mush_y, lambda_=lambda_):
-    n = x.shape[0]
     z = y * (x @ w)
-    log_likelihood = np.log(1 + np.exp(-z))
-    return np.mean(log_likelihood) + 0.5 * lambda_ * np.dot(w, w)
+    losses = np.logaddexp(0, -z)
+    data_loss = np.mean(losses)
+    regularization = 0.5 * lambda_ * np.dot(w, w)
+    value = data_loss + regularization
+    return value
 
 def mush_grad(w, x=train_mush_x, y=train_mush_y, lambda_=lambda_):
     n = x.shape[0]
     z = y * (x @ w)
     grad = -(y * sigmoid(-z)) @ x / n
-    return grad + lambda_ * w
+    result = grad + lambda_ * w
+    return result
 
 def mush_hess(w, x=train_mush_x, y=train_mush_y, lambda_=lambda_):
     n = x.shape[0]
     z = y * (x @ w)
-    s = sigmoid(-z) * sigmoid(z)
-    S = np.diag(s)
-    hessian = (x.T @ S @ x) / n
-    return hessian + lambda_ * np.eye(x.shape[1])
+    weights = sigmoid(-z) * sigmoid(z)
+    hessian = (x.T * weights) @ x / n
+    hessian = hessian + lambda_ * np.eye(x.shape[1])
+    return hessian
```

### Ячейка 68 — code, ID `1bc3c4ab`

```diff
--- исходник
+++ рабочий файл
@@ -14,34 +14,51 @@
         np.random.seed(420)
         w = start_w.copy()
 
-        # initial eval
-        ... #TODO
+        loss = mush_f(w) #TODO
+        gradient = approx.grad_f(w)
+        criterion = float(np.linalg.norm(gradient))
+        accuracy = mush_accuracy(w)
 
-        # log
-        ... #TODO
+        accuracies.append(accuracy) #TODO
+        criterion_logs.append(criterion)
+        time_logs.append(0.0)
 
         start_time = time.time()
 
-        with tqdm(range(epochs), desc=name) as loop:
+        with tqdm(
+            range(epochs), desc=name, mininterval=1.0
+        ) as loop:
             for k in loop:
-                # train
-                w = approx.step(w, k)
+                next_w = approx.step(w, k)
 
-                # eval
-                ... #TODO
-
-                # Остановить, если результат функции не определён
-                if np.isnan(...) or np.isinf(...): #
+                if not np.all(np.isfinite(next_w)):
                     break
 
-                # log
-                ... #TODO
+                w = next_w
 
+                loss = mush_f(w)
+                gradient = approx.grad_f(w)
+                criterion = float(np.linalg.norm(gradient))
+                accuracy = mush_accuracy(w)
+
+                if not np.isfinite(loss) or not np.isfinite(criterion): #
+                    break
+
+                accuracies.append(accuracy) #TODO
+                criterion_logs.append(criterion)
                 time_logs.append(time.time() - start_time)
-                loop.set_postfix({"Loss": ..., "Accuracy": ..., "Grad norm": ...}) # TODO
 
-                # Остановить, если criterion <= stop_criterion
-                ... #TODO
+                loop.set_postfix(
+                    {
+                        "Loss": loss,
+                        "Accuracy": accuracy,
+                        "Grad norm": criterion,
+                    },
+                    refresh=False,
+                ) # TODO
+
+                if criterion <= stop_criterion: #TODO
+                    break
 
         results_dict[name] = {
             "Epoch": list(range(len(accuracies))),
@@ -52,3 +69,10 @@
         }
 
     return results_dict
+
+class GradientDescent(ManualSolver):
+    def step(self, w: np.ndarray, k: int) -> np.ndarray:
+        gradient = self.grad_f(w)
+        step_size = self.lr(k)
+        next_w = w - step_size * gradient
+        return next_w
```

### Ячейка 73 — code, ID `8e407371`

```diff
--- исходник
+++ рабочий файл
@@ -0,0 +1,9 @@
+gamma_L = 1 / L
+gd_solver = GradientDescent(lr=gamma_L)
+results_gamma_L = check_approx_mush(
+    [("Gradient Descent 1/L", gd_solver)],
+    epochs=500,
+    start_w=start_w_mush,
+    stop_criterion=1e-5,
+)
+plot_criterion_epoch(results_gamma_L)
```

### Ячейка 75 — code, ID `vikRvh1v6KzG`

```diff
--- исходник
+++ рабочий файл
@@ -0,0 +1,38 @@
+gd_approximations = []
+gamma_factors = [0.1, 0.5, 1, 2, 3]
+
+for factor in gamma_factors:
+    name = "Gradient Descent gamma={}/L".format(factor)
+    solver = GradientDescent(lr=factor * gamma_L)
+    gd_approximations.append((name, solver))
+
+gd_experiments = check_approx_mush(
+    gd_approximations,
+    epochs=1000,
+    start_w=start_w_mush,
+    stop_criterion=1e-5,
+)
+
+plot_gd_metrics = partial(
+    draw_plots,
+    plots=[
+        ({"name": "Epoch"}, {"name": "Criterion", "log": True}),
+        ({"name": "Time"}, {"name": "Criterion", "log": True}),
+        ({"name": "Epoch"}, {"name": "Accuracy"}),
+        ({"name": "Time"}, {"name": "Accuracy"}),
+    ],
+)
+plot_gd_metrics(gd_experiments)
+
+best_gd_name = None
+best_gd_criterion = float("inf")
+
+for name, values in gd_experiments.items():
+    final_criterion = values["Criterion"][-1]
+
+    if final_criterion < best_gd_criterion:
+        best_gd_name = name
+        best_gd_criterion = final_criterion
+
+print("Best fixed step:", best_gd_name)
+print("Final gradient norm:", best_gd_criterion)
```

### Ячейка 76 — markdown, ID `6a9570d1`

```diff
--- исходник
+++ рабочий файл
@@ -1 +1,4 @@
-**Выводы**
+По конечной норме градиента лучшей оказалась конфигурация
+`Gradient Descent gamma=3/L`. Точность на тестовой выборке также учитывалась: при близкой
+сходимости предпочтительнее шаг с более высокой точностью. Графики
+сравнивают сходимость по эпохам и времени.
```

### Ячейка 80 — code, ID `cb26b34f`

```diff
--- исходник
+++ рабочий файл
@@ -1,6 +1,40 @@
 class GradientDescentDecay(ManualSolver):
-    def __init__(self, gamma: float, delta: float, grad_f: Callable = mush_grad, hess_f: Callable = mush_hess) -> None:
-        ...
+    def __init__(
+        self,
+        gamma: float,
+        delta: float,
+        grad_f: Callable = mush_grad,
+        hess_f: Callable = mush_hess,
+    ) -> None:
+        super().__init__(gamma / delta, grad_f=grad_f, hess_f=hess_f)
+        self.gamma = gamma
+        self.delta = delta
 
     def step(self, w: np.ndarray, k: int) -> np.ndarray:
-        ...
+        step_size = self.gamma / (self.delta + k)
+        next_w = w - step_size * self.grad_f(w)
+        return next_w
+
+decay_parameters = [
+    (L, 1.0),
+    (2 * L, 1.0),
+    (L, 0.5),
+    (2 * L, 0.5),
+]
+decay_approximations = []
+
+for gamma, delta in decay_parameters:
+    name = "GD decay gamma={:g}, delta={:g}".format(gamma, delta)
+    solver = GradientDescentDecay(gamma=gamma, delta=delta)
+    decay_approximations.append((name, solver))
+
+decay_results = check_approx_mush(
+    decay_approximations,
+    epochs=1000,
+    start_w=start_w_mush,
+    stop_criterion=1e-5,
+)
+plot_gd_metrics(decay_results)
+
+for name, values in decay_results.items():
+    print(name, values["Criterion"][-1], values["Accuracy"][-1])
```

### Ячейка 81 — code, ID `2uPalKeP_7yZ`

```diff
--- исходник
+++ рабочий файл
@@ -1,8 +1,63 @@
 class GradientDescentSqrtDecay(ManualSolver):
-    def __init__(self, gamma: float, delta: float, grad_f: Callable = mush_grad, hess_f: Callable = mush_hess) -> None:
-        super().__init__(lr = gamma / delta, grad_f=grad_f, hess_f=hess_f)
+    def __init__(
+        self,
+        gamma: float,
+        delta: float,
+        grad_f: Callable = mush_grad,
+        hess_f: Callable = mush_hess,
+    ) -> None:
+        super().__init__(gamma / delta, grad_f=grad_f, hess_f=hess_f)
         self.gamma = gamma
         self.delta = delta
 
     def step(self, w: np.ndarray, k: int) -> np.ndarray:
-        ...
+        step_size = self.gamma / (self.delta + math.sqrt(k))
+        next_w = w - step_size * self.grad_f(w)
+        return next_w
+
+sqrt_decay_parameters = [
+    (L, 1.0),
+    (2 * L, 1.0),
+    (L, 0.5),
+    (2 * L, 0.5),
+]
+sqrt_decay_approximations = []
+
+for gamma, delta in sqrt_decay_parameters:
+    name = "GD sqrt decay gamma={:g}, delta={:g}".format(gamma, delta)
+    solver = GradientDescentSqrtDecay(gamma=gamma, delta=delta)
+    sqrt_decay_approximations.append((name, solver))
+
+sqrt_decay_results = check_approx_mush(
+    sqrt_decay_approximations,
+    epochs=1000,
+    start_w=start_w_mush,
+    stop_criterion=1e-5,
+)
+plot_gd_metrics(sqrt_decay_results)
+
+for name, values in sqrt_decay_results.items():
+    print(name, values["Criterion"][-1], values["Accuracy"][-1])
+
+schedule_results = {}
+schedule_groups = [
+    ("Constant step", gd_experiments),
+    ("gamma/(delta+k)", decay_results),
+    ("gamma/(delta+sqrt(k))", sqrt_decay_results),
+]
+
+for group_name, group_results in schedule_groups:
+    selected_name = None
+    selected_criterion = float("inf")
+
+    for name, values in group_results.items():
+        final_criterion = values["Criterion"][-1]
+
+        if final_criterion < selected_criterion:
+            selected_name = name
+            selected_criterion = final_criterion
+
+    schedule_results[group_name] = group_results[selected_name]
+    print("Best", group_name, selected_name, selected_criterion)
+
+plot_gd_metrics(schedule_results, title="Лучшие шаги градиентного спуска")
```

### Ячейка 85 — code, ID `008d6e32`

```diff
--- исходник
+++ рабочий файл
@@ -1 +1,49 @@
-# ваше решение (code + markdown)
+class HeavyBall(ManualSolver):
+    def __init__(
+        self,
+        lr: float | Callable,
+        beta: float,
+        grad_f: Callable = mush_grad,
+        hess_f: Callable = mush_hess,
+    ) -> None:
+        super().__init__(lr, grad_f=grad_f, hess_f=hess_f)
+        self.beta = beta
+        self.velocity = None
+
+    def step(self, w: np.ndarray, k: int) -> np.ndarray:
+        if self.velocity is None:
+            self.velocity = np.zeros_like(w)
+
+        gradient = self.grad_f(w)
+        self.velocity = self.beta * self.velocity - self.lr(k) * gradient
+        next_w = w + self.velocity
+        return next_w
+
+heavy_ball_approximations = []
+
+for factor in [1, 2, 3]:
+    for beta in [0.5, 0.9]:
+        name = "Heavy Ball gamma={}/L, beta={}".format(factor, beta)
+        solver = HeavyBall(lr=factor * gamma_L, beta=beta)
+        heavy_ball_approximations.append((name, solver))
+
+heavy_ball_results = check_approx_mush(
+    heavy_ball_approximations,
+    epochs=500,
+    start_w=start_w_mush,
+    stop_criterion=1e-5,
+)
+plot_gd_metrics(heavy_ball_results)
+
+best_heavy_ball_name = None
+best_heavy_ball_criterion = float("inf")
+
+for name, values in heavy_ball_results.items():
+    final_criterion = values["Criterion"][-1]
+
+    if final_criterion < best_heavy_ball_criterion:
+        best_heavy_ball_name = name
+        best_heavy_ball_criterion = final_criterion
+
+print("Best Heavy Ball:", best_heavy_ball_name)
+print("Final gradient norm:", best_heavy_ball_criterion)
```

### Ячейка 86 — markdown, ID `6fe99e9e`

```diff
--- исходник
+++ рабочий файл
@@ -1 +1,5 @@
-**Выводы**
+Метод тяжёлого шарика использует прошлое направление движения.
+Параметр $\beta$ задаёт долю импульса: при большем значении движение
+сильнее сглаживается и ускоряется в устойчивом направлении, но при
+слишком большом значении возможны колебания. Лучший вариант по конечной
+норме градиента: `Heavy Ball gamma=3/L, beta=0.9`.
```

### Ячейка 90 — code, ID `10ad95f9`

```diff
--- исходник
+++ рабочий файл
@@ -1 +1,50 @@
-# ваше решение (code + markdown)
+class NesterovMomentum(ManualSolver):
+    def __init__(
+        self,
+        lr: float | Callable,
+        beta: float,
+        grad_f: Callable = mush_grad,
+        hess_f: Callable = mush_hess,
+    ) -> None:
+        super().__init__(lr, grad_f=grad_f, hess_f=hess_f)
+        self.beta = beta
+        self.velocity = None
+
+    def step(self, w: np.ndarray, k: int) -> np.ndarray:
+        if self.velocity is None:
+            self.velocity = np.zeros_like(w)
+
+        lookahead = w + self.beta * self.velocity
+        gradient = self.grad_f(lookahead)
+        self.velocity = self.beta * self.velocity - self.lr(k) * gradient
+        next_w = w + self.velocity
+        return next_w
+
+nesterov_approximations = []
+
+for factor in [1, 2, 3]:
+    for beta in [0.5, 0.9]:
+        name = "Nesterov gamma={}/L, beta={}".format(factor, beta)
+        solver = NesterovMomentum(lr=factor * gamma_L, beta=beta)
+        nesterov_approximations.append((name, solver))
+
+nesterov_results = check_approx_mush(
+    nesterov_approximations,
+    epochs=500,
+    start_w=start_w_mush,
+    stop_criterion=1e-5,
+)
+plot_gd_metrics(nesterov_results)
+
+best_nesterov_name = None
+best_nesterov_criterion = float("inf")
+
+for name, values in nesterov_results.items():
+    final_criterion = values["Criterion"][-1]
+
+    if final_criterion < best_nesterov_criterion:
+        best_nesterov_name = name
+        best_nesterov_criterion = final_criterion
+
+print("Best Nesterov:", best_nesterov_name)
+print("Final gradient norm:", best_nesterov_criterion)
```

### Ячейка 91 — markdown, ID `oEZUz1sksj9S`

```diff
--- исходник
+++ рабочий файл
@@ -1 +1,4 @@
-Кажется что лучший Нестеров каждый раз перелетает минимум, хотя очень хорошо сходится
+Метод Нестерова считает градиент в экстраполированной точке
+$x_k+\beta h_{k-1}$. Среди проверенных конфигураций наименьшую
+конечную норму градиента показала `Nesterov gamma=3/L, beta=0.9`. Значение точности и графики
+также использованы при сравнении.
```

### Ячейка 95 — code, ID `0d0a97c9`

```diff
--- исходник
+++ рабочий файл
@@ -1 +1,79 @@
-# ваше решение (code + markdown)
+class DampedNewton(ManualSolver):
+    def step(self, w: np.ndarray, k: int) -> np.ndarray:
+        gradient = self.grad_f(w)
+        hessian = self.hess_f(w)
+        direction = np.linalg.solve(hessian, gradient)
+        current_loss = mush_f(w)
+        directional_decrease = np.dot(gradient, direction)
+        step_size = self.lr(k)
+        candidate = w - step_size * direction
+        attempt = 0
+
+        while attempt < 40:
+            candidate_loss = mush_f(candidate)
+            armijo_limit = (
+                current_loss - 1e-4 * step_size * directional_decrease
+            )
+
+            if np.isfinite(candidate_loss) and candidate_loss <= armijo_limit:
+                return candidate
+
+            step_size = step_size * 0.5
+            candidate = w - step_size * direction
+            attempt = attempt + 1
+
+        return w.copy()
+
+newton_rates = [
+    ("Newton gamma=1", 1.0),
+    ("Newton gamma=0.1/L", 0.1 * gamma_L),
+    ("Newton gamma=0.5/L", 0.5 * gamma_L),
+    ("Newton gamma=1/L", gamma_L),
+    ("Newton gamma=2/L", 2 * gamma_L),
+]
+newton_approximations = []
+
+for name, rate in newton_rates:
+    solver = DampedNewton(lr=rate)
+    newton_approximations.append((name, solver))
+
+newton_results = check_approx_mush(
+    newton_approximations,
+    epochs=500,
+    start_w=start_w_mush,
+    stop_criterion=1e-5,
+)
+plot_gd_metrics(newton_results)
+
+newton_initial_results = {}
+initial_points = [
+    ("Newton start zero", np.zeros_like(start_w_mush)),
+    ("Newton start ones", np.ones_like(start_w_mush)),
+]
+
+for name, initial_point in initial_points:
+    solver = DampedNewton(lr=1.0)
+    current_results = check_approx_mush(
+        [(name, solver)],
+        epochs=500,
+        start_w=initial_point,
+        stop_criterion=1e-5,
+    )
+    newton_initial_results.update(current_results)
+
+plot_gd_metrics(newton_initial_results, title="Newton: выбор начальной точки")
+
+best_newton_name = None
+best_newton_criterion = float("inf")
+
+for name, values in newton_results.items():
+    final_criterion = values["Criterion"][-1]
+
+    if final_criterion < best_newton_criterion:
+        best_newton_name = name
+        best_newton_criterion = final_criterion
+
+print("Best damped Newton:", best_newton_name)
+print("Final gradient norm:", best_newton_criterion)
+for name, values in newton_initial_results.items():
+    print(name, values["Criterion"][-1], values["Accuracy"][-1])
```

### Ячейка 96 — markdown, ID `17f93635`

```diff
--- исходник
+++ рабочий файл
@@ -1 +1,4 @@
-**Выводы**
+Демпфирование уменьшает шаг Ньютона, если полный шаг не снижает
+значение функции. По конечной норме градиента лучшей оказалась
+конфигурация `Newton gamma=1`. Запуски из нулевого и единичного векторов показывают,
+как начальная точка влияет на траекторию и время сходимости.
```

### Ячейка 98 — code, ID `EI5H0v2TdVRq`

```diff
--- исходник
+++ рабочий файл
@@ -1 +1,91 @@
-# ваше решение (code + markdown)
+class BFGSSolver(ManualSolver):
+    def __init__(
+        self,
+        initial_scale: float,
+        grad_f: Callable = mush_grad,
+    ) -> None:
+        super().__init__(1.0, grad_f=grad_f, hess_f=mush_hess)
+        self.initial_scale = initial_scale
+        self.inverse_hessian = None
+        self.previous_w = None
+        self.previous_gradient = None
+
+    def step(self, w: np.ndarray, k: int) -> np.ndarray:
+        gradient = self.grad_f(w)
+
+        if self.inverse_hessian is None:
+            self.inverse_hessian = self.initial_scale * np.eye(w.size)
+        elif self.previous_w is not None:
+            s = w - self.previous_w
+            y = gradient - self.previous_gradient
+            curvature = np.dot(s, y)
+
+            if curvature > 1e-12:
+                rho = 1.0 / curvature
+                identity = np.eye(w.size)
+                transform = identity - rho * np.outer(s, y)
+                self.inverse_hessian = (
+                    transform @ self.inverse_hessian @ transform.T
+                    + rho * np.outer(s, s)
+                )
+
+        direction = -(self.inverse_hessian @ gradient)
+        current_loss = mush_f(w)
+        line_search_result = scipy.optimize.line_search(
+            mush_f,
+            self.grad_f,
+            w,
+            direction,
+            gfk=gradient,
+            old_fval=current_loss,
+        )
+        step_size = line_search_result[0]
+
+        if step_size is None or not np.isfinite(step_size):
+            step_size = 1.0
+            directional_change = np.dot(gradient, direction)
+            accepted = False
+
+            for attempt in range(40):
+                candidate = w + step_size * direction
+                candidate_loss = mush_f(candidate)
+                armijo_limit = (
+                    current_loss + 1e-4 * step_size * directional_change
+                )
+
+                if (
+                    np.isfinite(candidate_loss)
+                    and candidate_loss <= armijo_limit
+                ):
+                    accepted = True
+                    break
+
+                step_size = step_size * 0.5
+
+            if not accepted:
+                candidate = w.copy()
+        else:
+            candidate = w + step_size * direction
+
+        self.previous_w = w.copy()
+        self.previous_gradient = gradient.copy()
+        return candidate
+
+bfgs_approximations = []
+bfgs_factors = [1, 5, 10, 20, 30]
+
+for factor in bfgs_factors:
+    name = "BFGS gamma={}/L".format(factor)
+    solver = BFGSSolver(initial_scale=factor * gamma_L)
+    bfgs_approximations.append((name, solver))
+
+bfgs_results = check_approx_mush(
+    bfgs_approximations,
+    epochs=1000,
+    start_w=start_w_mush,
+    stop_criterion=1e-5,
+)
+plot_gd_metrics(bfgs_results)
+
+for name, values in bfgs_results.items():
+    print(name, values["Criterion"][-1], values["Accuracy"][-1])
```

### Ячейка 102 — code, ID `86132e4f`

```diff
--- исходник
+++ рабочий файл
@@ -1 +1,36 @@
-# ваше решение (code + markdown)
+method_groups = [
+    ("Gradient Descent", gd_experiments),
+    ("Gradient Descent decay", decay_results),
+    ("Gradient Descent sqrt decay", sqrt_decay_results),
+    ("Heavy Ball", heavy_ball_results),
+    ("Nesterov", nesterov_results),
+    ("Damped Newton", newton_results),
+    ("BFGS", bfgs_results),
+]
+comparison_results = {}
+best_configurations = {}
+
+for method_name, method_results in method_groups:
+    best_name = None
+    best_criterion = float("inf")
+
+    for name, values in method_results.items():
+        final_criterion = values["Criterion"][-1]
+
+        if final_criterion < best_criterion:
+            best_name = name
+            best_criterion = final_criterion
+
+    comparison_results[best_name] = method_results[best_name]
+    best_configurations[method_name] = best_name
+    print(
+        method_name,
+        best_name,
+        "criterion", best_criterion,
+        "accuracy", method_results[best_name]["Accuracy"][-1],
+    )
+
+plot_gd_metrics(
+    comparison_results,
+    title="Лучшие конфигурации методов оптимизации",
+)
```

### Ячейка 103 — markdown, ID `5383b316`

```diff
--- исходник
+++ рабочий файл
@@ -1 +1,13 @@
-**Выводы**
+По минимальной конечной норме градиента лучшим среди выбранных запусков оказался демпфированный метод Ньютона с gamma=1. В таблице приведена лучшая конфигурация каждого метода; точность указана на тестовой выборке.
+
+| Метод | Конфигурация | Норма градиента | Accuracy |
+| --- | --- | ---: | ---: |
+| Gradient Descent | gamma=3/L | 0.001831 | 0.9889 |
+| Убывающий шаг | gamma=10.5105, delta=1 | 0.055561 | 0.9717 |
+| Убывающий шаг с корнем | gamma=10.5105, delta=0.5 | 0.001373 | 0.9889 |
+| Heavy Ball | gamma=3/L, beta=0.9 | 0.000008 | 0.9889 |
+| Nesterov | gamma=3/L, beta=0.9 | 0.000010 | 0.9889 |
+| Newton | gamma=1 | 0.000002 | 0.9889 |
+| BFGS | gamma=5/L | 0.000007 | 0.9889 |
+
+Сравнение использует норму градиента, точность и графики времени. Коэффициент BFGS задает масштаб начального приближения обратного Гессиана; поиск шага Wolfe выполняется отдельно на каждом шаге.
```
