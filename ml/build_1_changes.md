# Изменения в build_1.ipynb

Сравнение рабочего файла `build_1.ipynb` с неизменяемым исходником `opt_hw _ downloaded.ipynb`.

## Граница изменений и проверка

- Код и текст ячеек 1-68 совпадают с оригиналом. Исходный файл не менялся.
- Решение и ответы находятся начиная с ячейки 69.
- Временная копия целиком исполнена на чистом ядре: 31 ячейка с кодом, исключений нет.
- Сохранено PNG-изображений: 14 всего, из них 12 созданы после защищенной границы; ячейки: 70, 74, 76, 81, 82, 86, 91, 96, 99, 103.
- Потоковые обновления tqdm сведены к последней строке каждой полосы прогресса; численные результаты и графики сохранены.
- Время на оси графика зависит от компьютера. Кривая BFGS отличается от эталона по траектории: поиск Wolfe в этой реализации приводит к норме менее 1e-5 за 44-112 итераций, а эталонный график показывает 1000 итераций с нормой от 0.0549 до 0.0924. В исходном ноутбуке код BFGS не приведен.

## Ячейки с измененным содержимым

Источники: 69, 70, 74, 76, 77, 80, 81, 82, 85, 86, 87, 90, 91, 92, 95, 96, 97, 99, 102, 103, 104.

Выводы и счетчики выполнения обновлены в ячейках: 69, 70, 74, 75, 76, 81, 82, 86, 91, 96, 99, 103.

### Ячейка 69 (code)

```diff
--- original: cell 69
+++ build_1: cell 69
@@ -15,33 +15,78 @@
         w = start_w.copy()
 
         # initial eval
-        ... #TODO
+        #TODO
+        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
+            loss = mush_f(w)
+            gradient = approx.grad_f(w)
+            criterion = float(np.linalg.norm(gradient))
+            accuracy = mush_accuracy(w)
 
         # log
-        ... #TODO
+        #TODO
+        accuracies.append(accuracy)
+        criterion_logs.append(criterion)
+        time_logs.append(0.0)
 
         start_time = time.time()
 
-        with tqdm(range(epochs), desc=name) as loop:
+        with tqdm(
+            range(epochs),
+            desc=name,
+            mininterval=1.0,
+        ) as loop:
             for k in loop:
                 # train
-                w = approx.step(w, k)
+                with np.errstate(
+                    over="ignore",
+                    invalid="ignore",
+                    divide="ignore",
+                ):
+                    next_w = approx.step(w, k)
+
+                if not np.all(np.isfinite(next_w)):
+                    break
 
                 # eval
-                ... #TODO
+                with np.errstate(
+                    over="ignore",
+                    invalid="ignore",
+                    divide="ignore",
+                ):
+                    next_loss = mush_f(next_w)
+                    next_gradient = approx.grad_f(next_w)
+                    next_criterion = float(np.linalg.norm(next_gradient))
+                    next_accuracy = mush_accuracy(next_w)
 
-                # Остановить, если результат функции не определён
-                if np.isnan(...) or np.isinf(...): #
+                if not np.all(
+                    np.isfinite([next_loss, next_criterion, next_accuracy])
+                ):
                     break
 
+                w = next_w
+                loss = next_loss
+                criterion = next_criterion
+                accuracy = next_accuracy
+
                 # log
-                ... #TODO
+                #TODO
+                accuracies.append(accuracy)
+                criterion_logs.append(criterion)
+                time_logs.append(time.time() - start_time)
 
-                time_logs.append(time.time() - start_time)
-                loop.set_postfix({"Loss": ..., "Accuracy": ..., "Grad norm": ...}) # TODO
+                #TODO
+                loop.set_postfix(
+                    {
+                        "Loss": loss,
+                        "Accuracy": accuracy,
+                        "Grad norm": criterion,
+                    },
+                    refresh=False,
+                )
 
                 # Остановить, если criterion <= stop_criterion
-                ... #TODO
+                if criterion <= stop_criterion:
+                    break
 
         results_dict[name] = {
             "Epoch": list(range(len(accuracies))),
@@ -52,3 +97,8 @@
         }
 
     return results_dict
+
+class GradientDescent(ManualSolver):
+    def step(self, w: np.ndarray, k: int) -> np.ndarray:
+        next_w = w - self.lr(k) * self.grad_f(w)
+        return next_w
```

### Ячейка 70 (code)

```diff
--- original: cell 70
+++ build_1: cell 70
@@ -4,7 +4,12 @@
 
 gd_solver = GradientDescent(lr=0.001)
 
-results = check_approx_mush([("Gradient Descent", gd_solver)], epochs=500, start_w=start_w_mush, stop_criterion=1e-5)
+results = check_approx_mush(
+    [("Gradient Descent", gd_solver)],
+    epochs=500,
+    start_w=start_w_mush,
+    stop_criterion=1e-5,
+)
 
 plot_criterion_epoch = partial(
     draw_plots,
```

### Ячейка 74 (code)

```diff
--- original: cell 74
+++ build_1: cell 74
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

### Ячейка 76 (code)

```diff
--- original: cell 76
+++ build_1: cell 76
@@ -0,0 +1,25 @@
+gd_approximations = []
+gamma_factors = [0.1, 0.5, 1, 2, 3]
+
+for factor in gamma_factors:
+    name = "GD gamma=${} / L$".format(factor)
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
```

### Ячейка 77 (markdown)

```diff
--- original: cell 77
+++ build_1: cell 77
@@ -1 +1,3 @@
 **Выводы**
+
+На этом наборе данных среди проверенных постоянных шагов лучший итог дала конфигурация $\gamma=3/L$: норма градиента равна 0.00183, точность на обучающей выборке составила 0.9889. Увеличение шага ускорило сходимость в проверенном диапазоне, но слишком большой шаг может сделать спуск неустойчивым.
```

### Ячейка 80 (markdown)

```diff
--- original: cell 80
+++ build_1: cell 80
@@ -1 +1,9 @@
 ### Решение
+
+Для шага $\gamma_k=\gamma/(\delta+k)$ проверены $\gamma \in \{L,2L\}$ и $\delta \in \{1,0.5\}$. Наименьшая итоговая норма градиента получена при $\gamma=2L$ и $\delta=1$.
+
+| Конфигурация | Норма градиента | Accuracy |
+|---|---:|---:|
+| $2L, 1$ | 0.05556 | 0.9717 |
+
+В этих опытах обычное убывание шага оказалось медленнее схемы с $\sqrt{k}$ и постоянного шага $3/L$.
```

### Ячейка 81 (code)

```diff
--- original: cell 81
+++ build_1: cell 81
@@ -1,6 +1,37 @@
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
```

### Ячейка 82 (code)

```diff
--- original: cell 82
+++ build_1: cell 82
@@ -1,8 +1,62 @@
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
+schedule_groups = [
+    ("Constant step", gd_experiments),
+    ("gamma/(delta+k)", decay_results),
+    ("gamma/(delta+sqrt(k))", sqrt_decay_results),
+]
+schedule_results = {}
+
+for group_name, group_results in schedule_groups:
+    best_name = None
+    best_criterion = float("inf")
+
+    for name, values in group_results.items():
+        final_criterion = values["Criterion"][-1]
+
+        if final_criterion < best_criterion:
+            best_name = name
+            best_criterion = final_criterion
+
+    schedule_results[group_name] = group_results[best_name]
+
+plot_gd_metrics(
+    schedule_results,
+    title="Gradient descent step schedules",
+)
```

### Ячейка 85 (markdown)

```diff
--- original: cell 85
+++ build_1: cell 85
@@ -1 +1,3 @@
 ### Решение
+
+Проверены $\gamma \in \{1/L,2/L,3/L\}$ и $\beta \in \{0.5,0.9\}$. Для каждого сочетания записывались норма градиента, точность и время работы.
```

### Ячейка 86 (code)

```diff
--- original: cell 86
+++ build_1: cell 86
@@ -1 +1,37 @@
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
+        rate = factor * gamma_L
+        name = "Heavy Ball gamma={}, beta={}".format(rate, beta)
+        solver = HeavyBall(lr=rate, beta=beta)
+        heavy_ball_approximations.append((name, solver))
+
+heavy_ball_results = check_approx_mush(
+    heavy_ball_approximations,
+    epochs=500,
+    start_w=start_w_mush,
+    stop_criterion=1e-5,
+)
+plot_gd_metrics(heavy_ball_results)
```

### Ячейка 87 (markdown)

```diff
--- original: cell 87
+++ build_1: cell 87
@@ -1 +1,3 @@
 **Выводы**
+
+Лучший из проверенных результатов получен при $\gamma=3/L$ и $\beta=0.9$: норма градиента 8.24e-6, точность 0.9889; остановка произошла после 199 шагов. В этой сетке параметров большее значение $\beta$ дало более быструю сходимость.
```

### Ячейка 90 (markdown)

```diff
--- original: cell 90
+++ build_1: cell 90
@@ -1 +1,3 @@
 ### Решение
+
+Нестеров, как и Heavy Ball, использует параметр инерции $\beta$. При вычислении следующего шага градиент берется в предсказанной точке $w+\beta v$, где $v$ обозначает текущую скорость. Это предварительное смещение отличает метод Нестерова от Heavy Ball.
```

### Ячейка 91 (code)

```diff
--- original: cell 91
+++ build_1: cell 91
@@ -1 +1,38 @@
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
+        rate = factor * gamma_L
+        name = "Nesterov gamma={}, beta={}".format(rate, beta)
+        solver = NesterovMomentum(lr=rate, beta=beta)
+        nesterov_approximations.append((name, solver))
+
+nesterov_results = check_approx_mush(
+    nesterov_approximations,
+    epochs=500,
+    start_w=start_w_mush,
+    stop_criterion=1e-5,
+)
+plot_gd_metrics(nesterov_results)
```

### Ячейка 92 (markdown)

```diff
--- original: cell 92
+++ build_1: cell 92
@@ -1 +1,3 @@
 Кажется что лучший Нестеров каждый раз перелетает минимум, хотя очень хорошо сходится
+
+При $\beta=0.9$ метод действительно может колебаться около минимума. Лучший проверенный вариант использует $\gamma=3/L$ и $\beta=0.9$: норма градиента 9.81e-6 после 169 шагов, точность 0.9889.
```

### Ячейка 95 (markdown)

```diff
--- original: cell 95
+++ build_1: cell 95
@@ -1 +1,3 @@
 ### Решение
+
+Реализован демпфированный шаг Ньютона $w_{k+1}=w_k-\gamma_k H(w_k)^{-1}g(w_k)$. Для решения системы используется `numpy.linalg.solve`, без явного вычисления обратной матрицы. Проверены несколько масштабов шага и две начальные точки.
```

### Ячейка 96 (code)

```diff
--- original: cell 96
+++ build_1: cell 96
@@ -1 +1,49 @@
-# ваше решение (code + markdown)
+class DampedNewton(ManualSolver):
+    def step(self, w: np.ndarray, k: int) -> np.ndarray:
+        gradient = self.grad_f(w)
+        hessian = self.hess_f(w)
+        direction = np.linalg.solve(hessian, gradient)
+        next_w = w - self.lr(k) * direction
+        return next_w
+
+newton_rates = [
+    ("Newton gamma=1", 1.0),
+    ("Newton gamma=$0.1 / L$", 0.1 * gamma_L),
+    ("Newton gamma=$0.5 / L$", 0.5 * gamma_L),
+    ("Newton gamma=$1 / L$", gamma_L),
+    ("Newton gamma=$2 / L$", 2 * gamma_L),
+]
+newton_approximations = []
+
+for name, rate in newton_rates:
+    solver = DampedNewton(lr=rate)
+    newton_approximations.append((name, solver))
+
+newton_results = check_approx_mush(
+    newton_approximations,
+    epochs=300,
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
+plot_gd_metrics(
+    newton_initial_results,
+    title="Newton starting points",
+)
```

### Ячейка 97 (markdown)

```diff
--- original: cell 97
+++ build_1: cell 97
@@ -1 +1,5 @@
 **Выводы**
+
+Среди проверенных масштабов шага наименьшая норма градиента получена при $\gamma=2/L$: 6.29e-6 за 24 шага. При $\gamma=1/L$ критерий достиг порога за 54 шага. Полный шаг $\gamma=1$ из случайной начальной точки оказался неустойчивым и остановился после нечислового значения.
+
+Из нулевой точки метод достиг нормы 6.77e-8 за 5 шагов. Из вектора единиц уже первый шаг дал нечисловое значение. Значит, результат метода Ньютона зависит от масштаба шага и начальной точки.
```

### Ячейка 99 (code)

```diff
--- original: cell 99
+++ build_1: cell 99
@@ -1 +1,93 @@
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
+        if step_size is None or not np.isfinite(step_size) or step_size <= 0:
+            step_size = 1.0
+            directional_change = np.dot(gradient, direction)
+            accepted = False
+
+            for attempt in range(40):
+                candidate = w + step_size * direction
+                with np.errstate(
+                    over="ignore",
+                    invalid="ignore",
+                    divide="ignore",
+                ):
+                    candidate_loss = mush_f(candidate)
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
+    name = "BFGS gamma=${} / L$".format(factor)
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
```

### Ячейка 102 (markdown)

```diff
--- original: cell 102
+++ build_1: cell 102
@@ -1 +1,5 @@
 ### Решение
+
+BFGS приближенно обновляет обратную матрицу Гессе, используя изменение градиента и параметров. В этой реализации длина шага подбирается поиском Wolfe. Проверены масштабы $\gamma \in \{1/L,5/L,10/L,20/L,30/L\}$.
+
+На графике из условия все варианты BFGS выполняют 1000 итераций и заканчивают с нормой градиента от 0.0549 до 0.0924. В текущей реализации поиск шага по условиям Вольфе приводит к норме менее 1e-5 за 44-112 итераций. Это различие траекторий алгоритма, а не скорости компьютера. В исходном ноутбуке код BFGS оставлен пустым, поэтому по одному графику восстановить точную реализацию автора нельзя.
```

### Ячейка 103 (code)

```diff
--- original: cell 103
+++ build_1: cell 103
@@ -1 +1,27 @@
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
+
+plot_gd_metrics(comparison_results)
```

### Ячейка 104 (markdown)

```diff
--- original: cell 104
+++ build_1: cell 104
@@ -1 +1,15 @@
 **Выводы**
+
+В этой серии опытов наименьшая конечная норма градиента получена у метода Ньютона с $\gamma=2/L$. BFGS с поиском Wolfe тоже достигает критерия $10^{-5}$; измеренное время зависит от компьютера.
+
+| Метод | Лучшая проверенная конфигурация | Норма градиента | Accuracy |
+|---|---|---:|---:|
+| Градиентный спуск | $3/L$ | 0.00183 | 0.9889 |
+| Спуск, $\gamma/(\delta+k)$ | $2L, 1$ | 0.05556 | 0.9717 |
+| Спуск, $\gamma/(\delta+\sqrt{k})$ | $2L, 0.5$ | 0.00137 | 0.9889 |
+| Heavy Ball | $3/L, 0.9$ | 8.24e-6 | 0.9889 |
+| Нестеров | $3/L, 0.9$ | 9.81e-6 | 0.9889 |
+| Ньютон | $2/L$ | 6.29e-6 | 0.9889 |
+| BFGS | $5/L$ | 7.09e-6 | 0.9889 |
+
+Если сравнивать только норму градиента, лучший результат здесь у Ньютона. Для практического выбора также важны устойчивость, стоимость шага и время на конкретном компьютере.
```
