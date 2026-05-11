import networkx as nx
import numpy as np
from matplotlib.backend_bases import MouseButton
from shapely.geometry import Point, Polygon, LineString
from model.geometry_manager import GeometryManager
from model.route_planner import RobustRoutePlanner
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


class EventHandler:
    def __init__(self, geometry_manager, plot_manager):
        """
        Инициализация обработчика событий

        Args:
            geometry_manager: Экземпляр GeometryManager
            plot_manager: Экземпляр PlotManager
        """
        self.geometry_manager = geometry_manager
        self.plot_manager = plot_manager
        self.cid = None
        self.selected_centers = []

        # Подключаем обработчики событий
        self.connect_events()

    def connect_events(self):
        """Подключает все обработчики событий"""
        # Обработчик кликов мыши
        self.cid = self.plot_manager.fig.canvas.mpl_connect(
            'button_press_event', self.onclick)

        # Подключение кнопок
        self.plot_manager.btn_finish_region.on_clicked(self.finish_region)
        self.plot_manager.btn_finish_building.on_clicked(self.finish_building)
        self.plot_manager.btn_optimize.on_clicked(self.run_optimization)
        self.plot_manager.btn_coverage.on_clicked(self.calculate_building_coverage)
        self.plot_manager.btn_build_route.on_clicked(self.build_route)

    def onclick(self, event):
        """Обрабатывает клики мыши на карте"""
        if event.inaxes != self.plot_manager.ax:
            return

        point = (event.xdata, event.ydata)

        if event.button == MouseButton.LEFT:  # Левый клик - участок
            self.geometry_manager.add_region_point(point)
            self.plot_manager.plot_point(point, 'region')
        elif event.button == MouseButton.RIGHT:  # Правый клик - здание
            self.geometry_manager.add_building_point(point)
            self.plot_manager.plot_point(point, 'building')

    def finish_region(self, event):
        """Завершает рисование участка"""
        if self.geometry_manager.finish_region():
            self.plot_manager.plot_polygon(
                self.geometry_manager.region, 'region')
            print("Участок завершен")

    def finish_building(self, event):
        """Завершает рисование здания"""
        if self.geometry_manager.finish_building():
            self.plot_manager.plot_polygon(
                self.geometry_manager.buildings[-1], 'building')
            print(f"Здание завершено. Всего зданий: {len(self.geometry_manager.buildings)}")

    def run_optimization(self, event):
        """Запускает оптимизацию покрытия"""
        if not self.geometry_manager.region or not self.geometry_manager.buildings:
            print("Ошибка: сначала нарисуйте участок и здания!")
            return

        print("Запуск оптимизации покрытия...")

        self.selected_centers, num_circles = self.geometry_manager.optimize_coverage()

        if self.selected_centers:
            print(f"Оптимизация завершена. Найдено {num_circles} окружностей")
            self._visualize_coverage()
        else:
            print("Не удалось выполнить оптимизацию")

    def _visualize_coverage(self):
        """Визуализирует результаты оптимизации покрытия"""
        self.plot_manager.clear_plots(preserve=['map', 'buildings', 'region'])

        R = 60.0 / 111320  # Радиус в градусах
        for idx, center in enumerate(self.selected_centers):
            # Рисуем окружности (серым цветом)
            circle = plt.Circle(center, R, color="#0EED85",
                                fill=False, linestyle='--')
            self.plot_manager.ax.add_patch(circle)

            # Рисуем центры (темно-зеленым)
            self.plot_manager.ax.plot(
                center[0], center[1], 'o',
                color=self.plot_manager.colors['center'],
                markersize=8)

            # Добавляем номера (белым)
            self.plot_manager.ax.text(
                center[0], center[1], str(idx + 1),
                color='white', ha='center', va='center',
                fontweight='bold')

        self.plot_manager.fig.canvas.draw()

    def calculate_building_coverage(self, event):
        """Рассчитывает процент покрытия зданиями"""
        if not self.geometry_manager.region:
            print("Ошибка: сначала нарисуйте участок!")
            return

        coverage = self.geometry_manager.calculate_building_coverage()
        print(f"Процент покрытия зданиями: {coverage:.2f}%")

    def build_route(self, event):
        """Строит маршрут через все центры"""
        if not hasattr(self, 'selected_centers') or not self.selected_centers:
            print("Ошибка: сначала выполните оптимизацию покрытия!")
            return

        print("Построение маршрута через все центры...")

        route_planner = RobustRoutePlanner(
            self.selected_centers,
            self.geometry_manager.buildings
        )

        # Визуализируем сеть соединений
        network = route_planner.build_network()
        self.plot_manager.plot_network(network, self.selected_centers)

        # Строим и отображаем маршрут
        route = route_planner.find_optimal_route()

        if route:
            self._process_route_results(route, route_planner)
        else:
            print("Не удалось построить маршрут через все центры")

    def _process_route_results(self, route, route_planner):
        """Обрабатывает и отображает результаты построения маршрута"""
        # Отображаем маршрут
        self.plot_manager.plot_route(route)

        # Проверяем посещение центров
        visited = self._check_visited_centers(route)
        print(f"\nРезультаты маршрутизации:")
        print(f"Посещено центров: {len(visited)}/{len(self.selected_centers)}")

        # Подсвечиваем результаты
        self.plot_manager.highlight_centers(self.selected_centers, visited)

        # Рассчитываем длину маршрута
        length = sum(Point(route[i]).distance(Point(route[i + 1]))
                     for i in range(len(route) - 1)) * 111320
        print(f"Общая длина маршрута: {length:.2f} метров")

        # Выводим проблемные центры
        if len(visited) < len(self.selected_centers):
            missed = set(range(len(self.selected_centers))) - visited
            print("Не посещены центры:", [i + 1 for i in sorted(missed)])

        # Анализ связности
        self._analyze_connectivity(route_planner.graph)

    def _analyze_connectivity(self, graph):
        """Анализирует связность графа"""
        if not nx.is_connected(graph):
            print("\nПредупреждение: не все центры связаны напрямую")
            components = list(nx.connected_components(graph))
            print(f"Найдено {len(components)} изолированных групп")

            # Проверяем, все ли центры в одной компоненте связности
            all_centers = set(range(len(self.selected_centers)))
            if not any(all_centers.issubset(c) for c in components):
                print("Критическая ошибка: невозможно посетить все центры одним маршрутом")

    def _check_visited_centers(self, route):
        """Определяет какие центры были посещены"""
        visited = set()
        tolerance = 1e-6  # Погрешность сравнения

        for i, center in enumerate(self.selected_centers):
            if any(Point(center).distance(Point(p)) < tolerance for p in route):
                visited.add(i)

        return visited

    def disconnect(self):
        """Отключает обработчики событий"""
        if self.cid:
            self.plot_manager.fig.canvas.mpl_disconnect(self.cid)
            self.cid = None
        print("Обработчики событий отключены")

    def __del__(self):
        """Деструктор - автоматически отключает обработчики"""
        self.disconnect()