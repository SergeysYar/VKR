import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image
from shapely.geometry import Point


class PlotManager:
    def __init__(self, min_lon, max_lon, min_lat, max_lat, image_path=None):
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.fig, self.ax = None, None
        self.image_path = image_path
        self.image = None

        # Цветовая схема
        self.colors = {
            'route': '#FF4500',  # Оранжевый
            'center': '#006400',  # Темно-зеленый
            'visited': "#31CF31",  # Ярко-зеленый
            'missed': '#FF0000',  # Красный
            'connection': "#04D1FF",  # Серый
            'building': "#1F1B9F",  # Темно-серый
            'current_building': "#A1B822",  # Димп-серый
            'region': '#000000'  # Черный
        }

    def setup_plot(self):
        """Инициализация графического интерфейса"""
        self.fig, self.ax = plt.subplots(figsize=(10, 10))

        if self.image_path:
            try:
                self.image = np.array(Image.open(self.image_path))
                self.ax.imshow(
                    self.image,
                    extent=[self.max_lon, self.min_lon,
                            self.max_lat, self.min_lat],
                    origin="upper"
                )
            except Exception as e:
                print(f"Ошибка загрузки изображения: {e}")
                self.image = None

        self.ax.set_xlim(self.max_lon, self.min_lon)
        self.ax.set_ylim(self.max_lat, self.min_lat)
        self.ax.set_aspect('equal')
        self.ax.set_title("")
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")

        self._create_buttons()

    def _create_buttons(self):
        """Создает элементы управления"""
        self.ax_button_finish_region = plt.axes([0.7, 0.02, 0.2, 0.05])
        self.btn_finish_region = Button(self.ax_button_finish_region, 'Завершить участок')

        self.ax_button_finish_building = plt.axes([0.4, 0.02, 0.2, 0.05])
        self.btn_finish_building = Button(self.ax_button_finish_building, 'Завершить здание')

        self.ax_button_optimize = plt.axes([0.1, 0.02, 0.2, 0.05])
        self.btn_optimize = Button(self.ax_button_optimize, 'Оптимизация')

        self.ax_button_coverage = plt.axes([0.7, 0.00, 0.1, 0.02])
        self.btn_coverage = Button(self.ax_button_coverage, 'Покрытие зданиями')

        self.ax_button_build_route = plt.axes([0.1, 0.08, 0.2, 0.05])
        self.btn_build_route = Button(self.ax_button_build_route, 'Построить маршрут')

    def plot_point(self, point, point_type):
        """Отрисовывает точку в зависимости от типа"""
        colors = {
            'region': ('o', self.colors['region']),
            'building': ('o', self.colors['building']),
            'current_building': ('o', self.colors['current_building'])
        }
        marker, color = colors.get(point_type, ('o', 'black'))
        self.ax.plot(point[0], point[1], marker, color=color)
        self.fig.canvas.draw()

    def plot_polygon(self, polygon, polygon_type, alpha=0.5):
        """Отрисовывает полигон в зависимости от типа"""
        if polygon_type == 'region':
            x, y = polygon.exterior.xy
            self.ax.plot(x, y, color=self.colors['region'], linewidth=2)
        elif polygon_type == 'building':
            x, y = polygon.exterior.xy
            self.ax.fill(x, y, color=self.colors['building'], alpha=alpha)
        self.fig.canvas.draw()

    def plot_route(self, route):
        """Отрисовывает маршрут оранжевым цветом (без стрелок)"""
        if len(route) > 1:
            # Преобразуем список точек в отдельные списки координат
            x_coords, y_coords = zip(*route)

            # Рисуем плавную линию маршрута
            self.ax.plot(x_coords, y_coords,
                         color=self.colors['route'],  # Оранжевый цвет
                         linewidth=0.5,  # Толщина линии
                         linestyle='-',  # Сплошная линия
                         marker='',  # Без маркеров
                         alpha=0.8,  # Легкая прозрачность
                         label='route')

            # Подчёркиваем точки центров
            for point in route:
                self.ax.plot(point[0], point[1],
                             'o',  # Кружок
                             color=self.colors['route'],
                             markersize=4,  # Размер точки
                             alpha=0.5)  # Полупрозрачность

            self.fig.canvas.draw()

    def plot_network(self, network, centers):
        """Визуализирует сеть соединений"""
        # Очищаем только маршруты и соединения
        self.clear_plots(preserve=['map', 'buildings', 'region'])

        # Рисуем соединения (серые пунктирные линии)
        for u, v, data in network.edges(data=True):
            if 'path' in data:
                x, y = zip(*data['path'])
                self.ax.plot(x, y, color=self.colors['connection'],
                             linestyle='--', alpha=0.7)
            else:
                x = [network.nodes[u]['pos'][0], network.nodes[v]['pos'][0]]
                y = [network.nodes[u]['pos'][1], network.nodes[v]['pos'][1]]
                self.ax.plot(x, y, color=self.colors['connection'],
                             linestyle='--', alpha=0.7)

        # Рисуем центры (темно-зеленые кружки с номерами)
        for i, pos in enumerate(centers):
            self.ax.plot(pos[0], pos[1], 'o', color=self.colors['center'],
                         markersize=8)
            self.ax.text(pos[0], pos[1], str(i + 1),
                         ha='center', va='center', color='white')

        self.fig.canvas.draw()

    def highlight_centers(self, centers, visited_indices):
        """Подсвечивает центры в зависимости от посещения"""
        for i, center in enumerate(centers):
            color = self.colors['visited'] if i in visited_indices else self.colors['missed']
            self.ax.plot(center[0], center[1], 'o', color=color, markersize=10)
            self.ax.text(center[0], center[1], str(i + 1),
                         ha='center', va='center', color='white', fontweight='bold')
        self.fig.canvas.draw()

    def clear_plots(self, preserve=None):
        """Очищает график с возможностью сохранения определенных элементов"""
        preserve = preserve or []

        if 'map' not in preserve and hasattr(self, 'image') and self.image is not None:
            self.ax.imshow(self.image, extent=[self.min_lon, self.max_lon,
                                               self.min_lat, self.max_lat])

        # Удаляем только линии и маркеры маршрута
        for artist in self.ax.lines + self.ax.collections:
            if hasattr(artist, '_label'):
                if artist._label == 'route' and 'route' not in preserve:
                    artist.remove()
            elif isinstance(artist, plt.Line2D):
                if artist.get_color() in [self.colors['route'],
                                          self.colors['connection']]:
                    artist.remove()

        self.fig.canvas.draw()

    def clear_all(self):
        """Полная очистка графика"""
        self.ax.clear()
        if hasattr(self, 'image') and self.image is not None:
            self.ax.imshow(self.image, extent=[self.min_lon, self.max_lon,
                                               self.min_lat, self.max_lat])
        self.ax.set_xlim(self.min_lon, self.max_lon)
        self.ax.set_ylim(self.min_lat, self.max_lat)
        self.fig.canvas.draw()