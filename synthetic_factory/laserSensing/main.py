from model.geometry_manager import GeometryManager
from view.plot_manager import PlotManager
from controller.event_handler import EventHandler
import matplotlib.pyplot as plt


def main():
    # Параметры геопривязки
    min_x, max_x = 0.0, 160.294
    min_y, max_y = 0.0, 95.134
    # Путь к изображению карты
    image_path = "C:\\Users\\Professional\\Pictures\\3003.png"

    # Инициализация менеджеров
    geometry_manager = GeometryManager()
    plot_manager = PlotManager(min_x, max_x, min_y, max_y, image_path)

    # Настройка интерфейса
    plot_manager.setup_plot()  

    # Затем создаем обработчик событий
    event_handler = EventHandler(geometry_manager, plot_manager)

    plt.show()


if __name__ == "__main__":
    main()