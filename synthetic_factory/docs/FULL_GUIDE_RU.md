# Synthetic Factory: Полное руководство (Windows + Linux)

Этот документ — практический runbook по всему циклу:
- генерация синтетического завода (Scene -> OBJ),
- генерация LiDAR-облаков (круги + сшитые PLY),
- настройка через внешние конфиги,
- запуск на Windows/Linux,
- воспроизводимость экспериментов для статьи.

Документ актуален для запуска через `uv` и CLI-модуль `synthetic_factory`.

---

## 1. Что находится в репозитории

Основной проект:
- `src/synthetic_factory/` — генерация сцены, инфраструктуры, экстерьера, LiDAR.
- `configs/presets/` — готовые пресеты.
- `scripts/` — готовые скрипты запуска.
- `docs/` — документация и описание алгоритмов.
- `out/` — результаты генерации.

Отдельный сканер:
- `pointcloud_scanner/` — автономный пакет OBJ -> point cloud.

---

## 2. Минимальные требования

Обязательно:
- Python `3.11+`
- `uv`
- Git

Для тяжелых прогонов:
- 32-64+ GB RAM
- много ядер CPU
- NVIDIA GPU (опционально, для CUDA-ускорения LiDAR)
- достаточно места на диске (PLY может быть очень большим)

---

## 3. Быстрый старт за 5 минут

### Windows (PowerShell)

```powershell
cd C:\_Work\_Factory\synthetic_factory
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1
```

### Linux

```bash
cd /path/to/synthetic_factory
uv sync --all-extras
uv run -m synthetic_factory --config configs/presets/scene.yaml --seed 42
```

---

## 4. Установка и окружение

### 4.1 Windows

```powershell
cd C:\_Work\_Factory\synthetic_factory
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
uv sync --all-extras
```

### 4.2 Linux

Установка `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Проверка:

```bash
uv --version
```

Синхронизация окружения:

```bash
cd /path/to/synthetic_factory
uv sync --all-extras
```

---

## 5. Как устроен pipeline

Пайплайн (в типичном порядке):
1. `ParameterSampling` — вычисляет итоговые параметры (включая выражения/диапазоны).
2. `LayoutGeneration` — строит план помещений.
3. `RoomGeneration` — наполняет комнаты по биомам.
4. `SceneAssembly` — собирает сцену.
5. `Infrastructure` — глобальные кабельные/трубные сети.
6. `Auxiliary` — вторичные детали.
7. `Export` — экспорт OBJ.
8. `LidarStationPlanning` — поиск точек стояния.
9. `LidarPointCloud` — генерация кругов и сшитых облаков.

Типичный лог:
- `[Pipeline] started with seed=...`
- `[ParameterSampling] sampled={...}`
- `[Export] path=...`
- `[LidarStationPlanning] stations=...`

Это диагностические этапы, не ошибки.

---

## 6. Конфиги и приоритет параметров (очень важно)

Источник истины для запуска:
- `--config <файл>`
- плюс `--set path=value` и shortcut-аргументы CLI.

Порядок применения:
1. Загружается preset.
2. Накладываются CLI override (`--set`, `--factory-width`, и т.д.).
3. Формируется `ParameterSet` из секции `parameters` (если она есть).

Ключевой нюанс:
- если в пресете есть `parameters.factory_width` и `parameters.factory_depth`, именно они попадут в `ParameterSampling`.
- просто изменить `factory.width`/`factory.depth` может быть недостаточно.

Поэтому для гарантии меняйте оба блока:
- `factory.width`, `factory.depth`
- `parameters.factory_width.*`, `parameters.factory_depth.*`

---

## 7. Основные пресеты

- `configs/presets/scene.yaml` — базовый сбалансированный.
- `configs/presets/scene_essential.yaml` — минимальный интерфейс ключевых параметров.
- `configs/presets/scene_lidar_working.yaml` — быстрые итерации LiDAR.
- `configs/presets/scene_lidar_ultra_dense.yaml` — очень плотный, тяжелый.
- `configs/presets/scene_lidar_portable_max.yaml` — переносимый/безопасный.
- `configs/presets/scene_lidar_full_factory_4x.json` — тяжелый full-factory preset.

---

## 8. Рекомендуемые команды запуска

### 8.1 Базовый запуск

```bash
uv run -m synthetic_factory --config configs/presets/scene.yaml --seed 42
```

### 8.2 Полный тяжелый запуск (Linux)

```bash
bash ./scripts/run_lidar_full_factory_4x.sh
```

Или напрямую:

```bash
uv run -m synthetic_factory --config configs/presets/scene_lidar_full_factory_4x.json --seed 42
```

### 8.3 Запуск с override без редактирования файла

```bash
uv run -m synthetic_factory \
  --config configs/presets/scene_lidar_full_factory_4x.json \
  --seed 42 \
  --set factory.width=110 \
  --set factory.depth=85 \
  --set parameters.factory_width.value=110 \
  --set parameters.factory_width.min=110 \
  --set parameters.factory_width.max=110 \
  --set parameters.factory_depth.value=85 \
  --set parameters.factory_depth.min=85 \
  --set parameters.factory_depth.max=85
```

---

## 9. LiDAR: ключевые параметры

`lidar.scan_range`
- дальность луча.

`lidar.angular_resolution_deg`, `lidar.vertical_resolution_deg`
- шаги углов; меньше шаг -> больше точек.

`lidar.point_multiplier`
- масштаб плотности облака.

`lidar.points_per_station`
- лимит точек на станцию (`0` = без лимита).

`lidar.station_spacing`
- расстояние между кандидатами станций.

`lidar.max_stations_per_room`, `lidar.max_stations`
- лимиты станций.
- `0` = без лимита.

`lidar.global_coverage`
- глобальное покрытие всей сцены.

`lidar.exterior_point_density_factor`
- относительная плотность наружных точек (например, `0.25` = в 4 раза реже, чем внутри).

---

## 10. NVIDIA GPU / CUDA для LiDAR

Поддерживается backend:
- `lidar.compute_backend: auto | cpu | cuda`

Поведение:
- `auto` — использовать CUDA при наличии, иначе CPU.
- `cpu` — всегда CPU.
- `cuda` — требовать CUDA, иначе ошибка.

Дополнительные настройки:
- `lidar.cuda_ray_batch_size`
- `lidar.cuda_object_batch_size`

Пример:

```bash
uv run -m synthetic_factory \
  --config configs/presets/scene_lidar_full_factory_4x.json \
  --seed 42 \
  --lidar-compute-backend cuda
```

Важно:
- нужен установленный `torch` с CUDA-поддержкой в текущем окружении.
- при runtime-проблеме pipeline автоматически падает обратно на CPU (в режиме `auto`).

---

## 11. Куда сохраняются результаты

OBJ:
- путь из `export.output_path` (обычно `out/...obj`).

LiDAR:
- структура в директории из `lidar.output_dir`, с подпапками по дате запуска:
  - `<date>/circles/` — отдельные круги (каждая станция отдельный `.ply`)
  - `<date>/interior/factory_stitched_interior.ply`
  - `<date>/exterior/factory_stitched_exterior.ply`
  - `<date>/combined/factory_stitched.ply`
  - metadata JSON

Это позволяет не перезаписывать предыдущие прогоны.

---

## 12. Как читать и проверять лог

Пример:
- `[ParameterSampling] sampled={'factory_width': 220.0, ...}`

Если ожидали `110`, а видите `220`:
- вы запускаете не тот preset, или
- в `parameters` остались старые значения.

Проверка:
```bash
grep -n '"factory_width"\|"factory_depth"\|"width"\|"depth"' configs/presets/scene_lidar_full_factory_4x.json
```

---

## 13. Чек-лист перед тяжелым запуском

1. `git pull` в актуальной ветке.
2. Проверить нужный `--config`.
3. Проверить `parameters.factory_width/depth`.
4. Зафиксировать `--seed`.
5. Проверить лимиты:
   - `max_stations*`
   - `points_per_station`
6. Для GPU:
   - `compute_backend=auto/cuda`
   - CUDA-совместимый `torch`.
7. Убедиться, что хватает RAM/диска.

---

## 14. Частые проблемы и решения

### `ModuleNotFoundError: setuptools`
- Не запускайте `py setup.py`.
- Используйте `uv sync` + `uv run ...`.

### “Скрипт не найден”
- запускайте из корня `synthetic_factory/`.

### “Размер завода не меняется”
- проверьте блок `parameters` (см. раздел 6).

### Слишком долго / OOM
- уменьшить:
  - `lidar.point_multiplier`
  - `lidar.points_per_station`
  - `number_of_rooms`
- увеличить `station_spacing`.
- взять `scene_lidar_portable_max.yaml`.

### `stations` больше ожидаемого
- это информативный вывод планировщика.
- ограничивайте через `max_stations*` или оставляйте `0` для безлимита.

### CUDA не включилась
- проверьте `lidar.compute_backend`.
- проверьте, что `torch` собран с CUDA.
- в `auto` pipeline может корректно уйти на CPU fallback.

---

## 15. Репродуцируемые эксперименты для статьи

Рекомендуется:
- фиксировать `seed`,
- фиксировать commit hash,
- сохранять полный preset рядом с артефактами,
- использовать неизменяемые директории результатов по дате.

Минимальный набор артефактов:
- `scene.obj`
- папка кругов `circles/*.ply`
- `factory_stitched_interior.ply`
- `factory_stitched_exterior.ply`
- `factory_stitched.ply`
- `metadata.json`

---

## 16. Полезные команды (шпаргалка)

Показать все CLI-опции:

```bash
uv run -m synthetic_factory --help
```

Быстрый тест:

```bash
uv run -m synthetic_factory --config configs/presets/scene_lidar_test_small.yaml --seed 42
```

Тяжелый запуск на сильной машине (без лимита станций):

```bash
uv run -m synthetic_factory \
  --config configs/presets/scene_lidar_full_factory_4x.json \
  --seed 42 \
  --set lidar.max_stations_per_room=0 \
  --set lidar.max_stations=0
```

---

## 17. Связанные документы

- Концепция/алгоритмы: `docs/PROJECT_CONCEPT_AND_ALGORITHMS_RU.md`
- Пояснения по конфигам: `configs/README.md`
- Отдельный сканер: `pointcloud_scanner/README.md`

