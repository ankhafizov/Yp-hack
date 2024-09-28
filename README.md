# Yappy-duplicates-hack
Проект по поиску дубликатов видео для сервиса Yappy

# Как запустить:

1. Скачать https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/MVIT_B_32x3_f294077834.pyth в ./torchvideo_cache/checkpoints

2. Поднять докер контейнеры
```
docker-compose -p video_duplicate_search up -d --build
```

Все настройки можно указать в файле configs/app_config.yaml

# Как тестировать

1. Простой тест написан в файле example_test.py

2. Тест через сваггер http://127.0.0.1:8000/docs
