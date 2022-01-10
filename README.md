# Isoparametric elements visualization
Проект для визуализации преобразования координат из нормированных в декартовые для изопараметрического гексаэдрального элемента с 8 и 20 узлами (app_20_nodes.py и app_8_nodes.py соответственно).
**Запуск приложения**
* Для работы приложения необходим **python 3.8.5** и выше. Ссылка для скачивания: https://www.python.org/downloads/
* После установки python следует установить необходимые зависимости с помощью команды: **pip install -r requirements.txt**
* Запуск происходит с помощью команды: **python app_20_nodes.py**
* Далее приложение запуститься на **http://127.0.0.1:8050/**. После это скопируйте данную ссылку в браузер и вы увидете визуализацию:
![alt text](/figures/fig1.png "Визуализация")

**Параметры приложения**
Все параметры находятся в начале каждого файла скрипта (app_20_nodes.py и app_8_nodes.py)
![alt text](/figures/fig2.png "Параметры")
В данном разделе можно задать координаты узлов гексаэдрального элемента в декартовых координатах. Координаты необходить задать в порядке нумерации узлов. Нумерация узлов приведена на следующем рисунке:
![alt text](/figures/fig3.png "Нумерация узлов")
Также есть возможность задать порт для приложения и режим работы. При режиме **debag=True** изменения визуализации в браузере происходят в реальном времени при изменении скрипта.