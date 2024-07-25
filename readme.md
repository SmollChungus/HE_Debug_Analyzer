Tool to analyze qmk console HID listen stuff

to run this, run ```qmk console``` to generate sensor data readings, copy this output to the sensor_data.txt.

to analyze, open a command prompt in this folder and run ```python main.py```

ehh install dependencies with ```pip install numpy pandas matplotlib```


generate .exe
pyinstaller --onefile --add-data "sensor_data.txt;." --hidden-import numpy --hidden-import matplotlib --hidden-import pandas main.py
