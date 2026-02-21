Debug model(YOLOv11) guide

=====Linux(Bash)=====

python=3.10.14

python -m venv .venv

source .venv/bin/activate.fish

pip install -r requirements

python test.py --predict /path/to/your/car_plate.jpg --show

=====Window(cmd)=====
python=3.10.14

python -m venv .venv

.venv\Scripts\activate

python test.py --predict C:\path\to\your\car_plate.jpg --show
