#!/bin/bash

# Tworzenie środowiska wirtualnego
if [ ! -d "venv" ]; then
  python -m venv venv
fi


source venv/bin/activate

# Instalacja zależności
pip install --upgrade pip
pip install -r requirements.txt

# Uruchomienie Gunicorna
exec gunicorn -w 4 -b 0.0.0.0:8080 app:app
