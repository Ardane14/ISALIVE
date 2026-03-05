@echo off

REM Nom du dossier de l'environnement virtuel
set ENV_DIR=venv

REM 1. Vérifie si le dossier venv existe, sinon le crée
if not exist %ENV_DIR%\ (
    echo Creation de l'environnement virtuel...
    python -m venv %ENV_DIR%
)

REM 2. Active l'environnement virtuel
echo Activation de l'environnement...
call %ENV_DIR%\Scripts\activate

REM 3. Installe les dependances
echo Installation des librairies...
pip install -r requirements.txt
