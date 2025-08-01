#!/bin/bash

GREEN="\033[1;32m"
BOLD="\033[1m"
RESET="\033[0m"
CYAN="\033[1;36m"
RED="\033[1;31m"
BOX="=================================================="


echo -e "${RED}${BOLD}$BOX${RESET}"
echo -e "${CYAN}${BOLD}                I USE ARCH BTW${RESET}"
echo -e "${RED}${BOLD}$BOX${RESET}"


if command -v python &> /dev/null; then
  echo -e "Python found: $(python --version)"
else
  echo -e "Python isn't installad"  
  exit
fi


if command -v poetry &> /dev/null; then
  echo -e "Poetry found: $(poetry --version)"
else
  echo -e "Poetry not found. Installing Poetry..."
  pip install poetry
  echo -e "Poetry installed!"
fi

echo -e "\n"
echo -e "${RED}${BOLD}Running${RESET}\n"
python scripts/data_scripts.py -ci

python scripts/train_scripts.py

python scripts/predict_scripts.py