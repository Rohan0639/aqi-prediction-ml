@echo off
echo Pulling latest data...
git pull origin main

echo Adding new data...
git add .

echo Committing changes...
git commit -m "Auto sync AQI data"

echo Pushing to GitHub...
git push origin main

echo Done!
pause