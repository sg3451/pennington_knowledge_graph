@echo off
echo ================================================
echo  Pennington KG Explorer - Setup Script
echo ================================================
echo.

REM Check Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker Desktop is not running.
    echo Please start Docker Desktop and run this script again.
    pause
    exit /b 1
)
echo [OK] Docker Desktop is running

REM Start Neo4j
echo.
echo Starting Neo4j database...
docker compose up -d neo4j
echo Waiting 45 seconds for Neo4j to initialize...
timeout /t 45 /nobreak >nul

REM Install Python packages
echo.
echo Installing Python packages...
pip install --quiet streamlit plotly pandas neo4j python-dotenv
echo [OK] Packages installed

REM Load the graph
echo.
echo Loading knowledge graph (this takes about 45 minutes)...
echo You can open http://localhost:8501 now - data will appear as it loads.
python 04_load_neo4j.py

REM Start Streamlit
echo.
echo Starting web application...
start "" http://localhost:8501
streamlit run 05_streamlit.py

pause