conda activate pennington_kg

# Standard monthly update (auto-detects since last run)
python update_periodic_kg.py

# Pull last 60 days
python update_periodic_kg.py 60

# Pull from a specific date
python update_periodic_kg.py --since 2026-05-01

# See what would run without actually running it
python update_periodic_kg.py --dry-run

# View history of past updates
python update_periodic_kg.py --history