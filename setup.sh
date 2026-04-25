#!/bin/bash
# VeriFace AI — One-Command Setup Script

set -e
echo "🛡️  Setting up VeriFace AI..."

# Create virtualenv
python3 -m venv venv
source venv/bin/activate

# Install deps
pip install --upgrade pip
pip install -r requirements.txt

# Copy env template
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Created .env from template"
fi

# Create directories
mkdir -p models data/train_real data/train_fake logs outputs

echo ""
echo "🚀 Setup complete!"
echo ""
echo "To launch VeriFace AI:"
echo "  source venv/bin/activate"
echo "  streamlit run ui/main_app.py"
echo ""
echo "To launch the API:"
echo "  uvicorn backend.main:app --reload --port 8000"
echo ""
echo "To launch with Docker:"
echo "  docker compose -f docker/docker-compose.yml up --build"
