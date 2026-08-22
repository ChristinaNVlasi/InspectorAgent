#!/bin/bash

# BORG RAG-Based Component Inspection System
# Optimized Startup Script

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║      BORG - RAG Component Inspection System v2.0          ║"
echo "║      Optimized with Similarity Search (RAG)               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check Python
echo -e "${YELLOW}Checking Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"

# Create/activate venv
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ Created venv${NC}"
fi

echo -e "${YELLOW}Activating venv...${NC}"
source venv/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
fi

# Check database
DB_PATH="data/rag_databases.pkl"
if [ ! -f "$DB_PATH" ]; then
    echo -e "${YELLOW}⚠️  RAG databases not found${NC}"
    echo -e "${CYAN}Building databases from your images...${NC}"
    python3 test_rag_system.py
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Databases built${NC}"
    else
        echo -e "${RED}⚠️  Database building had issues${NC}"
        echo -e "${YELLOW}The app will build them automatically on first launch${NC}"
    fi
fi

# Launch menu
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}Launch Options:${NC}"
echo -e "${GREEN}1)${NC} 🚀 Start RAG Inspection UI (Recommended)"
echo -e "${GREEN}2)${NC} 🧪 Run Tests"
echo -e "${GREEN}3)${NC} 📊 Show System Info"
echo -e "${GREEN}4)${NC} 🔄 Rebuild Databases"
echo -e "${GREEN}5)${NC} ❌ Exit"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""
read -p "$(echo -e ${CYAN}Select option [1-5]: ${NC})" option

case $option in
    1)
        echo -e "${GREEN}🚀 Starting RAG Inspection UI...${NC}"
        echo -e "${YELLOW}Opening in browser...${NC}"
        echo -e "${BLUE}Press Ctrl+C to stop${NC}"
        echo ""
        streamlit run web_ui/streamlit_app_rag.py
        ;;
    2)
        echo -e "${GREEN}🧪 Running tests...${NC}"
        python3 test_rag_system.py
        ;;
    3)
        echo -e "${CYAN}📊 System Information:${NC}"
        echo ""
        echo -e "${GREEN}Python:${NC} $(python3 --version)"
        echo -e "${GREEN}Working Directory:${NC} $SCRIPT_DIR"
        echo ""
        
        if [ -f "$DB_PATH" ]; then
            echo -e "${GREEN}✓ RAG Databases:${NC} Found at $DB_PATH"
            echo -e "  Size: $(ls -lh $DB_PATH | awk '{print $5}')"
        else
            echo -e "${YELLOW}⚠️  RAG Databases:${NC} Not found"
        fi
        echo ""
        
        echo -e "${GREEN}Reference Images:${NC}"
        for dir in "A1 - Pulley" "B - Broken Casting" "C - Broken Cover"; do
            count=$(find "../parts_images/$dir" -name "*.Jpeg" 2>/dev/null | wc -l)
            echo -e "  • $dir: $count images"
        done
        echo ""
        ;;
    4)
        echo -e "${YELLOW}🔄 Rebuilding RAG databases...${NC}"
        rm -f "$DB_PATH"
        python3 test_rag_system.py
        echo -e "${GREEN}✓ Databases rebuilt${NC}"
        ;;
    5)
        echo -e "${BLUE}Goodbye!${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid option${NC}"
        exit 1
        ;;
esac

deactivate
