#!/bin/bash
# deploy.sh — One-command deployment for AI Agent Playground
#
# Usage:
#   ./deploy.sh start         # Start all services
#   ./deploy.sh stop          # Stop all services
#   ./deploy.sh restart       # Restart all services
#   ./deploy.sh status        # Check health
#   ./deploy.sh logs          # Tail logs
#   ./deploy.sh update        # Pull latest code and restart
#   ./deploy.sh full-start    # Start with Ollama + ChromaDB
#   ./deploy.sh setup         # First-time setup
#   ./deploy.sh pentest       # Run security test

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[deploy]${NC} $1"; }
warn() { echo -e "${YELLOW}[warn]${NC} $1"; }
err() { echo -e "${RED}[error]${NC} $1"; }

check_deps() {
    command -v docker >/dev/null 2>&1 || { err "Docker not installed"; exit 1; }
    command -v docker-compose >/dev/null 2>&1 || command -v docker compose >/dev/null 2>&1 || \
        { err "Docker Compose not installed"; exit 1; }
}

check_env() {
    if [ ! -f .env ]; then
        warn ".env not found — creating from template"
        cp .env.example .env
        warn "EDIT .env BEFORE STARTING: add DEEPSEEK_API_KEY and generate security keys"
        exit 1
    fi

    # Validate required vars
    source <(grep -v '^#' .env | grep -v '^$' | sed 's/^/export /' 2>/dev/null || true)
    if [ -z "$DEEPSEEK_API_KEY" ] || [ "$DEEPSEEK_API_KEY" = "sk-your-deepseek-key-here" ]; then
        err "DEEPSEEK_API_KEY not configured in .env"
        exit 1
    fi
}

generate_keys() {
    log "Generating security keys..."
    if command -v openssl >/dev/null 2>&1; then
        GATEWAY_KEY=$(openssl rand -hex 32)
        SIGNING_KEY=$(openssl rand -hex 32)
        SESSION_KEY=$(openssl rand -hex 32)
    else
        GATEWAY_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
        SIGNING_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
        SESSION_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    fi
    log "Gateway API Key: $GATEWAY_KEY"
    log "Add these to your .env file"
}

docker_cmd() {
    if command -v docker compose >/dev/null 2>&1; then
        docker compose "$@"
    else
        docker-compose "$@"
    fi
}

case "${1:-start}" in
    setup)
        log "=== First-Time Setup ==="
        check_deps
        if [ ! -f .env ]; then
            cp .env.example .env
            log "Created .env — edit it now:"
            log "  nano .env"
            log "Add at minimum: DEEPSEEK_API_KEY"
        fi
        log "Building Docker image..."
        docker_cmd build
        log "Setup complete. Run: ./deploy.sh start"
        ;;

    start)
        log "Starting services..."
        check_deps
        check_env
        docker_cmd up -d
        sleep 3
        log "Checking health..."
        for i in 1 2 3 4 5; do
            if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
                log "Agent is healthy!"
                break
            fi
            sleep 2
        done
        log "Running quick verification..."
        curl -s http://localhost:8000/health | head -c 200
        echo ""
        ;;

    full-start)
        log "Starting full stack (agent + Ollama + ChromaDB)..."
        check_deps
        check_env
        docker_cmd --profile full up -d
        sleep 5
        log "Full stack starting. Agent: http://localhost:8000"
        ;;

    stop)
        log "Stopping services..."
        docker_cmd down
        log "Stopped."
        ;;

    restart)
        log "Restarting..."
        docker_cmd down
        docker_cmd up -d
        log "Restarted."
        ;;

    status)
        log "=== System Status ==="
        if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
            curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null || \
            curl -s http://localhost:8000/health
        else
            err "Agent not responding on port 8000"
        fi
        echo ""
        log "=== Container Status ==="
        docker_cmd ps 2>/dev/null || docker ps
        ;;

    logs)
        docker_cmd logs -f --tail=50 agent 2>/dev/null || \
        docker logs -f --tail=50 ai-agent-gateway
        ;;

    update)
        log "Pulling latest code..."
        git pull origin master
        log "Rebuilding..."
        docker_cmd build --no-cache
        log "Restarting..."
        docker_cmd down
        docker_cmd up -d
        log "Updated and restarted."
        ;;

    pentest)
        log "Running security penetration test..."
        if command -v uv >/dev/null 2>&1; then
            uv run python scripts/pentest.py
        else
            python3 scripts/pentest.py
        fi
        ;;

    bench)
        log "Running engine benchmarks..."
        if command -v uv >/dev/null 2>&1; then
            uv run python scripts/benchmark_engines.py
            uv run python scripts/hard_benchmark.py
        else
            python3 scripts/benchmark_engines.py
            python3 scripts/hard_benchmark.py
        fi
        ;;

    keys)
        generate_keys
        ;;

    *)
        echo "Usage: ./deploy.sh {start|stop|restart|status|logs|update|full-start|setup|pentest|bench|keys}"
        echo ""
        echo "  start       Start agent (Docker)"
        echo "  stop        Stop all services"
        echo "  restart     Restart all services"
        echo "  status      Show health and container status"
        echo "  logs        Tail agent logs"
        echo "  update      Pull latest git + rebuild + restart"
        echo "  full-start  Start with Ollama + ChromaDB"
        echo "  setup       First-time setup"
        echo "  pentest     Run security penetration test"
        echo "  bench       Run engine benchmarks"
        echo "  keys        Generate random security keys"
        exit 1
        ;;
esac
