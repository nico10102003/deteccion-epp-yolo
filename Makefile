# =============================================================================
# deteccion-epp-yolo - Automatizacion completa del proyecto
#
# Este Makefile es autonomo: no requiere uv ni ningun archivo de dependencias
# externo. Todo se ejecuta con el interprete de Python estandar. El entorno
# virtual se crea con `python -m venv` (modulo de la libreria estandar) y las
# dependencias se instalan desde la lista DEPS declarada mas abajo, que es la
# unica fuente de verdad de este proyecto.
#
# Uso:
#   make            Equivale a `make help`
#   make help       Lista todos los comandos
#   make install    Crea el entorno e instala las dependencias
#   make doctor     Verifica que las dependencias esten completas
#   make run        Levanta la interfaz Streamlit
#   make serve      Levanta el servidor de inferencia gRPC
#
# Variables configurables:
#   PYTHON   -> interprete usado para crear el entorno  (por defecto: python)
#   COMPOSE  -> cliente de Docker Compose               (por defecto: docker compose)
#   PORT     -> puerto del servidor de documentacion    (por defecto: 8000)
# =============================================================================

PYTHON  ?= python
COMPOSE ?= docker compose
PORT    ?= 8000

# --- Dependencias del proyecto -----------------------------------------------
# Se dejan entrecomilladas porque en Windows el caracter > de una version es un
# operador de redireccion para cmd.exe. Sin comillas, torch>=2.14.0 crearia un
# archivo llamado "=2.14.0" en lugar de instalar el paquete.

# torch y torchvision se resuelven contra el indice oficial de PyTorch en su
# variante CPU. Con este indice, pip elige la build +cpu y se evita descargar
# los cuas de 2.5 GB que trae la build de PyPI en Linux.
PYTORCH_INDEX := https://download.pytorch.org/whl/cpu

DEPS_RUNTIME := \
	"ultralytics>=8.3.0" \
	"torch>=2.14.0" \
	"torchvision>=0.29.0" \
	"grpcio>=1.66.0" \
	"grpcio-tools>=1.66.0" \
	"streamlit>=1.38.0" \
	"streamlit-webrtc>=0.77.0" \
	"mlflow>=2.16.0" \
	"opencv-python-headless>=4.10.0" \
	"numpy>=1.26.4" \
	"pillow>=10.4.0" \
	"pydantic>=2.9.0" \
	"protobuf>=5.27.0"

DEPS_DEV := \
	"pytest>=8.3.0" \
	"pytest-cov>=5.0.0" \
	"pytest-mock>=3.14.0" \
	"ruff>=0.6.0"

DEPS := $(DEPS_RUNTIME) $(DEPS_DEV)

# Modulos que se comprueban con `make doctor`.
MODULOS := torch torchvision ultralytics grpc streamlit mlflow cv2 numpy \
	PIL pydantic pytest ruff grpc_tools

# --- Rutas -------------------------------------------------------------------

VENV_DIR := .venv
STAMP    := $(VENV_DIR)/.dependencias-instaladas

SRC_DIRS := app src tests scripts

PROTO_DIR  := src/grpc_service
PROTO_FILE := $(PROTO_DIR)/inference.proto
PROTO_PY   := $(PROTO_DIR)/inference_pb2.py
PROTO_GRPC := $(PROTO_DIR)/inference_pb2_grpc.py

# El Makefile debe funcionar igual en Windows que en Linux, asi que se detectan
# el interprete del entorno y los comandos de borrado segun el sistema.
ifeq ($(OS),Windows_NT)
VENV_PY := $(VENV_DIR)/Scripts/python.exe
RM_RF   = if exist "$(1)" rmdir /s /q "$(1)"
RM_F    = if exist "$(1)" del /q "$(1)"
else
VENV_PY := $(VENV_DIR)/bin/python
RM_RF   = rm -rf "$(1)"
RM_F    = rm -f "$(1)"
endif

.DEFAULT_GOAL := help

.PHONY: help install deps deps-list doctor purge proto run serve test test-cov \
        lint lint-fix fmt check eval mlflow-log docs up down logs ps build \
        restart clean clean-docker clean-mlflow distclean

# --- Ayuda -------------------------------------------------------------------

help: ## Muestra la lista de comandos
	@echo deteccion-epp-yolo - comandos disponibles
	@echo ---------------------------------------
	@echo make install       Crea el entorno e instala dependencias
	@echo make deps-list     Muestra las dependencias que instala este Makefile
	@echo make doctor        Verifica que las dependencias esten completas
	@echo make proto         Genera los stubs gRPC desde inference.proto
	@echo make run           Levanta la interfaz Streamlit
	@echo make serve         Levanta el servidor de inferencia gRPC
	@echo make test          Ejecuta la suite de pruebas
	@echo make test-cov      Ejecuta pruebas con reporte HTML de cobertura
	@echo make lint          Verifica estilo con Ruff
	@echo make lint-fix      Verifica estilo y corrige lo corregible
	@echo make fmt           Reformatea el codigo con Ruff
	@echo make check         Verificacion completa: lint + test
	@echo make eval          Evalua el modelo y escribe el reporte
	@echo make mlflow-log    Envia el reporte de evaluacion a MLflow
	@echo make docs          Sirve docs/ en http://localhost:8000
	@echo make up            Levanta Streamlit + inferencia + MLflow
	@echo make down          Detiene los contenedores
	@echo make logs          Sigue los logs del servicio de inferencia
	@echo make ps            Estado de los contenedores
	@echo make build         Reconstruye las imagenes Docker
	@echo make restart       Reinicia los contenedores
	@echo make clean         Elimina caches de pruebas y lint
	@echo make distclean     Limpieza completa, incluido el entorno virtual

# --- Entorno -----------------------------------------------------------------
# El grafo de dependencias hace el trabajo: todo target que necesita Python
# depende de $(STAMP), y make solo lo construye si falta o si cambian las
# dependencias. Por eso "si falta una dependencia, se instala" de forma
# automatica, sin repetir la instalacion en cada ejecucion.

install: $(STAMP) ## Crea el entorno e instala dependencias

deps: install ## Alias de install

deps-list: ## Muestra la lista de dependencias que instala este Makefile
	@echo Dependencias de ejecucion:
	@echo $(DEPS_RUNTIME)
	@echo Dependencias de desarrollo:
	@echo $(DEPS_DEV)

$(VENV_PY):
	@echo [1/2] Creando el entorno virtual en $(VENV_DIR)
	$(PYTHON) -m venv $(VENV_DIR)

# El Makefile es pre-requisito del sello: al editar la lista DEPS la
# instalacion se vuelve a ejecutar de forma automatica.
$(STAMP): $(VENV_PY) Makefile
	@echo [2/2] Instalando dependencias declaradas en este Makefile
	$(VENV_PY) -m ensurepip --upgrade
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install --extra-index-url $(PYTORCH_INDEX) $(DEPS)
	@echo instaladas > $(STAMP)
	@echo Dependencias instaladas. Ejecuta `make doctor` para verificar.

doctor: $(STAMP) ## Verifica que todas las dependencias esten disponibles
	@echo Si falta alguna dependencia, corrige con: make install
	$(VENV_PY) -c "import importlib.util as u, sys; mods = '$(MODULOS)'.split(); bad = [m for m in mods if u.find_spec(m) is None]; print('FALTAN: ' + ', '.join(bad)) if bad else print('Todas las dependencias estan disponibles'); sys.exit(1 if bad else 0)"

purge: ## Elimina el entorno virtual
	$(call RM_RF,$(VENV_DIR))
	@echo Entorno virtual eliminado

# --- gRPC --------------------------------------------------------------------

proto: $(STAMP) ## Genera los stubs gRPC desde inference.proto
	$(VENV_PY) -m grpc_tools.protoc -I$(PROTO_DIR) --python_out=$(PROTO_DIR) --grpc_python_out=$(PROTO_DIR) $(PROTO_FILE)
	$(VENV_PY) -c "import pathlib; p = pathlib.Path(r'$(PROTO_GRPC)'); s = p.read_text(encoding='utf-8'); p.write_text(s.replace('import inference_pb2 as inference__pb2', 'from src.grpc_service import inference_pb2 as inference__pb2'), encoding='utf-8')"
	@echo Stubs generados en $(PROTO_PY) y $(PROTO_GRPC)

# --- Ejecucion ---------------------------------------------------------------

run: $(STAMP) $(PROTO_PY) ## Levanta la interfaz Streamlit
	$(VENV_PY) -m streamlit run app/detector_epp.py --server.port=8501

serve: $(STAMP) $(PROTO_PY) ## Levanta el servidor de inferencia gRPC
	$(VENV_PY) -m src.grpc_service.server

$(PROTO_PY): $(PROTO_FILE) $(STAMP)
	$(MAKE) proto

# --- Calidad -----------------------------------------------------------------

test: $(STAMP) $(PROTO_PY) ## Ejecuta la suite de pruebas
	$(VENV_PY) -m pytest

test-cov: $(STAMP) $(PROTO_PY) ## Ejecuta pruebas con reporte HTML de cobertura
	$(VENV_PY) -m pytest --cov=src --cov-report=html --cov-report=term-missing

lint: $(STAMP) ## Verifica estilo con Ruff
	$(VENV_PY) -m ruff check $(SRC_DIRS)

lint-fix: $(STAMP) ## Verifica estilo y corrige lo corregible
	$(VENV_PY) -m ruff check --fix $(SRC_DIRS)

fmt: $(STAMP) ## Reformatea el codigo con Ruff
	$(VENV_PY) -m ruff format $(SRC_DIRS)

check: lint test ## Verificacion completa: lint + test

# --- Modelo y experimentos ----------------------------------------------------

eval: $(STAMP) ## Evalua el modelo y genera el reporte
	$(VENV_PY) scripts/evaluate_model.py

mlflow-log: $(STAMP) ## Registra el reporte de evaluacion en MLflow
	$(VENV_PY) scripts/log_evaluation_to_mlflow.py

# --- Documentacion ------------------------------------------------------------

docs: $(STAMP) ## Sirve docs/ en http://localhost:8000
	$(VENV_PY) -m http.server $(PORT) --directory docs

# --- Docker ------------------------------------------------------------------

up: $(STAMP) ## Levanta Streamlit + inferencia + MLflow
	$(COMPOSE) up -d

down: ## Detiene los contenedores
	$(COMPOSE) down

logs: ## Sigue los logs del servicio de inferencia
	$(COMPOSE) logs -f inference

ps: ## Estado de los contenedores
	$(COMPOSE) ps

build: ## Reconstruye las imagenes Docker
	$(COMPOSE) build

restart: down up ## Reinicia los contenedores

# --- Limpieza ----------------------------------------------------------------

clean: ## Elimina caches de pruebas y lint
	$(call RM_RF,.pytest_cache)
	$(call RM_RF,.ruff_cache)
	$(call RM_RF,htmlcov)
	$(call RM_RF,.mypy_cache)
	$(call RM_F,.coverage)
	@echo Caches eliminados

clean-docker: ## Detiene contenedores y elimina volumenes
	$(COMPOSE) down --volumes --remove-orphans

clean-mlflow: ## Elimina los datos locales de MLflow
	$(call RM_RF,mlruns)
	$(call RM_RF,mlartifacts)

distclean: clean clean-docker clean-mlflow purge ## Limpieza completa