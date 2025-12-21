SHELL := /bin/bash
PY ?= python3
PY_BIN := $(if $(wildcard .venv/bin/python),.venv/bin/python,$(PY))
CLIENT_BIN := bin/client.py
DEFAULT_UPDATE_URL := https://yagodka.org:17778
DEFAULT_SERVER := yagodka.org:7777
DEFAULT_PUBKEY := 2c832025b325c453270b8bd90c3ccf58cdf23077fa287f2cba0b1bdc215b51f5
DEFAULT_DEPS := cryptography
REQUIRED_DIRS := bin modules var
CHECK_SCRIPT := scripts/check_structure.py
UPDATE_SCRIPT := scripts/manual_update.py

.PHONY: help run run-debug update ensure-structure ensure-client check-structure venv deps clean
.PHONY: test

help:
	@echo "make run [SERVER=host:port UPDATE_URL=...]   # скачать dist при необходимости и запустить клиента"
	@echo "make run-debug                              # то же, но с DEBUG логами"
	@echo "make update UPDATE_URL=...                  # скачать client.py из UPDATE_URL"
	@echo "make deps                                   # установить зависимости в .venv (PyPI)"

venv:
	@if [ ! -d ".venv" ]; then \
		echo "[client] creating virtualenv .venv"; \
		$(PY) -m venv .venv; \
	fi

deps: venv
	@. .venv/bin/activate && PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_PROGRESS_BAR=off PYTHONWARNINGS=ignore pip install -q -U pip >/dev/null
	@. .venv/bin/activate && PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_PROGRESS_BAR=off PYTHONWARNINGS=ignore pip install -q $(strip $(DEFAULT_DEPS) $(DEPS)) >/dev/null

run: deps ensure-structure ensure-client
	@SERVER_URL="$(or $(SERVER),$(DEFAULT_SERVER))"; \
	UPDATE_URL_USE="$(or $(UPDATE_URL),$(DEFAULT_UPDATE_URL))"; \
	if [ ! -t 0 ] || [ ! -t 1 ]; then \
		if command -v script >/dev/null 2>&1; then \
			exec script -q /dev/null -- $(MAKE) _run_inner SERVER="$$SERVER_URL" UPDATE_URL="$$UPDATE_URL_USE" UPDATE_PUBKEY="$(or $(UPDATE_PUBKEY),$(DEFAULT_PUBKEY))"; \
		else \
			echo "[client] stdin/stdout не TTY; откройте обычный терминал или установите util-linux 'script'"; \
			exit 2; \
		fi; \
	fi; \
	$(MAKE) _run_inner SERVER="$$SERVER_URL" UPDATE_URL="$$UPDATE_URL_USE" UPDATE_PUBKEY="$(or $(UPDATE_PUBKEY),$(DEFAULT_PUBKEY))"

.PHONY: _run_inner
_run_inner:
	@SERVER_URL="$(or $(SERVER),$(DEFAULT_SERVER))"; \
	UPDATE_URL_USE="$(or $(UPDATE_URL),$(DEFAULT_UPDATE_URL))"; \
	PKEY="$(or $(UPDATE_PUBKEY),$(DEFAULT_PUBKEY))"; \
	trap "stty sane; tput cnorm 2>/dev/null || true" EXIT; \
	$(PY_BIN) $(CHECK_SCRIPT) || { echo '[client] структура нарушена, исправьте и повторите'; exit 2; }; \
	CLIENT_UPDATE_PROGRESS=1 UPDATE_URL="$$UPDATE_URL_USE" UPDATE_PUBKEY="$$PKEY" $(PY_BIN) $(UPDATE_SCRIPT); \
	SERVER_ADDR="$$SERVER_URL" UPDATE_URL="$$UPDATE_URL_USE" UPDATE_PUBKEY="$$PKEY" $(PY_BIN) $(CLIENT_BIN); \
	rc=$$?; \
	if [ $$rc -ne 0 ]; then \
		echo "[client] клиент завершился с ошибкой (rc=$$rc). Последние строки логов:"; \
		tail -n 20 var/log/client-startup.log 2>/dev/null || true; \
		tail -n 20 var/log/client.log 2>/dev/null || true; \
		exit $$rc; \
	fi

run-debug: deps ensure-structure ensure-client
	@SERVER_URL="$(or $(SERVER),$(DEFAULT_SERVER))"; \
	UPDATE_URL_USE="$(or $(UPDATE_URL),$(DEFAULT_UPDATE_URL))"; \
	CLIENT_DEBUG_LOG=1 LOG_LEVEL=DEBUG SERVER_ADDR="$$SERVER_URL" UPDATE_URL="$$UPDATE_URL_USE" UPDATE_PUBKEY="$(or $(UPDATE_PUBKEY),$(DEFAULT_PUBKEY))" $(PY_BIN) $(CLIENT_BIN)

update: deps ensure-structure
	@if [ -z "$(UPDATE_URL)" ]; then UPDATE_URL=$(DEFAULT_UPDATE_URL); fi; \
	CLIENT_UPDATE_PROGRESS=1 UPDATE_URL="$$UPDATE_URL" UPDATE_PUBKEY="$(or $(UPDATE_PUBKEY),$(DEFAULT_PUBKEY))" $(PY_BIN) $(UPDATE_SCRIPT)

ensure-structure:
	@for d in $(REQUIRED_DIRS); do mkdir -p $$d; done
	@chmod +x $(CLIENT_BIN) 2>/dev/null || true

check-structure: ensure-structure
	@$(PY) $(CHECK_SCRIPT)

test:
	@$(PY) -m compileall -q . && $(PY) -m unittest discover -s tests -p 'test_*.py'

ensure-client:
	@if [ ! -s "$(CLIENT_BIN)" ] || [ ! -d "modules" ]; then \
		$(MAKE) update UPDATE_URL=$(or $(UPDATE_URL),$(DEFAULT_UPDATE_URL)); \
	fi

clean:
	rm -rf .venv var/log/* var/history/* var/users/* runtime
