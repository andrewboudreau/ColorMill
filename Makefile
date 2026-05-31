APP_NAME := colormill
BUILD_DIR := build
DIST_DIR := dist
SRC := src/main.c src/sim/material_sim.c
CFLAGS := -std=c99 -Wall -Wextra -Isrc
WEB_CFLAGS := $(CFLAGS) -Iraylib/src
WEB_SHELL := web/shell.html
WEB_PAGES := web/project.html web/resources.html
RAYLIB_WEB_A ?= raylib/src/libraylib.a

.PHONY: native web clean

native:
	mkdir -p $(BUILD_DIR)
	cc $(CFLAGS) $(SRC) -o $(BUILD_DIR)/$(APP_NAME) -lraylib -lm

web:
	mkdir -p $(DIST_DIR)
	emcc $(WEB_CFLAGS) $(SRC) -o $(DIST_DIR)/index.html -DPLATFORM_WEB -Os -s USE_GLFW=3 -s ASYNCIFY --shell-file $(WEB_SHELL) $(RAYLIB_WEB_A) -lm
	cp $(WEB_PAGES) $(DIST_DIR)/

clean:
	rm -rf $(BUILD_DIR) $(DIST_DIR)
