APP_NAME := colormill
BUILD_DIR := build
DIST_DIR := dist
SRC := src/main.c src/sim/material_sim.c
CFLAGS := -std=c99 -Wall -Wextra -Isrc
RAYLIB_WEB_A ?= raylib/src/libraylib.a

.PHONY: native web clean

native:
	mkdir -p $(BUILD_DIR)
	cc $(CFLAGS) $(SRC) -o $(BUILD_DIR)/$(APP_NAME) -lraylib -lm

web:
	mkdir -p $(DIST_DIR)
	emcc $(CFLAGS) $(SRC) -o $(DIST_DIR)/index.html -DPLATFORM_WEB -Os -s USE_GLFW=3 -s ASYNCIFY $(RAYLIB_WEB_A)

clean:
	rm -rf $(BUILD_DIR) $(DIST_DIR)
