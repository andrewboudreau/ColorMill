#include "raylib.h"

#include "sim/material_sim.h"
#include "sim/mixbox.h"

#include <math.h>

#if defined(PLATFORM_WEB)
#include <emscripten/emscripten.h>
#endif

#define SCREEN_WIDTH 960
#define SCREEN_HEIGHT 640
#define FIELD_X 48
#define FIELD_Y 96
#define FIELD_W 864
#define FIELD_H 486

typedef struct AppState {
    MaterialSim sim;
    Texture2D fieldTexture;
    Image fieldImage;
    Color *pixels;
    int viewMode;
    bool showGrid;
} AppState;

static AppState app;

static float Clamp01f(float value) {
    if (value < 0.0f) return 0.0f;
    if (value > 1.0f) return 1.0f;
    return value;
}

static unsigned char ToByte(float value) {
    return (unsigned char)(Clamp01f(value) * 255.0f);
}

static Color CellColor(float material, float red, float blue, float vx, float vy, float bandMask, float mixMask) {
    if (app.viewMode == 1) {
        const unsigned char shade = ToByte(material);
        return (Color){ shade, shade, shade, 255 };
    }

    if (app.viewMode == 2) {
        const float value = material > 0.001f ? red / material : 0.0f;
        return (Color){ ToByte(value), 28, 36, 255 };
    }

    if (app.viewMode == 3) {
        const float value = material > 0.001f ? blue / material : 0.0f;
        return (Color){ 28, 42, ToByte(value), 255 };
    }

    if (app.viewMode == 4) {
        const float speed = Clamp01f(sqrtf(vx * vx + vy * vy) / 26.0f);
        const float dir = atan2f(vy, vx) / 6.2831853f + 0.5f;
        return (Color){ ToByte(speed), ToByte(dir), ToByte(1.0f - speed), 255 };
    }

    if (app.viewMode == 5) {
        const unsigned char shade = ToByte(bandMask);
        return (Color){ shade, shade, shade, 255 };
    }

    if (app.viewMode == 6) {
        return (Color){ ToByte(mixMask), ToByte(mixMask * 0.18f), ToByte(mixMask), 255 };
    }

    if (material < 0.015f) return (Color){ 226, 232, 240, 255 };

    /* Pigment-correct mixing: treat the cell as a white silicone base carrying
       red and blue pigment, mixed in Mixbox latent space so blends follow real
       pigment behavior instead of linear RGB averaging. */
    const float invMaterial = 1.0f / fmaxf(material, 0.001f);
    const float redC = Clamp01f(red * invMaterial);
    const float blueC = Clamp01f(blue * invMaterial);

    float r, g, b;
    Mixbox_PigmentRgb(redC, blueC, &r, &g, &b);
    return (Color){ ToByte(r), ToByte(g), ToByte(b), 255 };
}

static const char *ViewName(void) {
    if (app.viewMode == 1) return "material";
    if (app.viewMode == 2) return "red";
    if (app.viewMode == 3) return "blue";
    if (app.viewMode == 4) return "velocity";
    if (app.viewMode == 5) return "band mask";
    if (app.viewMode == 6) return "mix mask";
    return "mixed";
}

static void UpdateFieldTexture(void) {
    for (int y = 0; y < SIM_HEIGHT; y++) {
        for (int x = 0; x < SIM_WIDTH; x++) {
            const float material = MaterialSim_MaterialAt(&app.sim, x, y);
            const float red = MaterialSim_RedAt(&app.sim, x, y);
            const float blue = MaterialSim_BlueAt(&app.sim, x, y);
            const float vx = MaterialSim_VelocityXAt(&app.sim, x, y);
            const float vy = MaterialSim_VelocityYAt(&app.sim, x, y);
            const float band = MaterialSim_BandMaskAt(&app.sim, x, y);
            const float mix = MaterialSim_MixMaskAt(&app.sim, x, y);
            app.pixels[MaterialSim_Index(x, y)] = CellColor(material, red, blue, vx, vy, band, mix);
        }
    }

    UpdateTexture(app.fieldTexture, app.pixels);
}

static void DrawGridOverlay(void) {
    if (!app.showGrid) return;

    const float cellW = (float)FIELD_W / (float)SIM_WIDTH;
    const float cellH = (float)FIELD_H / (float)SIM_HEIGHT;

    for (int x = 0; x <= SIM_WIDTH; x += 8) {
        const float px = FIELD_X + (float)x * cellW;
        DrawLine((int)px, FIELD_Y, (int)px, FIELD_Y + FIELD_H, (Color){ 15, 23, 42, 42 });
    }

    for (int y = 0; y <= SIM_HEIGHT; y += 8) {
        const float py = FIELD_Y + (float)y * cellH;
        DrawLine(FIELD_X, (int)py, FIELD_X + FIELD_W, (int)py, (Color){ 15, 23, 42, 42 });
    }
}

static void DrawDomainGuides(void) {
    const int nipY = FIELD_Y + (int)((float)FIELD_H * 0.48f);
    DrawRectangle(FIELD_X, FIELD_Y, FIELD_W, nipY - FIELD_Y - 28, (Color){ 37, 99, 235, 24 });
    DrawRectangle(FIELD_X, nipY - 10, FIELD_W, 20, (Color){ 147, 51, 234, 58 });
    DrawRectangle(FIELD_X, nipY + 20, FIELD_W, FIELD_Y + FIELD_H - nipY - 20, (Color){ 15, 118, 110, 22 });
    DrawLineEx((Vector2){ FIELD_X, nipY }, (Vector2){ FIELD_X + FIELD_W, nipY }, 3.0f, (Color){ 126, 34, 206, 210 });
    DrawText("feed / entry", FIELD_X + 12, FIELD_Y + 10, 16, (Color){ 30, 64, 175, 235 });
    DrawText("nip / mixing horizon", FIELD_X + 12, nipY - 28, 16, (Color){ 88, 28, 135, 255 });
    DrawText("return / transport", FIELD_X + 12, nipY + 28, 16, (Color){ 15, 95, 90, 235 });
}

static bool MouseInField(Vector2 mouse) {
    return mouse.x >= FIELD_X && mouse.x <= FIELD_X + FIELD_W && mouse.y >= FIELD_Y && mouse.y <= FIELD_Y + FIELD_H;
}

static void AddPigmentAtMouse(float red, float blue) {
    const Vector2 mouse = GetMousePosition();
    if (!MouseInField(mouse)) return;

    const float x = (mouse.x - FIELD_X) / (float)FIELD_W;
    const float y = (mouse.y - FIELD_Y) / (float)FIELD_H;
    MaterialSim_AddPigment(&app.sim, x, y, red, blue, 0.04f);
}

static void HandleInput(void) {
    if (IsKeyPressed(KEY_SPACE)) app.sim.paused = !app.sim.paused;
    if (IsKeyPressed(KEY_R)) MaterialSim_Reset(&app.sim);
    if (IsKeyPressed(KEY_G)) app.showGrid = !app.showGrid;

    if (IsKeyPressed(KEY_ONE)) app.viewMode = 0;
    if (IsKeyPressed(KEY_TWO)) app.viewMode = 1;
    if (IsKeyPressed(KEY_THREE)) app.viewMode = 2;
    if (IsKeyPressed(KEY_FOUR)) app.viewMode = 3;
    if (IsKeyPressed(KEY_FIVE)) app.viewMode = 4;
    if (IsKeyPressed(KEY_SIX)) app.viewMode = 5;
    if (IsKeyPressed(KEY_SEVEN)) app.viewMode = 6;

    if (IsKeyDown(KEY_UP)) app.sim.rollerSpeed += GetFrameTime() * 0.55f;
    if (IsKeyDown(KEY_DOWN)) app.sim.rollerSpeed -= GetFrameTime() * 0.55f;
    if (IsKeyDown(KEY_RIGHT)) app.sim.gap += GetFrameTime() * 0.16f;
    if (IsKeyDown(KEY_LEFT)) app.sim.gap -= GetFrameTime() * 0.16f;

    app.sim.rollerSpeed = Clamp01f(app.sim.rollerSpeed);
    app.sim.gap = 0.12f + Clamp01f((app.sim.gap - 0.12f) / 0.38f) * 0.38f;

    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) AddPigmentAtMouse(1.0f, 0.05f);
    if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) AddPigmentAtMouse(0.05f, 1.0f);
}

static void DrawHud(void) {
    DrawText("ColorMill simulation grid", 48, 24, 30, (Color){ 15, 23, 42, 255 });
    DrawText(TextFormat("view: %s", ViewName()), 50, 60, 18, (Color){ 51, 65, 85, 255 });
    DrawText(TextFormat("speed %.2f   gap %.2f", app.sim.rollerSpeed, app.sim.gap), 260, 60, 18, (Color){ 51, 65, 85, 255 });
    DrawText(app.sim.paused ? "paused" : "running", 800, 60, 18, app.sim.paused ? ORANGE : DARKGREEN);
    DrawText("1 mixed 2 material 3 red 4 blue 5 velocity 6 band 7 mix | G grid SPACE pause R reset", 54, 602, 16, (Color){ 51, 65, 85, 255 });
}

static void UpdateDrawFrame(void) {
    HandleInput();
    MaterialSim_Step(&app.sim, GetFrameTime());
    UpdateFieldTexture();

    BeginDrawing();
    ClearBackground((Color){ 238, 242, 247, 255 });

    DrawTexturePro(
        app.fieldTexture,
        (Rectangle){ 0, 0, (float)app.fieldTexture.width, (float)app.fieldTexture.height },
        (Rectangle){ FIELD_X, FIELD_Y, FIELD_W, FIELD_H },
        (Vector2){ 0, 0 },
        0.0f,
        WHITE
    );
    DrawDomainGuides();
    DrawGridOverlay();
    DrawRectangleLinesEx((Rectangle){ FIELD_X, FIELD_Y, FIELD_W, FIELD_H }, 2.0f, (Color){ 15, 23, 42, 180 });
    DrawHud();

    EndDrawing();
}

int main(void) {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "ColorMill");
    SetTargetFPS(60);

    MaterialSim_Init(&app.sim);
    app.viewMode = 0;
    app.showGrid = true;
    app.fieldImage = GenImageColor(SIM_WIDTH, SIM_HEIGHT, BLANK);
    app.fieldTexture = LoadTextureFromImage(app.fieldImage);
    app.pixels = (Color *)MemAlloc(sizeof(Color) * SIM_CELL_COUNT);
    UpdateFieldTexture();

#if defined(PLATFORM_WEB)
    emscripten_set_main_loop(UpdateDrawFrame, 0, 1);
#else
    while (!WindowShouldClose()) {
        UpdateDrawFrame();
    }
#endif

    MemFree(app.pixels);
    UnloadTexture(app.fieldTexture);
    UnloadImage(app.fieldImage);
    CloseWindow();
    return 0;
}
