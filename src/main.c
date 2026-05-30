#include "raylib.h"

#include "sim/material_sim.h"

#include <math.h>

#if defined(PLATFORM_WEB)
#include <emscripten/emscripten.h>
#endif

#define SCREEN_WIDTH 960
#define SCREEN_HEIGHT 640
#define FIELD_X 72
#define FIELD_Y 118
#define FIELD_W 816
#define FIELD_H 408

typedef struct AppState {
    MaterialSim sim;
    Texture2D fieldTexture;
    Image fieldImage;
    Color *pixels;
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

static Color PigmentColor(float red, float blue) {
    const float mixed = red < blue ? red : blue;
    const float material = Clamp01f(0.34f + (red + blue) * 0.34f);

    Color color = {
        ToByte(material + red * 0.62f + mixed * 0.22f),
        ToByte(material * 0.88f - mixed * 0.18f),
        ToByte(material + blue * 0.68f + mixed * 0.28f),
        255
    };

    return color;
}

static void UpdateFieldTexture(void) {
    for (int y = 0; y < SIM_HEIGHT; y++) {
        for (int x = 0; x < SIM_WIDTH; x++) {
            const float red = MaterialSim_RedAt(&app.sim, x, y);
            const float blue = MaterialSim_BlueAt(&app.sim, x, y);
            app.pixels[MaterialSim_Index(x, y)] = PigmentColor(red, blue);
        }
    }

    UpdateTexture(app.fieldTexture, app.pixels);
}

static void DrawRoller(Vector2 center, float radius, float angle, const char *label) {
    DrawCircleV(center, radius + 4.0f, (Color){ 31, 41, 55, 255 });
    DrawCircleV(center, radius, (Color){ 148, 163, 184, 255 });
    DrawCircleV((Vector2){ center.x - radius * 0.25f, center.y - radius * 0.25f }, radius * 0.42f, (Color){ 226, 232, 240, 120 });

    for (int i = 0; i < 8; i++) {
        const float theta = angle + (float)i * 0.7853982f;
        const Vector2 a = { center.x + cosf(theta) * radius * 0.18f, center.y + sinf(theta) * radius * 0.18f };
        const Vector2 b = { center.x + cosf(theta) * radius * 0.84f, center.y + sinf(theta) * radius * 0.84f };
        DrawLineEx(a, b, 3.0f, (Color){ 248, 250, 252, 210 });
    }

    DrawText(label, (int)(center.x - MeasureText(label, 18) / 2), (int)(center.y + radius + 22), 18, (Color){ 15, 23, 42, 255 });
}

static void DrawControls(void) {
    DrawRectangleRounded((Rectangle){ 72, 546, 816, 66 }, 0.18f, 12, (Color){ 255, 255, 255, 215 });
    DrawText("SPACE pause  |  R reset  |  1 red pigment  |  2 blue pigment  |  UP/DOWN speed  |  LEFT/RIGHT gap", 96, 568, 18, (Color){ 51, 65, 85, 255 });

    const char *status = app.sim.paused ? "Paused" : "Running";
    DrawText(status, 96, 590, 16, app.sim.paused ? ORANGE : DARKGREEN);

    DrawText(TextFormat("speed %.2f", app.sim.rollerSpeed), 198, 590, 16, (Color){ 71, 85, 105, 255 });
    DrawText(TextFormat("gap %.2f", app.sim.gap), 310, 590, 16, (Color){ 71, 85, 105, 255 });
    DrawText(TextFormat("diffusion %.3f", app.sim.diffusion), 402, 590, 16, (Color){ 71, 85, 105, 255 });
}

static bool MouseInField(Vector2 mouse) {
    return mouse.x >= FIELD_X && mouse.x <= FIELD_X + FIELD_W && mouse.y >= FIELD_Y && mouse.y <= FIELD_Y + FIELD_H;
}

static void AddPigmentAtMouse(float red, float blue) {
    const Vector2 mouse = GetMousePosition();
    if (!MouseInField(mouse)) return;

    const float x = (mouse.x - FIELD_X) / (float)FIELD_W;
    const float y = (mouse.y - FIELD_Y) / (float)FIELD_H;
    MaterialSim_AddPigment(&app.sim, x, y, red, blue, 0.045f);
}

static void HandleInput(void) {
    if (IsKeyPressed(KEY_SPACE)) app.sim.paused = !app.sim.paused;
    if (IsKeyPressed(KEY_R)) MaterialSim_Reset(&app.sim);
    if (IsKeyPressed(KEY_ONE)) MaterialSim_AddPigment(&app.sim, 0.36f, 0.46f, 1.0f, 0.02f, 0.105f);
    if (IsKeyPressed(KEY_TWO)) MaterialSim_AddPigment(&app.sim, 0.64f, 0.54f, 0.02f, 1.0f, 0.105f);

    if (IsKeyDown(KEY_UP)) app.sim.rollerSpeed += GetFrameTime() * 0.55f;
    if (IsKeyDown(KEY_DOWN)) app.sim.rollerSpeed -= GetFrameTime() * 0.55f;
    if (IsKeyDown(KEY_RIGHT)) app.sim.gap += GetFrameTime() * 0.16f;
    if (IsKeyDown(KEY_LEFT)) app.sim.gap -= GetFrameTime() * 0.16f;

    app.sim.rollerSpeed = Clamp01f(app.sim.rollerSpeed);
    app.sim.gap = 0.12f + Clamp01f((app.sim.gap - 0.12f) / 0.38f) * 0.38f;

    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) AddPigmentAtMouse(1.0f, 0.05f);
    if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) AddPigmentAtMouse(0.05f, 1.0f);
}

static void UpdateDrawFrame(void) {
    HandleInput();
    MaterialSim_Step(&app.sim, GetFrameTime());
    UpdateFieldTexture();

    BeginDrawing();
    ClearBackground((Color){ 238, 242, 247, 255 });

    DrawText("ColorMill", 72, 34, 42, (Color){ 15, 23, 42, 255 });
    DrawText("C / raylib / WASM pigment field simulator", 74, 82, 20, (Color){ 71, 85, 105, 255 });

    DrawRectangleRounded((Rectangle){ 48, 100, 864, 430 }, 0.035f, 14, (Color){ 203, 213, 225, 255 });
    DrawTexturePro(
        app.fieldTexture,
        (Rectangle){ 0, 0, (float)app.fieldTexture.width, (float)app.fieldTexture.height },
        (Rectangle){ FIELD_X, FIELD_Y, FIELD_W, FIELD_H },
        (Vector2){ 0, 0 },
        0.0f,
        WHITE
    );

    DrawLineEx((Vector2){ 480, FIELD_Y }, (Vector2){ 480, FIELD_Y + FIELD_H }, 2.0f, (Color){ 15, 23, 42, 160 });
    DrawText("nip / shear zone", 410, 126, 18, (Color){ 15, 23, 42, 220 });

    DrawRoller((Vector2){ 360, 322 }, 82.0f, app.sim.rollerAngle, "left roller");
    DrawRoller((Vector2){ 600, 322 }, 82.0f, -app.sim.rollerAngle, "right roller");
    DrawRectangleLinesEx((Rectangle){ FIELD_X, FIELD_Y, FIELD_W, FIELD_H }, 2.0f, (Color){ 15, 23, 42, 180 });

    DrawControls();
    EndDrawing();
}

int main(void) {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "ColorMill");
    SetTargetFPS(60);

    MaterialSim_Init(&app.sim);
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
