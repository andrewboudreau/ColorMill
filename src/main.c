#include "raylib.h"

#include "sim/mpm_sim.h"
#include "sim/mixbox.h"

#include <math.h>

#if defined(PLATFORM_WEB)
#include <emscripten/emscripten.h>
#endif

#define SCREEN_WIDTH 960
#define SCREEN_HEIGHT 640
#define FIELD_SIDE 540
#define FIELD_X ((SCREEN_WIDTH - FIELD_SIDE) / 2)
#define FIELD_Y 70

typedef struct AppState {
    MpmSim sim;
    Texture2D fieldTexture;
    Image fieldImage;
    Color *pixels;
    int viewMode;
    bool showRollers;
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

static Color CellColor(float material, float red, float blue, float vx, float vy) {
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
        const float speed = Clamp01f(sqrtf(vx * vx + vy * vy) / 1.5f);
        const float dir = atan2f(vy, vx) / 6.2831853f + 0.5f;
        return (Color){ ToByte(speed), ToByte(dir), ToByte(1.0f - speed), 255 };
    }

    if (material < 0.06f) return (Color){ 226, 232, 240, 255 };

    /* Pigment-correct mixing: white silicone base carrying red + blue pigment,
       mixed in Mixbox latent space so folds read as real pigment blends. */
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
    return "mixed (Mixbox)";
}

static void UpdateFieldTexture(void) {
    for (int y = 0; y < MPM_GRID; y++) {
        for (int x = 0; x < MPM_GRID; x++) {
            const float material = MpmSim_MaterialAt(&app.sim, x, y);
            const float red = MpmSim_RedAt(&app.sim, x, y);
            const float blue = MpmSim_BlueAt(&app.sim, x, y);
            const float vx = MpmSim_VelocityXAt(&app.sim, x, y);
            const float vy = MpmSim_VelocityYAt(&app.sim, x, y);
            app.pixels[MpmSim_Index(x, y)] = CellColor(material, red, blue, vx, vy);
        }
    }

    UpdateTexture(app.fieldTexture, app.pixels);
}

static Vector2 FieldPoint(float normalizedX, float normalizedY) {
    return (Vector2){
        FIELD_X + normalizedX * FIELD_SIDE,
        FIELD_Y + normalizedY * FIELD_SIDE
    };
}

static void DrawRoller(Vector2 center, float radius, float angle, bool clockwise) {
    DrawCircleV(center, radius, (Color){ 30, 41, 59, 70 });
    DrawCircleLines((int)center.x, (int)center.y, radius, (Color){ 226, 232, 240, 230 });
    DrawCircleV(center, 4.0f, (Color){ 226, 232, 240, 230 });

    /* spokes communicate rotation direction */
    const float dir = clockwise ? 1.0f : -1.0f;
    for (int s = 0; s < 4; s++) {
        const float a = dir * angle + (float)s * (3.14159265f * 0.5f);
        const Vector2 tip = { center.x + cosf(a) * radius, center.y + sinf(a) * radius };
        DrawLineEx(center, tip, 2.0f, (Color){ 148, 163, 184, 180 });
    }
}

static void DrawRollers(void) {
    if (!app.showRollers) return;

    float leftCx, rightCx, centerY, radius;
    MpmSim_Rollers(&app.sim, &leftCx, &rightCx, &centerY, &radius);
    const float pixelRadius = radius * FIELD_SIDE;

    DrawRoller(FieldPoint(leftCx, centerY), pixelRadius, app.sim.rollerAngle, true);
    DrawRoller(FieldPoint(rightCx, centerY), pixelRadius, app.sim.rollerAngle, false);

    const Vector2 nipTop = FieldPoint(0.5f, centerY - radius - 0.04f);
    DrawText("nip", (int)nipTop.x - 12, (int)nipTop.y, 16, (Color){ 88, 28, 135, 220 });
}

static bool MouseInField(Vector2 mouse) {
    return mouse.x >= FIELD_X && mouse.x <= FIELD_X + FIELD_SIDE &&
           mouse.y >= FIELD_Y && mouse.y <= FIELD_Y + FIELD_SIDE;
}

static void AddPigmentAtMouse(float red, float blue) {
    const Vector2 mouse = GetMousePosition();
    if (!MouseInField(mouse)) return;

    const float x = (mouse.x - FIELD_X) / (float)FIELD_SIDE;
    const float y = (mouse.y - FIELD_Y) / (float)FIELD_SIDE;
    MpmSim_AddPigment(&app.sim, x, y, red, blue, 0.06f);
}

static void HandleInput(void) {
    if (IsKeyPressed(KEY_SPACE)) app.sim.paused = !app.sim.paused;
    if (IsKeyPressed(KEY_R)) MpmSim_Reset(&app.sim);
    if (IsKeyPressed(KEY_G)) app.showRollers = !app.showRollers;

    if (IsKeyPressed(KEY_ONE)) app.viewMode = 0;
    if (IsKeyPressed(KEY_TWO)) app.viewMode = 1;
    if (IsKeyPressed(KEY_THREE)) app.viewMode = 2;
    if (IsKeyPressed(KEY_FOUR)) app.viewMode = 3;
    if (IsKeyPressed(KEY_FIVE)) app.viewMode = 4;

    if (IsKeyDown(KEY_UP)) app.sim.rollerSpeed += GetFrameTime() * 0.55f;
    if (IsKeyDown(KEY_DOWN)) app.sim.rollerSpeed -= GetFrameTime() * 0.55f;
    if (IsKeyDown(KEY_RIGHT)) app.sim.gap += GetFrameTime() * 0.5f;
    if (IsKeyDown(KEY_LEFT)) app.sim.gap -= GetFrameTime() * 0.5f;

    app.sim.rollerSpeed = Clamp01f(app.sim.rollerSpeed);
    app.sim.gap = Clamp01f(app.sim.gap);

    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) AddPigmentAtMouse(1.0f, 0.0f);
    if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) AddPigmentAtMouse(0.0f, 1.0f);
}

static void DrawHud(void) {
    DrawText("ColorMill - MLS-MPM mill", FIELD_X, 24, 28, (Color){ 15, 23, 42, 255 });
    DrawText(TextFormat("view: %s", ViewName()), FIELD_X, 56, 18, (Color){ 51, 65, 85, 255 });
    DrawText(TextFormat("speed %.2f   gap %.2f   particles %d",
                        app.sim.rollerSpeed, app.sim.gap, app.sim.particleCount),
             FIELD_X + 230, 56, 18, (Color){ 51, 65, 85, 255 });
    DrawText(app.sim.paused ? "paused" : "running",
             FIELD_X + FIELD_SIDE - 80, 56, 18, app.sim.paused ? ORANGE : DARKGREEN);
    DrawText("1 mixed 2 material 3 red 4 blue 5 velocity | LMB red RMB blue | UP/DOWN speed LEFT/RIGHT gap | G rollers SPACE pause R reset",
             FIELD_X - 120, SCREEN_HEIGHT - 26, 15, (Color){ 51, 65, 85, 255 });
}

static void UpdateDrawFrame(void) {
    HandleInput();
    MpmSim_Step(&app.sim, GetFrameTime());
    UpdateFieldTexture();

    BeginDrawing();
    ClearBackground((Color){ 238, 242, 247, 255 });

    DrawTexturePro(
        app.fieldTexture,
        (Rectangle){ 0, 0, (float)app.fieldTexture.width, (float)app.fieldTexture.height },
        (Rectangle){ FIELD_X, FIELD_Y, FIELD_SIDE, FIELD_SIDE },
        (Vector2){ 0, 0 },
        0.0f,
        WHITE
    );
    DrawRollers();
    DrawRectangleLinesEx((Rectangle){ FIELD_X, FIELD_Y, FIELD_SIDE, FIELD_SIDE }, 2.0f, (Color){ 15, 23, 42, 180 });
    DrawHud();

    EndDrawing();
}

int main(void) {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "ColorMill");
    SetTargetFPS(60);

    MpmSim_Init(&app.sim);
    app.viewMode = 0;
    app.showRollers = true;
    app.fieldImage = GenImageColor(MPM_GRID, MPM_GRID, BLANK);
    app.fieldTexture = LoadTextureFromImage(app.fieldImage);
    app.pixels = (Color *)MemAlloc(sizeof(Color) * MPM_GRID * MPM_GRID);
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
