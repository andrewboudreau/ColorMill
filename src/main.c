#include "raylib.h"

#include "sim/mpm3d_sim.h"
#include "sim/mixbox.h"

#include <math.h>

#if defined(PLATFORM_WEB)
#include <emscripten/emscripten.h>
#endif

#define SCREEN_WIDTH 960
#define SCREEN_HEIGHT 640
#define VOXEL_THRESHOLD 0.16f

typedef struct UiButton {
    Rectangle rect;
    Color color;
    const char *label;
    int action;   /* 0 red, 1 blue, 2 yellow, 3 clear, 4 zoom out, 5 zoom in */
} UiButton;

typedef struct AppState {
    MpmSim3D sim;
    Camera3D camera;
    float orbit;
    float elevation;
    float distance;
    bool autoOrbit;
    bool showRollers;
    bool dragging;
    UiButton buttons[8];
    int buttonCount;
} AppState;

static AppState app;

static float Clamp01f(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }
static unsigned char ToByte(float v) { return (unsigned char)(Clamp01f(v) * 255.0f); }

/* sim coords are [0,1] with +y down; map to a unit cube centered at origin, +y up */
static Vector3 SimToWorld(float sx, float sy, float sz) {
    return (Vector3){ sx - 0.5f, -(sy - 0.5f), sz - 0.5f };
}

static Color VoxelColor(float material, float red, float blue, float yellow, float shade) {
    const float inv = 1.0f / fmaxf(material, 0.001f);
    float r, g, b;
    Mixbox_PigmentRgb(Clamp01f(red * inv), Clamp01f(blue * inv), Clamp01f(yellow * inv),
                      &r, &g, &b);
    return (Color){ ToByte(r * shade), ToByte(g * shade), ToByte(b * shade), 255 };
}

static void DrawVoxels(void) {
    const float cell = 1.0f / (float)MPM3D_GRID;
    const float size = cell * 0.92f;
    for (int z = 0; z < MPM3D_GRID; z++) {
        for (int y = 0; y < MPM3D_GRID; y++) {
            for (int x = 0; x < MPM3D_GRID; x++) {
                const float material = MpmSim3D_MaterialAt(&app.sim, x, y, z);
                if (material < VOXEL_THRESHOLD) continue;
                const float red = MpmSim3D_RedAt(&app.sim, x, y, z);
                const float blue = MpmSim3D_BlueAt(&app.sim, x, y, z);
                const float yellow = MpmSim3D_YellowAt(&app.sim, x, y, z);
                /* simple height shading so the form reads on a flat-lit scene */
                const float shade = 0.62f + 0.38f * (1.0f - (y + 0.5f) * cell);
                const Color color = VoxelColor(material, red, blue, yellow, shade);
                const Vector3 pos = SimToWorld((x + 0.5f) * cell,
                                               (y + 0.5f) * cell,
                                               (z + 0.5f) * cell);
                DrawCube(pos, size, size, size, color);
            }
        }
    }
}

static void DrawRollers(void) {
    if (!app.showRollers) return;
    float leftCx, rightCx, centerY, radius, zMin, zMax;
    MpmSim3D_Rollers(&app.sim, &leftCx, &rightCx, &centerY, &radius, &zMin, &zMax);
    const Color shell = (Color){ 148, 163, 184, 80 };
    const Color wire = (Color){ 226, 232, 240, 150 };
    const Vector3 lF = SimToWorld(leftCx, centerY, zMin), lB = SimToWorld(leftCx, centerY, zMax);
    const Vector3 rF = SimToWorld(rightCx, centerY, zMin), rB = SimToWorld(rightCx, centerY, zMax);
    DrawCylinderEx(lF, lB, radius, radius, 20, shell);
    DrawCylinderWiresEx(lF, lB, radius, radius, 20, wire);
    DrawCylinderEx(rF, rB, radius, radius, 20, shell);
    DrawCylinderWiresEx(rF, rB, radius, radius, 20, wire);
}

static void UpdateCameraOrbit(void) {
    if (app.autoOrbit && !app.dragging) app.orbit += GetFrameTime() * 0.3f;

    if (app.dragging && IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        const Vector2 d = GetMouseDelta();
        app.orbit -= d.x * 0.006f;
        app.elevation -= d.y * 0.006f;
        if (app.elevation > 1.35f) app.elevation = 1.35f;
        if (app.elevation < -0.4f) app.elevation = -0.4f;
    }

    const float ce = cosf(app.elevation);
    app.camera.position = (Vector3){
        app.distance * ce * sinf(app.orbit),
        app.distance * sinf(app.elevation),
        app.distance * ce * cosf(app.orbit)
    };
}

static void InjectPigment(float r, float b, float y) {
    /* drop a pigment blob at a jittered spot inside the material */
    const float nx = 0.40f + (float)GetRandomValue(0, 200) / 1000.0f;
    const float ny = 0.50f + (float)GetRandomValue(0, 160) / 1000.0f;
    const float nz = 0.44f + (float)GetRandomValue(0, 120) / 1000.0f;
    MpmSim3D_AddPigment(&app.sim, nx, ny, nz, r, b, y, 0.16f);
}

static void DoAction(int action) {
    switch (action) {
        case 0: InjectPigment(1.0f, 0.0f, 0.0f); break;
        case 1: InjectPigment(0.0f, 1.0f, 0.0f); break;
        case 2: InjectPigment(0.0f, 0.0f, 1.0f); break;
        case 3: MpmSim3D_ClearPigment(&app.sim); break;
        case 4: app.distance += 0.3f; break;
        case 5: app.distance -= 0.3f; break;
        case 6: MpmSim3D_Reset(&app.sim); break;
        default: break;
    }
    if (app.distance < 1.4f) app.distance = 1.4f;
    if (app.distance > 4.0f) app.distance = 4.0f;
}

static int ButtonAt(Vector2 m) {
    for (int i = 0; i < app.buttonCount; i++) {
        if (CheckCollisionPointRec(m, app.buttons[i].rect)) return i;
    }
    return -1;
}

static void HandleInput(void) {
    const Vector2 mouse = GetMousePosition();
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        const int b = ButtonAt(mouse);
        if (b >= 0) {
            DoAction(app.buttons[b].action);
            app.dragging = false;
        } else {
            app.dragging = true;   /* drag in the 3D area orbits the camera */
        }
    }
    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) app.dragging = false;

    app.distance -= GetMouseWheelMove() * 0.15f;
    if (app.distance < 1.4f) app.distance = 1.4f;
    if (app.distance > 4.0f) app.distance = 4.0f;

    if (IsKeyPressed(KEY_SPACE)) app.sim.paused = !app.sim.paused;
    if (IsKeyPressed(KEY_R)) MpmSim3D_Reset(&app.sim);
    if (IsKeyPressed(KEY_G)) app.showRollers = !app.showRollers;
    if (IsKeyPressed(KEY_O)) app.autoOrbit = !app.autoOrbit;
    if (IsKeyPressed(KEY_X)) DoAction(0);
    if (IsKeyPressed(KEY_C)) DoAction(1);
    if (IsKeyPressed(KEY_V)) DoAction(2);
    if (IsKeyPressed(KEY_K)) DoAction(3);

    if (IsKeyDown(KEY_UP)) app.sim.rollerSpeed += GetFrameTime() * 0.55f;
    if (IsKeyDown(KEY_DOWN)) app.sim.rollerSpeed -= GetFrameTime() * 0.55f;
    if (IsKeyDown(KEY_RIGHT)) app.sim.gap += GetFrameTime() * 0.5f;
    if (IsKeyDown(KEY_LEFT)) app.sim.gap -= GetFrameTime() * 0.5f;
    app.sim.rollerSpeed = Clamp01f(app.sim.rollerSpeed);
    app.sim.gap = Clamp01f(app.sim.gap);
}

static void DrawButton(const UiButton *b) {
    const Color base = b->color;
    DrawRectangleRounded(b->rect, 0.35f, 6, base);
    DrawRectangleRoundedLines(b->rect, 0.35f, 6, (Color){ 226, 232, 240, 90 });
    const int fs = 20;
    const int tw = MeasureText(b->label, fs);
    const Color ink = (b->action == 2) ? (Color){ 30, 30, 30, 255 } : (Color){ 245, 248, 252, 255 };
    DrawText(b->label, (int)(b->rect.x + (b->rect.width - tw) / 2),
             (int)(b->rect.y + (b->rect.height - fs) / 2), fs, ink);
}

static void DrawUI(void) {
    DrawText("ColorMill - 3D MLS-MPM mill", 20, 16, 26, (Color){ 226, 232, 240, 255 });
    DrawText(TextFormat("speed %.2f   gap %.2f   particles %d",
                        app.sim.rollerSpeed, app.sim.gap, app.sim.particleCount),
             20, 48, 18, (Color){ 148, 163, 184, 255 });
    DrawText(app.sim.paused ? "paused" : "running",
             SCREEN_WIDTH - 96, 18, 18, app.sim.paused ? ORANGE : GREEN);
    DrawText("tap a color to add pigment  -  drag to orbit  -  +/- zoom",
             20, SCREEN_HEIGHT - 76, 16, (Color){ 148, 163, 184, 255 });
    for (int i = 0; i < app.buttonCount; i++) DrawButton(&app.buttons[i]);
}

static void InitUI(void) {
    const float h = 40.0f, w = 96.0f, gap = 8.0f, y = SCREEN_HEIGHT - 52.0f;
    float x = 20.0f;
    app.buttons[0] = (UiButton){ { x, y, w, h }, (Color){ 214, 64, 50, 255 }, "Red", 0 };     x += w + gap;
    app.buttons[1] = (UiButton){ { x, y, w, h }, (Color){ 40, 92, 200, 255 }, "Blue", 1 };    x += w + gap;
    app.buttons[2] = (UiButton){ { x, y, w, h }, (Color){ 236, 200, 30, 255 }, "Yellow", 2 };  x += w + gap;
    app.buttons[3] = (UiButton){ { x, y, w, h }, (Color){ 100, 116, 139, 255 }, "Clear", 3 };  x += w + gap;
    app.buttons[4] = (UiButton){ { x, y, w, h }, (Color){ 71, 85, 105, 255 }, "Reset", 6 };
    app.buttons[5] = (UiButton){ { SCREEN_WIDTH - 124.0f, y, 50.0f, h }, (Color){ 51, 65, 85, 255 }, "-", 4 };
    app.buttons[6] = (UiButton){ { SCREEN_WIDTH - 66.0f, y, 50.0f, h }, (Color){ 51, 65, 85, 255 }, "+", 5 };
    app.buttonCount = 7;
}

static void UpdateDrawFrame(void) {
    HandleInput();
    MpmSim3D_Step(&app.sim, GetFrameTime());
    UpdateCameraOrbit();

    BeginDrawing();
    ClearBackground((Color){ 15, 23, 42, 255 });

    BeginMode3D(app.camera);
    DrawCubeWires((Vector3){ 0, 0, 0 }, 1.0f, 1.0f, 1.0f, (Color){ 71, 85, 105, 150 });
    DrawVoxels();
    DrawRollers();
    EndMode3D();

    DrawUI();
    EndDrawing();
}

int main(void) {
    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "ColorMill");
    SetTargetFPS(60);

    MpmSim3D_Init(&app.sim);
    app.orbit = 0.7f;
    app.elevation = 0.5f;
    app.distance = 2.4f;
    app.autoOrbit = true;
    app.showRollers = true;
    app.dragging = false;
    app.camera.target = (Vector3){ 0.0f, 0.0f, 0.0f };
    app.camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    app.camera.fovy = 45.0f;
    app.camera.projection = CAMERA_PERSPECTIVE;
    InitUI();
    UpdateCameraOrbit();

#if defined(PLATFORM_WEB)
    emscripten_set_main_loop(UpdateDrawFrame, 0, 1);
#else
    while (!WindowShouldClose()) {
        UpdateDrawFrame();
    }
#endif

    CloseWindow();
    return 0;
}
