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

typedef struct AppState {
    MpmSim3D sim;
    Camera3D camera;
    float orbit;       /* azimuth angle */
    float elevation;   /* pitch angle */
    float distance;
    bool autoOrbit;
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

/* sim coordinates are [0,1] with +y down; map to a unit cube centered at the
   origin with +y up for rendering */
static Vector3 SimToWorld(float sx, float sy, float sz) {
    return (Vector3){ sx - 0.5f, -(sy - 0.5f), sz - 0.5f };
}

static Color VoxelColor(float material, float red, float blue) {
    const float inv = 1.0f / fmaxf(material, 0.001f);
    const float redC = Clamp01f(red * inv);
    const float blueC = Clamp01f(blue * inv);
    float r, g, b;
    Mixbox_PigmentRgb(redC, blueC, &r, &g, &b);
    return (Color){ ToByte(r), ToByte(g), ToByte(b), 255 };
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
                const Color color = VoxelColor(material, red, blue);
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

    float leftCx, rightCx, centerY, radius;
    MpmSim3D_Rollers(&app.sim, &leftCx, &rightCx, &centerY, &radius);

    const Color shell = (Color){ 148, 163, 184, 90 };
    const Color wire = (Color){ 226, 232, 240, 180 };
    const float zFront = -0.52f, zBack = 0.52f;

    const Vector3 lFront = SimToWorld(leftCx, centerY, zFront + 0.5f);
    const Vector3 lBack = SimToWorld(leftCx, centerY, zBack + 0.5f);
    const Vector3 rFront = SimToWorld(rightCx, centerY, zFront + 0.5f);
    const Vector3 rBack = SimToWorld(rightCx, centerY, zBack + 0.5f);

    DrawCylinderEx(lFront, lBack, radius, radius, 20, shell);
    DrawCylinderWiresEx(lFront, lBack, radius, radius, 20, wire);
    DrawCylinderEx(rFront, rBack, radius, radius, 20, shell);
    DrawCylinderWiresEx(rFront, rBack, radius, radius, 20, wire);
}

static void UpdateCameraOrbit(void) {
    if (app.autoOrbit) app.orbit += GetFrameTime() * 0.35f;

    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        const Vector2 delta = GetMouseDelta();
        app.orbit -= delta.x * 0.006f;
        app.elevation = Clamp01f((app.elevation - delta.y * 0.006f + 1.5f) / 3.0f) * 3.0f - 1.5f;
    }

    const float ce = cosf(app.elevation);
    app.camera.position = (Vector3){
        app.distance * ce * sinf(app.orbit),
        app.distance * sinf(app.elevation),
        app.distance * ce * cosf(app.orbit)
    };
}

static void HandleInput(void) {
    if (IsKeyPressed(KEY_SPACE)) app.sim.paused = !app.sim.paused;
    if (IsKeyPressed(KEY_R)) MpmSim3D_Reset(&app.sim);
    if (IsKeyPressed(KEY_G)) app.showRollers = !app.showRollers;
    if (IsKeyPressed(KEY_O)) app.autoOrbit = !app.autoOrbit;

    if (IsKeyDown(KEY_UP)) app.sim.rollerSpeed += GetFrameTime() * 0.55f;
    if (IsKeyDown(KEY_DOWN)) app.sim.rollerSpeed -= GetFrameTime() * 0.55f;
    if (IsKeyDown(KEY_RIGHT)) app.sim.gap += GetFrameTime() * 0.5f;
    if (IsKeyDown(KEY_LEFT)) app.sim.gap -= GetFrameTime() * 0.5f;
    app.sim.rollerSpeed = Clamp01f(app.sim.rollerSpeed);
    app.sim.gap = Clamp01f(app.sim.gap);

    if (IsKeyDown(KEY_X)) MpmSim3D_AddPigment(&app.sim, 0.5f, 0.45f, 0.5f, 1.0f, 0.0f, 0.12f);
    if (IsKeyDown(KEY_C)) MpmSim3D_AddPigment(&app.sim, 0.5f, 0.45f, 0.5f, 0.0f, 1.0f, 0.12f);

    app.distance -= GetMouseWheelMove() * 0.15f;
    if (app.distance < 1.4f) app.distance = 1.4f;
    if (app.distance > 4.0f) app.distance = 4.0f;
}

static void DrawHud(void) {
    DrawText("ColorMill - 3D MLS-MPM mill", 24, 20, 28, (Color){ 15, 23, 42, 255 });
    DrawText(TextFormat("speed %.2f   gap %.2f   particles %d   filled voxels rendered",
                        app.sim.rollerSpeed, app.sim.gap, app.sim.particleCount),
             24, 54, 18, (Color){ 51, 65, 85, 255 });
    DrawText(app.sim.paused ? "paused" : "running",
             SCREEN_WIDTH - 90, 24, 18, app.sim.paused ? ORANGE : DARKGREEN);
    DrawText("drag: orbit  wheel: zoom  O auto-orbit | UP/DOWN speed  LEFT/RIGHT gap | X/C add red/blue | G rollers SPACE pause R reset",
             24, SCREEN_HEIGHT - 26, 15, (Color){ 51, 65, 85, 255 });
}

static void UpdateDrawFrame(void) {
    HandleInput();
    MpmSim3D_Step(&app.sim, GetFrameTime());
    UpdateCameraOrbit();

    BeginDrawing();
    ClearBackground((Color){ 238, 242, 247, 255 });

    BeginMode3D(app.camera);
    DrawCubeWires((Vector3){ 0, 0, 0 }, 1.0f, 1.0f, 1.0f, (Color){ 148, 163, 184, 120 });
    DrawVoxels();
    DrawRollers();
    EndMode3D();

    DrawHud();
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
    app.camera.target = (Vector3){ 0.0f, 0.0f, 0.0f };
    app.camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    app.camera.fovy = 45.0f;
    app.camera.projection = CAMERA_PERSPECTIVE;
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
