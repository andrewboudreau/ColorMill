#include "material_sim.h"

#include <math.h>
#include <string.h>

static float Clamp01(float value) {
    if (value < 0.0f) return 0.0f;
    if (value > 1.0f) return 1.0f;
    return value;
}

static float SampleField(const float *field, float x, float y) {
    if (x < 0.0f) x = 0.0f;
    if (y < 0.0f) y = 0.0f;
    if (x > (float)(SIM_WIDTH - 1)) x = (float)(SIM_WIDTH - 1);
    if (y > (float)(SIM_HEIGHT - 1)) y = (float)(SIM_HEIGHT - 1);

    const int x0 = (int)x;
    const int y0 = (int)y;
    const int x1 = x0 < SIM_WIDTH - 1 ? x0 + 1 : x0;
    const int y1 = y0 < SIM_HEIGHT - 1 ? y0 + 1 : y0;
    const float tx = x - (float)x0;
    const float ty = y - (float)y0;

    const float a = field[MaterialSim_Index(x0, y0)] * (1.0f - tx) + field[MaterialSim_Index(x1, y0)] * tx;
    const float b = field[MaterialSim_Index(x0, y1)] * (1.0f - tx) + field[MaterialSim_Index(x1, y1)] * tx;
    return a * (1.0f - ty) + b * ty;
}

static float DiffuseAt(const float *field, int x, int y, float center, float diffusion) {
    const int left = x > 0 ? x - 1 : x;
    const int right = x < SIM_WIDTH - 1 ? x + 1 : x;
    const int up = y > 0 ? y - 1 : y;
    const int down = y < SIM_HEIGHT - 1 ? y + 1 : y;

    const float neighbors = (
        field[MaterialSim_Index(left, y)] +
        field[MaterialSim_Index(right, y)] +
        field[MaterialSim_Index(x, up)] +
        field[MaterialSim_Index(x, down)]
    ) * 0.25f;

    return center + (neighbors - center) * diffusion;
}

int MaterialSim_Index(int x, int y) {
    return y * SIM_WIDTH + x;
}

void MaterialSim_Reset(MaterialSim *sim) {
    memset(sim->red, 0, sizeof(sim->red));
    memset(sim->blue, 0, sizeof(sim->blue));
    memset(sim->nextRed, 0, sizeof(sim->nextRed));
    memset(sim->nextBlue, 0, sizeof(sim->nextBlue));

    sim->rollerAngle = 0.0f;
    sim->rollerSpeed = 0.85f;
    sim->diffusion = 0.055f;
    sim->gap = 0.26f;
    sim->paused = false;

    MaterialSim_AddPigment(sim, 0.33f, 0.34f, 1.0f, 0.04f, 0.115f);
    MaterialSim_AddPigment(sim, 0.68f, 0.40f, 0.04f, 1.0f, 0.125f);
    MaterialSim_AddPigment(sim, 0.46f, 0.62f, 0.95f, 0.05f, 0.09f);
    MaterialSim_AddPigment(sim, 0.58f, 0.66f, 0.05f, 0.95f, 0.09f);
}

void MaterialSim_Init(MaterialSim *sim) {
    MaterialSim_Reset(sim);
}

void MaterialSim_AddPigment(MaterialSim *sim, float normalizedX, float normalizedY, float red, float blue, float radius) {
    const float cx = normalizedX * (float)(SIM_WIDTH - 1);
    const float cy = normalizedY * (float)(SIM_HEIGHT - 1);
    const float r = radius * (float)SIM_WIDTH;
    const float r2 = r * r;

    for (int y = 0; y < SIM_HEIGHT; y++) {
        for (int x = 0; x < SIM_WIDTH; x++) {
            const float dx = (float)x - cx;
            const float dy = (float)y - cy;
            const float distance2 = dx * dx + dy * dy;
            if (distance2 > r2) continue;

            const float amount = 1.0f - sqrtf(distance2 / r2);
            const int index = MaterialSim_Index(x, y);
            sim->red[index] = Clamp01(sim->red[index] + red * amount);
            sim->blue[index] = Clamp01(sim->blue[index] + blue * amount);
        }
    }
}

void MaterialSim_Step(MaterialSim *sim, float dt) {
    if (sim->paused) return;

    const float safeDt = dt > 0.04f ? 0.04f : dt;
    sim->rollerAngle += sim->rollerSpeed * safeDt;

    const float centerX = (float)(SIM_WIDTH - 1) * 0.5f;
    const float centerY = (float)(SIM_HEIGHT - 1) * 0.5f;
    const float gapPixels = sim->gap * (float)SIM_HEIGHT;
    const float diffusion = Clamp01(sim->diffusion * safeDt * 55.0f);

    for (int y = 0; y < SIM_HEIGHT; y++) {
        for (int x = 0; x < SIM_WIDTH; x++) {
            const float nx = ((float)x - centerX) / centerX;
            const float ny = ((float)y - centerY) / centerY;
            const float distanceToNip = fabsf((float)y - centerY);
            const float nipInfluence = Clamp01(1.0f - distanceToNip / gapPixels);

            const float shear = nx * nipInfluence * sim->rollerSpeed * safeDt * 38.0f;
            const float squeeze = ny * nipInfluence * sim->rollerSpeed * safeDt * 18.0f;
            const float wave = sinf(sim->rollerAngle * 3.0f + (float)y * 0.16f) * nipInfluence * 2.2f;

            const float sourceX = (float)x - shear - wave;
            const float sourceY = (float)y + squeeze;
            const float red = SampleField(sim->red, sourceX, sourceY);
            const float blue = SampleField(sim->blue, sourceX, sourceY);

            const int index = MaterialSim_Index(x, y);
            sim->nextRed[index] = Clamp01(DiffuseAt(sim->red, x, y, red, diffusion));
            sim->nextBlue[index] = Clamp01(DiffuseAt(sim->blue, x, y, blue, diffusion));
        }
    }

    memcpy(sim->red, sim->nextRed, sizeof(sim->red));
    memcpy(sim->blue, sim->nextBlue, sizeof(sim->blue));
}

float MaterialSim_RedAt(const MaterialSim *sim, int x, int y) {
    return sim->red[MaterialSim_Index(x, y)];
}

float MaterialSim_BlueAt(const MaterialSim *sim, int x, int y) {
    return sim->blue[MaterialSim_Index(x, y)];
}
