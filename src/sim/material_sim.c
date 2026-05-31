#include "material_sim.h"

#include <math.h>
#include <string.h>

static float Clamp01(float value) {
    if (value < 0.0f) return 0.0f;
    if (value > 1.0f) return 1.0f;
    return value;
}

static float Smooth01(float edge0, float edge1, float value) {
    const float t = Clamp01((value - edge0) / (edge1 - edge0));
    return t * t * (3.0f - 2.0f * t);
}

static float SampleField(const float *field, float x, float y) {
    if (x < 0.0f || y < 0.0f || x > (float)(SIM_WIDTH - 1) || y > (float)(SIM_HEIGHT - 1)) {
        return 0.0f;
    }

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

static float MixingHorizonAt(int x, int y, float centerX, float centerY, float halfGap) {
    const float fx = (float)x;
    const float fy = (float)y;
    const float nipX = 1.0f - Smooth01(halfGap * 0.65f, halfGap * 2.1f, fabsf(fx - centerX));
    const float nipY = 1.0f - Smooth01(3.0f, 14.0f, fabsf(fy - centerY));
    return nipX * nipY;
}

static void WriteRollerVelocity(MaterialSim *sim) {
    const float centerX = (float)(SIM_WIDTH - 1) * 0.5f;
    const float centerY = (float)(SIM_HEIGHT - 1) * 0.48f;
    const float halfGap = 7.0f + sim->gap * 14.0f;
    const float rollerRadius = 25.0f;
    const float leftCenterX = centerX - halfGap - rollerRadius;
    const float rightCenterX = centerX + halfGap + rollerRadius;
    const float surfaceSpeed = 28.0f * sim->rollerSpeed;

    for (int y = 0; y < SIM_HEIGHT; y++) {
        for (int x = 0; x < SIM_WIDTH; x++) {
            const float fx = (float)x;
            const float fy = (float)y;
            const float dxLeft = fx - leftCenterX;
            const float dxRight = fx - rightCenterX;
            const float dy = fy - centerY;
            const float dLeft = sqrtf(dxLeft * dxLeft + dy * dy);
            const float dRight = sqrtf(dxRight * dxRight + dy * dy);
            const float leftSurface = 1.0f - Smooth01(rollerRadius, rollerRadius + 14.0f, dLeft);
            const float rightSurface = 1.0f - Smooth01(rollerRadius, rollerRadius + 14.0f, dRight);
            const float nip = MixingHorizonAt(x, y, centerX, centerY, halfGap);
            const int index = MaterialSim_Index(x, y);

            float vx = 0.0f;
            float vy = 0.0f;

            if (dLeft > 0.001f) {
                vx += (-dy / dLeft) * surfaceSpeed * leftSurface;
                vy += (dxLeft / dLeft) * surfaceSpeed * leftSurface;
            }

            if (dRight > 0.001f) {
                vx += (dy / dRight) * surfaceSpeed * rightSurface;
                vy += (-dxRight / dRight) * surfaceSpeed * rightSurface;
            }

            vy += surfaceSpeed * (0.35f + nip * 1.25f);
            vx += -((fx - centerX) / centerX) * nip * surfaceSpeed * 0.42f;

            if (fy > centerY + 18.0f) {
                const float returnBand = Smooth01(centerY + 18.0f, (float)SIM_HEIGHT - 4.0f, fy);
                vx += -surfaceSpeed * 0.45f * returnBand;
                vy += -surfaceSpeed * 0.24f * returnBand;
            }

            sim->vx[index] = vx;
            sim->vy[index] = vy;
        }
    }
}

int MaterialSim_Index(int x, int y) {
    return y * SIM_WIDTH + x;
}

void MaterialSim_Reset(MaterialSim *sim) {
    memset(sim->material, 0, sizeof(sim->material));
    memset(sim->red, 0, sizeof(sim->red));
    memset(sim->blue, 0, sizeof(sim->blue));
    memset(sim->vx, 0, sizeof(sim->vx));
    memset(sim->vy, 0, sizeof(sim->vy));
    memset(sim->nextMaterial, 0, sizeof(sim->nextMaterial));
    memset(sim->nextRed, 0, sizeof(sim->nextRed));
    memset(sim->nextBlue, 0, sizeof(sim->nextBlue));

    sim->rollerAngle = 0.0f;
    sim->rollerSpeed = 0.85f;
    sim->diffusion = 0.055f;
    sim->gap = 0.26f;
    sim->paused = false;

    for (int y = 4; y < 31; y++) {
        for (int x = 45; x < 83; x++) {
            const int index = MaterialSim_Index(x, y);
            sim->material[index] = 0.92f;
        }
    }

    MaterialSim_AddPigment(sim, 0.43f, 0.22f, 1.0f, 0.04f, 0.08f);
    MaterialSim_AddPigment(sim, 0.57f, 0.25f, 0.04f, 1.0f, 0.08f);
    MaterialSim_AddPigment(sim, 0.50f, 0.36f, 0.95f, 0.05f, 0.07f);
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
            sim->material[index] = Clamp01(sim->material[index] + amount * 0.24f);
            sim->red[index] = Clamp01(sim->red[index] + red * amount * sim->material[index]);
            sim->blue[index] = Clamp01(sim->blue[index] + blue * amount * sim->material[index]);
        }
    }
}

void MaterialSim_Step(MaterialSim *sim, float dt) {
    if (sim->paused) return;

    const float safeDt = dt > 0.04f ? 0.04f : dt;
    const float centerX = (float)(SIM_WIDTH - 1) * 0.5f;
    const float centerY = (float)(SIM_HEIGHT - 1) * 0.48f;
    const float halfGap = 7.0f + sim->gap * 14.0f;
    const float baseDiffusion = Clamp01(sim->diffusion * safeDt * 70.0f);

    sim->rollerAngle += sim->rollerSpeed * safeDt;
    WriteRollerVelocity(sim);

    for (int y = 0; y < SIM_HEIGHT; y++) {
        for (int x = 0; x < SIM_WIDTH; x++) {
            const int index = MaterialSim_Index(x, y);
            const float sourceX = (float)x - sim->vx[index] * safeDt;
            const float sourceY = (float)y - sim->vy[index] * safeDt;
            const float material = SampleField(sim->material, sourceX, sourceY);
            const float red = SampleField(sim->red, sourceX, sourceY);
            const float blue = SampleField(sim->blue, sourceX, sourceY);
            const float mix = MixingHorizonAt(x, y, centerX, centerY, halfGap);
            const float diffusion = baseDiffusion * mix;

            sim->nextMaterial[index] = Clamp01(material);
            sim->nextRed[index] = Clamp01(DiffuseAt(sim->red, x, y, red, diffusion) * sim->nextMaterial[index]);
            sim->nextBlue[index] = Clamp01(DiffuseAt(sim->blue, x, y, blue, diffusion) * sim->nextMaterial[index]);
        }
    }

    memcpy(sim->material, sim->nextMaterial, sizeof(sim->material));
    memcpy(sim->red, sim->nextRed, sizeof(sim->red));
    memcpy(sim->blue, sim->nextBlue, sizeof(sim->blue));
}

float MaterialSim_MaterialAt(const MaterialSim *sim, int x, int y) {
    return sim->material[MaterialSim_Index(x, y)];
}

float MaterialSim_RedAt(const MaterialSim *sim, int x, int y) {
    return sim->red[MaterialSim_Index(x, y)];
}

float MaterialSim_BlueAt(const MaterialSim *sim, int x, int y) {
    return sim->blue[MaterialSim_Index(x, y)];
}

float MaterialSim_VelocityXAt(const MaterialSim *sim, int x, int y) {
    return sim->vx[MaterialSim_Index(x, y)];
}

float MaterialSim_VelocityYAt(const MaterialSim *sim, int x, int y) {
    return sim->vy[MaterialSim_Index(x, y)];
}
