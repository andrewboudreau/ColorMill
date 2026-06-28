#include "mixbox.h"

/*
 * Mixbox latent-space pigment mixing, ported from the official Mixbox library
 * (mixbox.js: unpackedFloatRgbToLatent / evalPolynomial / latentToFloatRgb).
 *
 * A Mixbox latent is 7 floats: { c0, c1, c2, c3, residualR, residualG, residualB }
 * where c0..c3 are Kubelka-Munk pigment weights (c3 = 1 - c0 - c1 - c2).
 * Mixing is a componentwise weighted average of latents; reconstruction is
 * evalPolynomial(c0..c3) + residual, clamped to 0..1.
 *
 * Endpoint latents below were generated from the official library for fixed
 * pigment colors (see scripts in the PR / web/vendor/mixbox):
 *   WHITE  base    rgb(255, 255, 255)
 *   RED    pigment Cadmium Red    rgb(255, 39, 2)
 *   BLUE   pigment Cobalt Blue    rgb(0, 33, 133)
 *   YELLOW pigment Cadmium Yellow rgb(254, 236, 0)
 *
 * Mixbox is (c) Secret Weapons, CC BY-NC 4.0 (non-commercial use).
 */

#define MB_LATENT 7

static const float MB_WHITE[MB_LATENT] = {
    0.00000000f, 0.00000000f, 0.00000000f, 1.00000000f, 0.00481862f, 0.00021851f, 0.00295198f
};
static const float MB_RED[MB_LATENT] = {
    0.00000000f, 0.33960806f, 0.65860828f, 0.00178365f, 0.08873356f, -0.01747544f, -0.06755477f
};
static const float MB_BLUE[MB_LATENT] = {
    0.86413725f, 0.00441961f, 0.02987264f, 0.10157050f, -0.05379499f, -0.01226009f, 0.00350272f
};
static const float MB_YELLOW[MB_LATENT] = {
    0.00392157f, 0.85950980f, 0.00000000f, 0.13656863f, 0.03300247f, 0.10249173f, -0.08066935f
};

static float Clamp01(float x) {
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return x;
}

static void EvalPolynomial(float c0, float c1, float c2, float c3,
                           float *r, float *g, float *b) {
    const float c00 = c0 * c0;
    const float c11 = c1 * c1;
    const float c22 = c2 * c2;
    const float c33 = c3 * c3;
    const float c01 = c0 * c1;
    const float c02 = c0 * c2;
    const float c12 = c1 * c2;

    float R = 0.0f, G = 0.0f, B = 0.0f, w;
    w = c0 * c00; R += +0.07717053f * w; G += +0.02826978f * w; B += +0.24832992f * w;
    w = c1 * c11; R += +0.95912302f * w; G += +0.80256528f * w; B += +0.03561839f * w;
    w = c2 * c22; R += +0.74683774f * w; G += +0.04868586f * w; B += +0.00000000f * w;
    w = c3 * c33; R += +0.99518138f * w; G += +0.99978149f * w; B += +0.99704802f * w;
    w = c00 * c1; R += +0.04819146f * w; G += +0.83363781f * w; B += +0.32515377f * w;
    w = c01 * c1; R += -0.68146950f * w; G += +1.46107803f * w; B += +1.06980936f * w;
    w = c00 * c2; R += +0.27058419f * w; G += -0.15324870f * w; B += +1.98735057f * w;
    w = c02 * c2; R += +0.80478189f * w; G += +0.67093710f * w; B += +0.18424500f * w;
    w = c00 * c3; R += -0.35031003f * w; G += +1.37855826f * w; B += +3.68865000f * w;
    w = c0 * c33; R += +1.05128046f * w; G += +1.97815239f * w; B += +2.82989073f * w;
    w = c11 * c2; R += +3.21607125f * w; G += +0.81270228f * w; B += +1.03384539f * w;
    w = c1 * c22; R += +2.78893374f * w; G += +0.41565549f * w; B += -0.04487295f * w;
    w = c11 * c3; R += +3.02162577f * w; G += +2.55374103f * w; B += +0.32766114f * w;
    w = c1 * c33; R += +2.95124691f * w; G += +2.81201112f * w; B += +1.17578442f * w;
    w = c22 * c3; R += +2.82677043f * w; G += +0.79933038f * w; B += +1.81715262f * w;
    w = c2 * c33; R += +2.99691099f * w; G += +1.22593053f * w; B += +1.80653661f * w;
    w = c01 * c2; R += +1.87394106f * w; G += +2.05027182f * w; B += -0.29835996f * w;
    w = c01 * c3; R += +2.56609566f * w; G += +7.03428198f * w; B += +0.62575374f * w;
    w = c02 * c3; R += +4.08329484f * w; G += -1.40408358f * w; B += +2.14995522f * w;
    w = c12 * c3; R += +6.00078678f * w; G += +2.55552042f * w; B += +1.90739502f * w;

    *r = R;
    *g = G;
    *b = B;
}

void Mixbox_PigmentRgb(float redConcentration, float blueConcentration,
                       float yellowConcentration,
                       float *outR, float *outG, float *outB) {
    const float redC = Clamp01(redConcentration);
    const float blueC = Clamp01(blueConcentration);
    const float yellowC = Clamp01(yellowConcentration);

    /* White base fills the remaining mass; if the pigments overflow the
       material, renormalize so the pigment weights sum to one. */
    const float total = redC + blueC + yellowC;
    float wWhite, wRed, wBlue, wYellow;
    if (total > 1.0f) {
        wWhite = 0.0f;
        wRed = redC / total;
        wBlue = blueC / total;
        wYellow = yellowC / total;
    } else {
        wWhite = 1.0f - total;
        wRed = redC;
        wBlue = blueC;
        wYellow = yellowC;
    }

    float z[MB_LATENT];
    for (int i = 0; i < MB_LATENT; i++) {
        z[i] = wWhite * MB_WHITE[i] + wRed * MB_RED[i] +
               wBlue * MB_BLUE[i] + wYellow * MB_YELLOW[i];
    }

    float r, g, b;
    EvalPolynomial(z[0], z[1], z[2], z[3], &r, &g, &b);
    *outR = Clamp01(r + z[4]);
    *outG = Clamp01(g + z[5]);
    *outB = Clamp01(b + z[6]);
}
