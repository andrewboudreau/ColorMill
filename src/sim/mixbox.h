#ifndef MIXBOX_H
#define MIXBOX_H

/*
 * Pigment-correct color mixing for ColorMill.
 *
 * Mixes a white silicone base with red, blue, and yellow pigment by
 * concentration (each in 0..1, measured as pigment mass / material mass) using
 * Mixbox latent-space mixing (Kubelka-Munk). Blue + red yields purple and blue
 * + yellow yields green, the way real pigment does, instead of the muddy gray
 * produced by linear RGB averaging.
 *
 * Output is sRGB in 0..1.
 *
 * Mixbox is (c) Secret Weapons, licensed CC BY-NC 4.0 (non-commercial).
 * The latent endpoint constants in mixbox.c were generated from the official
 * Mixbox library; see web/vendor/mixbox/ for attribution and the source build.
 */
void Mixbox_PigmentRgb(float redConcentration, float blueConcentration,
                       float yellowConcentration,
                       float *outR, float *outG, float *outB);

#endif
