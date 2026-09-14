/*
 * millref - headless driver for the ColorMill v2 CPU reference solver.
 *
 *   millref [--cells 32] [--frames 240] [--every 20] [--out build/ref]
 *           [--E 60 --nu .35 --thetaC .025 --thetaS .0075 --gravity 2 --omega 3
 *            --ratio 1.25 --gap .05 --tack 1.5 --tackblend 1 --dt 1.6e-3 --sub 8
 *            --rho 0 --k 0.6 --mu 0.4 --floorfric 0.6 --damp 0 --fixed 0
 *            --blue 40 --yellow 60 --gap2 -1 --gapframe 120 --threads 0
 *            --plastic 0|1|2 --yield .02 --jmin/--jmax --bankh .22 --bankd .25 --imgevery N --noimg --selftest]
 *
 * Prints per-frame diagnostics (also build/ref/diag.csv) and writes PNG side
 * (y-z cross-section, +z to the right) and front (x-y) views.
 */
#define _GNU_SOURCE
#include "sim/millref.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* ------------------------------------------------------------------------ */
/* PNG writer: stored (uncompressed) deflate blocks, no external libs.        */

static unsigned crcTable[256];
static void CrcInit(void) {
    for (unsigned n = 0; n < 256; n++) {
        unsigned c = n;
        for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        crcTable[n] = c;
    }
}
static unsigned Crc(const unsigned char *b, size_t n, unsigned c) {
    c ^= 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) c = crcTable[(c ^ b[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}
static void Put32(unsigned char *p, unsigned v) { p[0] = v >> 24; p[1] = v >> 16; p[2] = v >> 8; p[3] = v; }
static void Chunk(FILE *f, const char *type, const unsigned char *data, size_t n) {
    unsigned char hdr[8];
    Put32(hdr, (unsigned)n);
    memcpy(hdr + 4, type, 4);
    fwrite(hdr, 1, 8, f);
    unsigned c = Crc(hdr + 4, 4, 0);
    if (n) { fwrite(data, 1, n, f); c = Crc(data, n, c); }
    unsigned char cb[4];
    Put32(cb, c);
    fwrite(cb, 1, 4, f);
}
static int PngWrite(const char *path, const unsigned char *rgb, int w, int h) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    const size_t rowBytes = (size_t)w * 3 + 1, raw = rowBytes * (size_t)h;
    unsigned char *rawBuf = (unsigned char *)malloc(raw);
    for (int y = 0; y < h; y++) {
        rawBuf[y * rowBytes] = 0;
        memcpy(rawBuf + y * rowBytes + 1, rgb + (size_t)y * w * 3, (size_t)w * 3);
    }
    const size_t nBlocks = (raw + 65534) / 65535;
    unsigned char *z = (unsigned char *)malloc(2 + raw + nBlocks * 5 + 4);
    size_t zi = 0;
    z[zi++] = 0x78; z[zi++] = 0x01;
    size_t off = 0;
    unsigned a = 1, b = 0;
    for (size_t i = 0; i < raw; i++) { a = (a + rawBuf[i]) % 65521; b = (b + a) % 65521; }
    while (off < raw) {
        size_t len = raw - off; if (len > 65535) len = 65535;
        z[zi++] = (off + len == raw) ? 1 : 0;
        z[zi++] = len & 0xFF; z[zi++] = (len >> 8) & 0xFF;
        z[zi++] = ~len & 0xFF; z[zi++] = (~len >> 8) & 0xFF;
        memcpy(z + zi, rawBuf + off, len);
        zi += len; off += len;
    }
    Put32(z + zi, (b << 16) | a); zi += 4;
    static const unsigned char sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    fwrite(sig, 1, 8, f);
    unsigned char ihdr[13];
    Put32(ihdr, (unsigned)w); Put32(ihdr + 4, (unsigned)h);
    ihdr[8] = 8; ihdr[9] = 2; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
    Chunk(f, "IHDR", ihdr, 13);
    Chunk(f, "IDAT", z, zi);
    Chunk(f, "IEND", NULL, 0);
    fclose(f);
    free(rawBuf); free(z);
    return 1;
}

/* ------------------------------------------------------------------------ */
/* image helpers                                                             */

#define SCALE 400   /* px per sim unit */
#define IMG_W 600   /* 1.5 units */
#define IMG_H 500   /* 1.25 units */

typedef struct { unsigned char *px; int w, h; } Img;

static void ImgClear(Img *im, unsigned char r, unsigned char g, unsigned char b) {
    for (int i = 0; i < im->w * im->h; i++) { im->px[i * 3] = r; im->px[i * 3 + 1] = g; im->px[i * 3 + 2] = b; }
}
static void ImgSet(Img *im, int x, int y, unsigned char r, unsigned char g, unsigned char b) {
    if (x < 0 || y < 0 || x >= im->w || y >= im->h) return;
    unsigned char *p = im->px + (y * im->w + x) * 3;
    p[0] = r; p[1] = g; p[2] = b;
}
/* sim (u, v) with u horizontal (0..1.5), v = y up (0..1.25) */
static void ImgDot(Img *im, float u, float v, int size, unsigned char r, unsigned char g, unsigned char b) {
    const int x = (int)(u * SCALE), y = im->h - 1 - (int)(v * SCALE);
    for (int dy = 0; dy < size; dy++) for (int dx = 0; dx < size; dx++) ImgSet(im, x + dx, y + dy, r, g, b);
}
static void ImgCircle(Img *im, float cu, float cv, float rad, unsigned char r, unsigned char g, unsigned char b) {
    const int n = 720;
    for (int i = 0; i < n; i++) {
        const float a = (float)i / n * 6.2831853f;
        ImgDot(im, cu + rad * cosf(a), cv + rad * sinf(a), 1, r, g, b);
    }
}
static void ImgHLine(Img *im, float v, float u0, float u1, unsigned char r, unsigned char g, unsigned char b) {
    for (float u = u0; u <= u1; u += 1.f / SCALE) ImgDot(im, u, v, 1, r, g, b);
}

static int cmpZ(const void *a, const void *b, void *ctx) {
    const MillRefParticle *p = (const MillRefParticle *)ctx;
    const float za = p[*(const int *)a].x[2], zb = p[*(const int *)b].x[2];
    return za < zb ? -1 : (za > zb ? 1 : 0);
}

static void WriteViews(const MillRefSim *sim, const char *out, int frame) {
    const int n = MillRef_ParticleCount(sim);
    const MillRefParticle *p = MillRef_Particles(sim);
    const float h = MillRef_H(sim);
    float bz, fz;
    MillRef_RollerAxes(sim, &bz, &fz);
    Img im;
    im.w = IMG_W; im.h = IMG_H;
    im.px = (unsigned char *)malloc((size_t)IMG_W * IMG_H * 3);
    char path[512];

    /* --- side view: u = z (front roller on the right), v = y ---------------- */
    ImgClear(&im, 24, 24, 28);
    ImgHLine(&im, 0.f, 0.f, 1.5f, 90, 90, 90);
    /* silhouette of everything in grey */
    for (int i = 0; i < n; i++) ImgDot(&im, p[i].x[2], p[i].x[1], 2, 70, 70, 76);
    /* mid-slab cross-section in colour (|x - L/2| < 3h) */
    for (int i = 0; i < n; i++) {
        if (fabsf(p[i].x[0] - 0.75f) > 3.f * h) continue;
        float rgb[3];
        MillRef_LatentToRgb(p[i].lat, rgb);
        ImgDot(&im, p[i].x[2], p[i].x[1], 2, (unsigned char)(rgb[0] * 255), (unsigned char)(rgb[1] * 255), (unsigned char)(rgb[2] * 255));
    }
    ImgCircle(&im, bz, MILLREF_AXIS_Y, MILLREF_RADIUS, 255, 90, 90);
    ImgCircle(&im, fz, MILLREF_AXIS_Y, MILLREF_RADIUS, 90, 200, 255);
    ImgCircle(&im, fz, MILLREF_AXIS_Y, MILLREF_RADIUS + 1.5f * h, 40, 90, 120);
    snprintf(path, sizeof path, "%s/side_%04d.png", out, frame);
    PngWrite(path, im.px, im.w, im.h);

    /* --- front view: u = x, v = y, painter's order along z ------------------ */
    ImgClear(&im, 24, 24, 28);
    ImgHLine(&im, MILLREF_AXIS_Y + MILLREF_RADIUS, 0.f, 1.5f, 90, 90, 90);
    ImgHLine(&im, MILLREF_AXIS_Y - MILLREF_RADIUS, 0.f, 1.5f, 90, 90, 90);
    ImgHLine(&im, MILLREF_AXIS_Y, 0.f, 1.5f, 60, 60, 60);
    ImgHLine(&im, 0.f, 0.f, 1.5f, 90, 90, 90);
    int *order = (int *)malloc(sizeof(int) * (size_t)n);
    for (int i = 0; i < n; i++) order[i] = i;
    qsort_r(order, (size_t)n, sizeof(int), cmpZ, (void *)p);
    for (int oi = 0; oi < n; oi++) {
        const int i = order[oi];
        float rgb[3];
        MillRef_LatentToRgb(p[i].lat, rgb);
        /* darken by depth a little so the sheet in front reads as nearer */
        const float shade = 0.6f + 0.4f * (p[i].x[2] / 1.5f);
        ImgDot(&im, p[i].x[0], p[i].x[1], 2, (unsigned char)(rgb[0] * 255 * shade), (unsigned char)(rgb[1] * 255 * shade), (unsigned char)(rgb[2] * 255 * shade));
    }
    free(order);
    snprintf(path, sizeof path, "%s/front_%04d.png", out, frame);
    PngWrite(path, im.px, im.w, im.h);
    free(im.px);
}

/* ------------------------------------------------------------------------ */

static int SelfTest(void) {
    unsigned s = 7;
    int bad = 0;
    for (int t = 0; t < 2000; t++) {
        float F[9];
        for (int i = 0; i < 9; i++) {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            F[i] = ((s / 4294967296.f) - 0.5f) * (t < 1000 ? 0.1f : 2.f) + (i % 4 == 0 ? 1.f : 0.f);
        }
        if (t % 50 == 0) F[8] = 0.f;                  /* singular case */
        float U[9], S[3], V[9];
        MillRef_Svd3(F, U, S, V);
        float US[9], Fr[9];
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) US[r * 3 + c] = U[r * 3 + c] * S[c];
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++)
            Fr[r * 3 + c] = US[r * 3] * V[c * 3] + US[r * 3 + 1] * V[c * 3 + 1] + US[r * 3 + 2] * V[c * 3 + 2];
        float err = 0.f, orthoU = 0.f, orthoV = 0.f;
        for (int i = 0; i < 9; i++) err += fabsf(Fr[i] - F[i]);
        for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) {
            float du = 0.f, dv = 0.f;
            for (int k = 0; k < 3; k++) { du += U[k * 3 + a] * U[k * 3 + b]; dv += V[k * 3 + a] * V[k * 3 + b]; }
            orthoU += fabsf(du - (a == b)); orthoV += fabsf(dv - (a == b));
        }
        const float dU = U[0] * (U[4] * U[8] - U[5] * U[7]) - U[1] * (U[3] * U[8] - U[5] * U[6]) + U[2] * (U[3] * U[7] - U[4] * U[6]);
        const float dV = V[0] * (V[4] * V[8] - V[5] * V[7]) - V[1] * (V[3] * V[8] - V[5] * V[6]) + V[2] * (V[3] * V[7] - V[4] * V[6]);
        if (err > 1e-4f || orthoU > 1e-4f || orthoV > 1e-4f || dU * dV < 0.f) {
            bad++;
            if (bad < 5) printf("svd fail t=%d err=%g oU=%g oV=%g dU=%g dV=%g\n", t, err, orthoU, orthoV, dU, dV);
        }
    }
    float rgb[3];
    MillRef_LatentToRgb(MILLREF_LAT_WHITE, rgb); printf("white -> %.3f %.3f %.3f\n", rgb[0], rgb[1], rgb[2]);
    MillRef_LatentToRgb(MILLREF_LAT_BLUE, rgb); printf("blue  -> %.3f %.3f %.3f\n", rgb[0], rgb[1], rgb[2]);
    MillRef_LatentToRgb(MILLREF_LAT_YELLOW, rgb); printf("yellow-> %.3f %.3f %.3f\n", rgb[0], rgb[1], rgb[2]);
    float mix[7];
    for (int i = 0; i < 7; i++) mix[i] = 0.5f * (MILLREF_LAT_BLUE[i] + MILLREF_LAT_YELLOW[i]);
    MillRef_LatentToRgb(mix, rgb); printf("blue+yellow -> %.3f %.3f %.3f (expect green)\n", rgb[0], rgb[1], rgb[2]);
    printf("svd selftest: %d failures / 2000\n", bad);
    return bad == 0 ? 0 : 1;
}

static float ArgF(int argc, char **argv, int *i, float def) {
    if (*i + 1 < argc) { (*i)++; return (float)atof(argv[*i]); }
    return def;
}

int main(int argc, char **argv) {
    CrcInit();
    MillRefOptions opt = MillRef_DefaultOptions();
    MillRefMaterial mat = MillRef_DefaultMaterial();
    MillRefParams par = MillRef_DefaultParams();
    int frames = 240, every = 20, imgEvery = -1, noImg = 0;
    int blueFrame = 40, yellowFrame = 60, gapFrame = 120;
    float gap2 = -1.f;
    float injX = 0.75f, injR = 0.09f, injStrength = 1.f;
    const char *out = "build/ref";
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--selftest")) return SelfTest();
        else if (!strcmp(a, "--cells")) opt.cells = (int)ArgF(argc, argv, &i, 32);
        else if (!strcmp(a, "--frames")) frames = (int)ArgF(argc, argv, &i, 240);
        else if (!strcmp(a, "--every")) every = (int)ArgF(argc, argv, &i, 20);
        else if (!strcmp(a, "--imgevery")) imgEvery = (int)ArgF(argc, argv, &i, 20);
        else if (!strcmp(a, "--noimg")) noImg = 1;
        else if (!strcmp(a, "--out")) { if (i + 1 < argc) out = argv[++i]; }
        else if (!strcmp(a, "--E")) mat.E = ArgF(argc, argv, &i, 60);
        else if (!strcmp(a, "--nu")) mat.nu = ArgF(argc, argv, &i, .35f);
        else if (!strcmp(a, "--thetaC")) mat.thetaC = ArgF(argc, argv, &i, .025f);
        else if (!strcmp(a, "--thetaS")) mat.thetaS = ArgF(argc, argv, &i, .0075f);
        else if (!strcmp(a, "--gravity")) par.gravity = ArgF(argc, argv, &i, 2);
        else if (!strcmp(a, "--omega")) par.omega = ArgF(argc, argv, &i, 3);
        else if (!strcmp(a, "--ratio")) par.frictionRatio = ArgF(argc, argv, &i, 1.25f);
        else if (!strcmp(a, "--gap")) par.gap = ArgF(argc, argv, &i, .05f);
        else if (!strcmp(a, "--k")) par.dispersion = ArgF(argc, argv, &i, .6f);
        else if (!strcmp(a, "--mu")) par.backFriction = ArgF(argc, argv, &i, .4f);
        else if (!strcmp(a, "--tack")) opt.tackBand = ArgF(argc, argv, &i, 1.5f);
        else if (!strcmp(a, "--tackblend")) opt.tackBlend = ArgF(argc, argv, &i, 1.f);
        else if (!strcmp(a, "--dt")) opt.dt = ArgF(argc, argv, &i, 1.6e-3f);
        else if (!strcmp(a, "--sub")) opt.sub = (int)ArgF(argc, argv, &i, 8);
        else if (!strcmp(a, "--rho")) opt.rho = ArgF(argc, argv, &i, 0);
        else if (!strcmp(a, "--floorfric")) opt.floorFriction = ArgF(argc, argv, &i, .6f);
        else if (!strcmp(a, "--damp")) opt.damping = ArgF(argc, argv, &i, 0);
        else if (!strcmp(a, "--fixed")) opt.fixedPoint = (int)ArgF(argc, argv, &i, 0);
        else if (!strcmp(a, "--threads")) opt.threads = (int)ArgF(argc, argv, &i, 0);
        else if (!strcmp(a, "--plastic")) opt.plastic = (int)ArgF(argc, argv, &i, 0);
        else if (!strcmp(a, "--yield")) opt.yieldStrain = ArgF(argc, argv, &i, .02f);
        else if (!strcmp(a, "--jmin")) opt.jMin = ArgF(argc, argv, &i, 0);
        else if (!strcmp(a, "--jmax")) opt.jMax = ArgF(argc, argv, &i, 0);
        else if (!strcmp(a, "--bankh")) opt.bankHeight = ArgF(argc, argv, &i, .22f);
        else if (!strcmp(a, "--bankd")) opt.bankHalfDepth = ArgF(argc, argv, &i, .25f);
        else if (!strcmp(a, "--seed")) opt.seed = (unsigned)ArgF(argc, argv, &i, 1234);
        else if (!strcmp(a, "--blue")) blueFrame = (int)ArgF(argc, argv, &i, 40);
        else if (!strcmp(a, "--yellow")) yellowFrame = (int)ArgF(argc, argv, &i, 60);
        else if (!strcmp(a, "--injx")) injX = ArgF(argc, argv, &i, .75f);
        else if (!strcmp(a, "--injr")) injR = ArgF(argc, argv, &i, .09f);
        else if (!strcmp(a, "--injs")) injStrength = ArgF(argc, argv, &i, 1.f);
        else if (!strcmp(a, "--gap2")) gap2 = ArgF(argc, argv, &i, -1);
        else if (!strcmp(a, "--gapframe")) gapFrame = (int)ArgF(argc, argv, &i, 120);
        else { fprintf(stderr, "unknown arg %s\n", a); return 2; }
    }
    if (imgEvery < 0) imgEvery = every;
    mkdir(out, 0755);

    MillRefSim *sim = MillRef_Create(&opt, &mat, &par);
    const float h = MillRef_H(sim);
    const float rhoSpec = 8.f / (h * h * h);
    const float mu = mat.E / (2.f * (1.f + mat.nu)), lam = mat.E * mat.nu / ((1.f + mat.nu) * (1.f - 2.f * mat.nu));
    const float rhoEff = opt.rho > 0.f ? opt.rho : rhoSpec;
    const float c = sqrtf((lam + 2.f * mu) / rhoEff);
    printf("millref: cells=%d h=%.5f particles=%d nodes=%dx%dx%d dt=%g sub=%d  E=%g nu=%g thetaC=%g thetaS=%g\n",
           opt.cells, h, MillRef_ParticleCount(sim), (int)lroundf(1.5f / h) + 1, (int)lroundf(1.25f / h) + 1,
           (int)lroundf(1.5f / h) + 1, opt.dt, opt.sub, mat.E, mat.nu, mat.thetaC, mat.thetaS);
    printf("  omega=%g ratio=%g gap=%g g=%g k=%g backmu=%g tack=%gh blend=%g rho=%s(%g) c=%.4g h/c=%.3g dt/(h/c)=%.3f  0.4h/sqrtE=%.3g\n",
           par.omega, par.frictionRatio, par.gap, par.gravity, par.dispersion, par.backFriction, opt.tackBand, opt.tackBlend,
           opt.rho > 0.f ? "user" : "spec", rhoEff, c, h / c, opt.dt / (h / c), 0.4f * h / sqrtf(mat.E));

    char path[512];
    snprintf(path, sizeof path, "%s/diag.csv", out);
    FILE *csv = fopen(path, "w");
    if (csv) fprintf(csv, "frame,t,ms,maxV,meanV,minJ,maxJ,nan,inDomain,inRoller,pushOuts,sheetFront,sheetBack,bank,floor,frontTop,backTop,inNip,nipShear,sheetY0,pigmented,pigSheet,greenSheet,pigBank,greenBank,blueBank,yellowBank,maxNodeMass,loose,high,sheetThick\n");

    printf("%5s %6s %6s | %7s %6s %6s %3s %6s %4s %5s | %6s %6s %6s %6s %6s %6s | %6s %6s %6s %6s | %5s %5s %5s %5s\n",
           "frame", "t", "ms", "maxV", "minJ", "maxJ", "nan", "inDom", "inR", "push", "front", "back", "bank", "floor",
           "fTop", "bTop", "nipG", "thick", "loose", "high", "pig", "grnS", "bluB", "yelB");
    double msSum = 0.0; int msCount = 0;
    for (int f = 0; f <= frames; f++) {
        if (f == blueFrame && blueFrame >= 0) {
            const float cpos[3] = {injX, MILLREF_AXIS_Y + MILLREF_RADIUS + opt.bankHeight - 0.06f, MILLREF_NIP_Z - 0.06f};
            MillRef_AddPigment(sim, cpos, injR, MILLREF_LAT_BLUE, injStrength);
        }
        if (f == yellowFrame && yellowFrame >= 0) {
            const float cpos[3] = {injX, MILLREF_AXIS_Y + MILLREF_RADIUS + opt.bankHeight - 0.06f, MILLREF_NIP_Z + 0.06f};
            MillRef_AddPigment(sim, cpos, injR, MILLREF_LAT_YELLOW, injStrength);
        }
        if (gap2 > 0.f && f == gapFrame) {
            MillRefParams p2 = par; p2.gap = gap2;
            MillRef_SetParams(sim, &p2);
            printf("-- gap changed to %g at frame %d\n", gap2, f);
        }
        if (f % every == 0 || f == frames) {
            MillRefStats st;
            MillRef_Stats(sim, &st);
            double fms, p2g, grid, g2p;
            MillRef_Timings(sim, &fms, &p2g, &grid, &g2p);
            printf("%5d %6.3f %6.1f | %7.3f %6.4f %6.4f %3d %6d %4d %5ld | %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f | %6.2f %6.3f %6.3f %6.3f | %5d %5.2f %5.2f %5.2f\n",
                   f, MillRef_Time(sim), fms, st.maxV, st.minJ, st.maxJ, st.nan, st.inDomain, st.inRoller, st.pushOuts,
                   st.sheetFront, st.sheetBack, st.bank, st.floor, st.frontTop, st.backTop, st.nipShear, st.sheetThick,
                   st.loose, st.high, st.pigmented, st.greenSheet, st.blueBank, st.yellowBank);
            fflush(stdout);
            if (csv) fprintf(csv, "%d,%.4f,%.1f,%g,%g,%g,%g,%d,%d,%d,%ld,%g,%g,%g,%g,%g,%g,%g,%g,%g,%d,%d,%g,%d,%g,%g,%g,%g,%g,%g,%g\n",
                             f, MillRef_Time(sim), fms, st.maxV, st.meanV, st.minJ, st.maxJ, st.nan, st.inDomain, st.inRoller, st.pushOuts,
                             st.sheetFront, st.sheetBack, st.bank, st.floor, st.frontTop, st.backTop, st.inNip, st.nipShear, st.sheetY0,
                             st.pigmented, st.pigmentedSheet, st.greenSheet, st.pigmentedBank, st.greenBank, st.blueBank, st.yellowBank, st.maxNodeMass, st.loose, st.high, st.sheetThick);
            if (st.nan > 0) { printf("NaN detected at frame %d; stopping\n", f); if (!noImg) WriteViews(sim, out, f); break; }
        }
        if (!noImg && (f % imgEvery == 0 || f == frames)) WriteViews(sim, out, f);
        if (f == frames) break;
        MillRef_Frame(sim);
        double fms, p2g, grid, g2p;
        MillRef_Timings(sim, &fms, &p2g, &grid, &g2p);
        if (f >= 2) { msSum += fms; msCount++; }
    }
    double fms, p2g, grid, g2p;
    MillRef_Timings(sim, &fms, &p2g, &grid, &g2p);
    printf("perf: mean %.1f ms/frame (last frame: p2g %.1f grid %.1f g2p %.1f ms over %d substeps)\n",
           msCount ? msSum / msCount : 0.0, p2g, grid, g2p, opt.sub);
    if (csv) fclose(csv);
    MillRef_Destroy(sim);
    return 0;
}
